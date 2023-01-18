from __future__ import print_function

import os

import argparse

import timm
import torch
import torch.nn as nn
import faiss
import numpy as np

from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from train import validate
from vit_model import amg_vit_base_patch16_224
from model import ClassBlock, LATransformer, LATransformerTest
from utils import save_network, update_summary, get_id, setup_device, load_check
from metrics import rank1, rank5, rank10, calc_map

activation = {}


def search(query: str, index, k=1):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    return top_k


def get_activation(name):

    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def extract_feature(model, dataloaders, device):
    features = torch.FloatTensor()
    count = 0
    idx = 0
    total = len(dataloaders)

    for i, (img, label) in enumerate(dataloaders):

        img, label = img.to(device), label.to(device)

        output = model(img)
        # print(output.shape)

        n, c, h, w = img.size()

        count += n
        features = torch.cat((features, output.detach().cpu()), 0)
        idx += 1
        if i % 10 == 0:
            print(f"iter : {i} / {total}", end="\r")
    print()
    return features


def test():
    parser = argparse.ArgumentParser('AMG-LA test script', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--dataset', default='market', type=str, choices=['market', 'o-reid', 'o-duke'])
    parser.add_argument('--data-dir', default='', type=str)
    parser.add_argument('--weight-path', default='')
    parser.add_argument('--n-gpu', default=1, type=int)
    parser.add_argument('--gamma', default=0.7, type=float)
    # parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    device, device_ids = setup_device(args.n_gpu)

    checkpoint, channels, heads = load_check(args.weight_path)

    # channels = [149] * 12
    # print(checkpoint.keys())
    # heads = [int(dim/64) for dim in heads]
    num_classes = checkpoint['model.head.weight'].shape[0]
    # num_classes = 702
    print(num_classes)

    vit_base = timm.create_model('amg_vit_tiny_patch16_224',
                                 pretrained=False, channels=channels, num_heads=heads, num_classes=num_classes)
    # vit_base.load_state_dict(checkpoint, strict=False)
    vit_base = vit_base.to(device)

    print('flops:', vit_base.flops()[0], 'attn_flops:', vit_base.flops()[1])

    # Create La-Transformer
    model = LATransformerTest(vit_base, lmbd=8).to(device)

    msa_param = 0
    total_param = 0
    for name, param in model.named_parameters():
        if 'attn' in name:
            msa_param += param.numel()
        total_param += param.numel()

    print('Total:{} MSA:{}'.format(total_param, msa_param))
    # Load LA-Transformer
    name = "la_with_lmbd_8"
    # save_path = os.path.join('./model', name, 'net_best.pth')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    transform_query_list = [
        transforms.Resize((224, 224), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform_gallery_list = [
        transforms.Resize(size=(224, 224), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    data_transforms = {
        'query': transforms.Compose(transform_query_list),
        'gallery': transforms.Compose(transform_gallery_list),
    }

    image_datasets = {}

    image_datasets['query'] = datasets.ImageFolder(os.path.join(args.data_dir, 'query'),
                                                   data_transforms['query'])
    image_datasets['gallery'] = datasets.ImageFolder(os.path.join(args.data_dir, 'gallery'),
                                                     data_transforms['gallery'])
    query_loader = DataLoader(dataset=image_datasets['query'], batch_size=args.batch_size, num_workers=16, shuffle=False)
    gallery_loader = DataLoader(dataset=image_datasets['gallery'], batch_size=args.batch_size, num_workers=16, shuffle=False)

    class_names = image_datasets['query'].classes
    print(len(class_names))

    # criterion = nn.CrossEntropyLoss()
    # eval_metrics = validate(model, valid_loader, criterion, device, 0)

    # Extract Query Features
    query_feature = extract_feature(model, query_loader, device)

    # Extract Gallery Features
    gallery_feature = extract_feature(model, gallery_loader, device)

    # Retrieve labels
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path, args.dataset)
    query_cam, query_label = get_id(query_path, args.dataset)

    # print(len(gallery_label))
    # print(len(query_label))
    # assert 0

    concatenated_query_vectors = []
    print('Total query:', len(query_feature))
    for i, query in enumerate(query_feature):
        # print(query.shape)
        fnorm = torch.norm(query, p=2, dim=1, keepdim=True) * np.sqrt(14)

        query_norm = query.div(fnorm.expand_as(query))

        concatenated_query_vectors.append(query_norm.view((-1)))  # 14*768 -> 10752

    concatenated_gallery_vectors = []
    print('Total gallery:', len(gallery_feature))
    for i, gallery in enumerate(gallery_feature):
        fnorm = torch.norm(gallery, p=2, dim=1, keepdim=True) * np.sqrt(14)

        gallery_norm = gallery.div(fnorm.expand_as(gallery))

        concatenated_gallery_vectors.append(gallery_norm.view((-1)))  # 14*768 -> 10752

    index = faiss.IndexIDMap(faiss.IndexFlatIP(model.model.embed_dim * 14))

    index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]), np.array(gallery_label))

    rank1_score = 0
    rank5_score = 0
    rank10_score = 0
    ap = 0
    count = 0
    for query, label in zip(concatenated_query_vectors, query_label):
        count += 1
        # label = label
        output = search(query, index, k=10)

        rank1_score += rank1(label, output)
        rank5_score += rank5(label, output)
        rank10_score += rank10(label, output)
        print("Correct: {}, Total: {}, Incorrect: {}".format(rank1_score, count, count - rank1_score), end="\r")
        ap += calc_map(label, output)

    print("Rank1: {}, Rank5: {}, Rank10: {}, mAP: {}".format(rank1_score / len(query_feature),
                                                             rank5_score / len(query_feature),
                                                             rank10_score / len(query_feature),
                                                             ap / len(query_feature)))


if __name__ == '__main__':
    test()
