from __future__ import print_function

import os
import time
import random
import zipfile
import argparse
import copy
from itertools import chain

import timm

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from model import ClassBlock, LATransformer
from vit_model import amg_vit_base_patch16_224
from timm.scheduler import create_scheduler
from losses import DistillationLoss
from utils import save_network, update_summary, setup_device, load_check
from train import train_one_epoch, AverageMeter, freeze_all_blocks, unfreeze_blocks, validate
from prune import prune


def main():
    parser = argparse.ArgumentParser('AMG-LA training script', add_help=False)
    parser.add_argument('--name', default='la_with_lmbd_8', type=str)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--data-dir', default='/home/maojunzhu/data/Market-Pytorch/Market', type=str)
    parser.add_argument('--weight-path', default='/home/maojunzhu/pycharmprojects/LA-Transformer/model/la_oc-duke/net_best.pth')
    parser.add_argument('--n-gpu', default=1, type=int)
    parser.add_argument('--num-epochs', default=30, type=int)

    parser.add_argument("--prune_mode", type=str, default='attn', help='choose which mode to prune',
                        choices=['seq', 'attn', 'random', 'head_entropy', 'hrand', 'token_entro'])
    parser.add_argument("--prune_rate", type=float, default=0.25)
    parser.add_argument("--iter_nums", type=int, default=1)
    parser.add_argument("--final_finetune", type=int, default=1)
    # parser.add_argument("--token_settings", default=[])
    parser.add_argument("--finetune_nums", type=int, default=0, help="number of training/fine-tuning steps")
    # parser.add_argument("--prune-step", type=int, default=2, help="num of steps each layer to prune")

    parser.add_argument('--distill-type', type=str, default='soft', choices=['soft', 'none', 'hard'])
    parser.add_argument('--teacher-path', type=str,
                        default='/home/maojunzhu/pycharmprojects/LA-Transformer/model/net_best.pth')
    parser.add_argument('--distill-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distill-tau', default=1.0, type=float, help="")

    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--gamma', default=0.7, type=float)
    parser.add_argument('--decay-epochs', default=3, type=int)
    # parser.add_argument('--unfreeze-after', default=2, type=int)
    # parser.add_argument('--lr-decay', default=0.8, type=float)
    parser.add_argument('--lmbd', default=8, type=int)

    args = parser.parse_args()

    device, device_ids = setup_device(args.n_gpu)

    transform_train_list = [
        transforms.Resize((224, 224), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform_val_list = [
        transforms.Resize(size=(224, 224), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    image_datasets = {}

    image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                                   data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(args.data_dir, 'gallery'),
                                                 data_transforms['val'])
    train_loader = DataLoader(dataset=image_datasets['train'], batch_size=args.batch_size, num_workers=16,shuffle=True)
    valid_loader = DataLoader(dataset=image_datasets['val'], batch_size=args.batch_size, num_workers=16, shuffle=True)

    class_names = image_datasets['train'].classes
    print(len(class_names))

    checkpoint, channels, heads = load_check(args.weight_path)

    vit_base = timm.create_model('amg_vit_tiny_patch16_224', pretrained=False, channels=channels, num_heads=heads, num_classes=len(class_names))
    vit_base = vit_base.to(device)
    vit_base.eval()

    # Create LA Transformer
    model = LATransformer(vit_base, args.lmbd).to(device)
    # print(model.eval())

    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    teacher_model = None
    if args.distill_type != 'none':
        teacher_model = copy.deepcopy(model)

    n_parameters = sum(p.numel() for p in model.model.parameters())
    print('number of params:', n_parameters)

    # loss function
    base_criterion = nn.CrossEntropyLoss()

    criterion = DistillationLoss(
        base_criterion, teacher_model, args.distill_type, args.distill_alpha, args.distill_tau
    )

    # freeze_all_blocks(model)

    best_acc = 0.0
    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    print("training...")
    output_dir = ""
    best_acc = 0
    # name = "la_with_lmbd_{}".format(args.lmbd)

    compute_attn = False
    compute_entro = False
    if args.prune_mode == 'head_entropy':
        compute_entro = True
        num_to_prune = int(12 * 3 * args.prune_rate)
        num_per_iter = int(num_to_prune / args.iter_nums)

    elif args.prune_mode == 'attn':
        compute_attn = True
        num_patches = model.model.patch_embed.num_patches
        num_to_prune = int(num_patches * 12 * args.prune_rate)
        num_per_iter = int(num_to_prune / args.iter_nums)

    # print(num_per_iter)

    if not os.path.exists("./model/" + args.name):
        os.mkdir("./model/" + args.name)

    output_dir = "model/" + args.name

    # channels = []
    # heads = []
    for i in range(args.iter_nums):
        train_metrics = train_one_epoch(
            0, model, train_loader, criterion, device, optimizer=None,
            compute_attn=compute_attn, compute_entro=compute_entro,
            distill=True, lr_scheduler=None, saver=None)
        if args.prune_mode == 'head_entropy':
            heads, _ = taylor_prune(model.model, num_per_iter, mode='head_entropy')
        elif args.prune_mode == 'attn':
            channels = taylor_prune(model.model, num_per_iter, mode='attn')
        else:
            print('Prune nothing')

        n_parameters = sum(p.numel() for p in model.model.parameters())
        print('number of params:', n_parameters)

        model.train()
        model.to(device)

        # optimizer
        optimizer = optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)

        for epoch in range(1, args.finetune_nums+1):
            train_metrics = train_one_epoch(
                epoch, model, train_loader, criterion, device, optimizer=optimizer, distill=True)
            # eval_metrics = validate(model, valid_loader, criterion, device, epoch, distill=True)
            # update_summary(epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
            #                write_header=True)
            save_network(model, epoch, args.name, channels=channels, heads=heads)

    # scheduler
    scheduler = StepLR(optimizer, step_size=args.decay_epochs, gamma=args.gamma)
    best_acc = 0
    for epoch in range(args.final_finetune):

        # if epoch % args.unfreeze_after == 0:
        #     unfrozen_blocks += 1
        #     model = unfreeze_blocks(model, unfrozen_blocks)
        #     optimizer.param_groups[0]['lr'] *= args.lr_decay
        #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     print("Unfrozen Blocks: {}, Current lr: {}, Trainable Params: {}".format(unfrozen_blocks,
        #                                                                              optimizer.param_groups[0]['lr'],
        #                                                                              trainable_params))

        train_metrics = train_one_epoch(
            epoch, model, train_loader, criterion, device, optimizer=optimizer,
            distill=True, lr_scheduler=None, saver=None)

        # eval_metrics = validate(model, valid_loader, criterion, device, epoch, distill=True)
        scheduler.step(epoch)

        # update summary
        # update_summary(epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
        #                write_header=True)

        # deep copy the model
        # last_model_wts = model.state_dict()
        if train_metrics['train_accuracy'] > best_acc:
            best_acc = train_metrics['train_accuracy']
            save_network(model, epoch, args.name, best=True, channels=channels, heads=heads)
            print("SAVED!")
        else:
            save_network(model, epoch, args.name, channels=channels, heads=heads)
            print("SAVED!")


if __name__ == '__main__':
    main()
