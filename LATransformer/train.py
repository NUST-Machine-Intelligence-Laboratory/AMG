from __future__ import print_function

import os
import time
import random
import zipfile
import argparse
from itertools import chain

import timm

from collections import OrderedDict

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

from vit_model import amg_vit_base_patch16_224
from model import ClassBlock, LATransformer
from utils import save_network, update_summary, setup_device


def freeze_all_blocks(model):
    frozen_blocks = 12
    for block in model.model.blocks[:frozen_blocks]:
        for param in block.parameters():
            param.requires_grad=False


def unfreeze_blocks(model, amount=1):
    for block in model.model.blocks[11 - amount:]:
        for name, param in block.named_parameters():
            if 'index' not in name:
                param.requires_grad = True
    return model


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
        epoch, model, loader, loss_fn, device, optimizer=None,
        compute_attn=False, compute_entro=False, distill=False,
        lr_scheduler=None, saver=None, output_dir='',
        loss_scaler=None, model_ema=None, mixup_fn=None):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    epoch_accuracy = 0
    epoch_loss = 0
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    running_loss = 0.0
    running_corrects = 0.0

    # print('compute_attn:', compute_attn)

    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        data_time_m.update(time.time() - end)
        if optimizer is not None:
            optimizer.zero_grad()
        output = model(data, compute_attn=compute_attn, compute_entro=compute_entro)
        # print(output.shape)
        score = 0.0
        sm = nn.Softmax(dim=1)
        for k, v in output.items():
            score += sm(output[k])
        # score = sm(output)
        _, preds = torch.max(score.data, 1)

        loss = 0.0
        for k, v in output.items():
            if distill:
                loss += loss_fn(data, output[k], k, target)
            else:
                loss += loss_fn(output[k], target)
        # loss = loss_fn(output, target)
        loss.backward()

        if optimizer is not None:
            optimizer.step()

        batch_time_m.update(time.time() - end)

        #         print(preds, target.data)
        acc = (preds == target.data).float().mean()

        #         print(acc)
        epoch_loss += loss / len(loader)
        epoch_accuracy += acc / len(loader)
        #         if acc:
        #             print(acc, epreds, target.data)
        if i % 10 == 0:
            print(f"Epoch : {epoch + 1} - iter:{i} / {len(loader)} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}")
            # break
    # print()

    return OrderedDict([('train_loss', epoch_loss.data.item()), ("train_accuracy", epoch_accuracy.data.item())])


def validate(model, loader, loss_fn, device, epoch, distill=False):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    epoch_accuracy = 0
    epoch_loss = 0
    end = time.time()
    last_idx = len(loader) - 1

    running_loss = 0.0
    running_corrects = 0.0

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):

            input, target = input.to(device), target.to(device)

            output = model(input)

            score = 0.0
            sm = nn.Softmax(dim=1)
            for k, v in output.items():
                score += sm(output[k])
            _, preds = torch.max(score.data, 1)

            loss = 0.0
            for k, v in output.items():
                if distill:
                    loss += loss_fn(input, output[k], k, target)
                else:
                    loss += loss_fn(output[k], target)

            batch_time_m.update(time.time() - end)
            acc = (preds == target.data).float().mean()
            epoch_loss += loss / len(loader)
            epoch_accuracy += acc / len(loader)

            print(f"Epoch : {epoch + 1} - val_loss : {epoch_loss:.4f} - val_acc: {epoch_accuracy:.4f}", end="\r")
    print()
    metrics = OrderedDict([('val_loss', epoch_loss.data.item()), ("val_accuracy", epoch_accuracy.data.item())])

    return metrics


def main():
    parser = argparse.ArgumentParser('AMG-LA training script', add_help=False)
    parser.add_argument('--name', default='la_with_lmbd_8', type=str)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--data-dir', default='../../data/Market-Pytorch/Market', type=str)
    parser.add_argument('--n-gpu', default=1, type=int)
    parser.add_argument('--num-epochs', default=30, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--gamma', default=0.7, type=float)
    parser.add_argument('--unfreeze-after', default=2, type=int)
    parser.add_argument('--lr-decay', default=0.8, type=float)
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
    # data_dir = "data/Market-Pytorch/Market/"

    image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                                   data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(args.data_dir, 'query'),
                                                 data_transforms['val'])
    train_loader = DataLoader(dataset=image_datasets['train'], batch_size=args.batch_size, num_workers=16, shuffle=True)
    valid_loader = DataLoader(dataset=image_datasets['val'], batch_size=args.batch_size, shuffle=False)
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
    #                                              shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
    #               for x in ['train', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(len(class_names))

    channels = []
    vit_base = timm.create_model('amg_vit_small_patch16_224', pretrained=True,
                                 # checkpoint_path='/home/maojunzhu/pycharmprojects/TransReID/weight/jx_vit_base_p16_224-80ecf9dd.pth',
                                 num_classes=len(class_names))
    # channels = load_index(vit_base, args.index_path)
    # print(channels)
    vit_base = vit_base.to(device)
    vit_base.eval()



    # Create LA Transformer
    model = LATransformer(vit_base, args.lmbd).to(device)
    # model = vit_base
    print(model.eval())

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay=5e-4, lr=args.lr)

    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    freeze_all_blocks(model)

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

    if not os.path.exists("./model/" + args.name):
        os.mkdir("./model/" + args.name)

    output_dir = "model/" + args.name
    unfrozen_blocks = 0

    for epoch in range(args.num_epochs):

        if epoch % args.unfreeze_after == 0:
            unfrozen_blocks += 1
            model = unfreeze_blocks(model, unfrozen_blocks)
            optimizer.param_groups[0]['lr'] *= args.lr_decay
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Unfrozen Blocks: {}, Current lr: {}, Trainable Params: {}".format(unfrozen_blocks,
                                                                                     optimizer.param_groups[0]['lr'],
                                                                                     trainable_params))

        train_metrics = train_one_epoch(
            epoch, model, train_loader, criterion, device, optimizer=optimizer,
            lr_scheduler=None, saver=None)

        # eval_metrics = validate(model, valid_loader, criterion, device, epoch)

        # update summary
        # update_summary(epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
        #                write_header=True)

        # deep copy the model
        last_model_wts = model.state_dict()
        if train_metrics['train_accuracy'] > best_acc:
            best_acc = train_metrics['train_accuracy']
            save_network(model, epoch, args.name, best=True, channels=channels)
            print("SAVED!")
        else:
            save_network(model, epoch, args.name, channels=channels)
            print("SAVED!")


if __name__ == '__main__':
    main()
