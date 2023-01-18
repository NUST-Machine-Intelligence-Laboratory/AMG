import numpy as np
import torch
import torch.nn as nn
# from model import VisionTransformer, SelfAttention
# from timm.models.vision_transformer import VisionTransformer, Attention
# from prune_by_layer import get_new_attn, get_new_out
from vit_model import Attention


def normalize_ranks_per_layer(layer_ranks):
    for i in range(len(layer_ranks)):
        v = torch.abs(layer_ranks[i])
        v = v / torch.sqrt(torch.sum(v * v))
        layer_ranks[i] = v
    return layer_ranks


def get_new_qkv(module, head, head_dim, idx):
    # print(module.weight.data.size())
    in_dim = module.weight.data.size(1)
    # print('head:{}  in_dim:{}'.format(head, in_dim))
    new_out_dim = len(idx) * head_dim

    weight = module.weight.data.reshape(3, head, head_dim, in_dim)
    bias = module.bias.data.reshape(3, head, head_dim)

    new_weight = weight[:, idx, :, :].clone()
    new_bias = bias[:, idx, :].clone()

    new_qkv = nn.Linear(in_dim, new_out_dim)
    new_qkv.weight.data = new_weight.reshape(-1, in_dim)
    new_qkv.bias.data = new_bias.reshape(-1)

    return new_qkv


def get_new_proj(module, head, idx):
    out_dim, in_dim = module.weight.data.size()
    head_dim = in_dim // head
    new_in_dim = len(idx) * head_dim

    weight = module.weight.data.reshape(out_dim, head, head_dim)
    bias = module.bias.data

    new_weight = weight[:, idx, :].clone()

    new_proj = nn.Linear(new_in_dim, out_dim)
    new_proj.weight.data = new_weight.reshape(out_dim, -1)
    new_proj.bias.data = bias.clone()

    return new_proj


def get_new_conv(model, idx):
    num_seq = len(idx)
    new_conv = nn.Conv2d(num_seq, num_seq, kernel_size=1, stride=1, bias=None, groups=num_seq)
    new_weight = model.weight.data[idx, :, :, :].clone()
    new_conv.weight.data = new_weight.clone()
    return new_conv


def prune_tokens(model, masks):
    model.cpu()
    layer = 0
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            # idx = np.squeeze(np.argwhere(np.asarray(masks[layer])))
            # # add class token to index
            # idx = np.concatenate([[1], idx + 1], axis=0)
            #
            # module.index = nn.Parameter(torch.tensor(idx), requires_grad=False)
            mask = masks[layer]
            module.token_index = nn.Parameter(torch.cat([torch.tensor([True]), mask], dim=0), requires_grad=False)
            layer += 1
            # for i in range(layer, len(masks)):
            #     masks[i] = torch.cat([masks[i][mask], masks[i][~mask]], dim=0)
            # print(module.index)
            module.reset_rank()


def prune_attention(model, masks):
    model.cpu()
    layer = 0
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            idx = np.argwhere(np.asarray(masks[layer]))
            # print(idx)

            new_qkv = get_new_qkv(module.qkv, module.num_heads, module.head_dim, idx)
            new_proj = get_new_proj(module.proj, module.num_heads, idx)

            module.qkv = new_qkv
            module.proj = new_proj
            module.num_heads = len(idx)

            module.reset_rank()

            layer += 1


def prune_token_by_layer(model, layer, layer_num):
    # print(layer)
    for name, module in model.named_modules():
        if isinstance(module, Attention) and '.{}.attn'.format(layer) in name:
            # print(name)
            layer_rank = module.seq_ranks.cpu().clone()
            smallest = np.sort(np.asarray(layer_rank))[-layer_num]
            # print(smallest)
            mask = layer_rank > smallest
            # print(mask)
            # idx = np.argwhere(np.asarray(mask))
            module.token_index = nn.Parameter(torch.cat([torch.tensor([True]), mask], dim=0), requires_grad=False)
            # print(module.token_index)
            module.reset_rank()
            break


def prune(model, nums, mode='head_entropy'):
    layer_ranks = []
    attn_maps = []
    iter_cnts = []
    if mode == 'head':
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                layer_ranks.append(module.head_ranks)

    elif mode == 'attn':
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                layer_ranks.append(module.seq_ranks)
                # print(len(layer_ranks))
    elif mode == 'random':
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                num = module.index.data.shape[0]
                layer_ranks.append(torch.rand(num))
    elif mode == 'head_entropy':
        for name, module in model.named_modules():
            if isinstance(module, Attention):
                attn_maps.append(module.attn)
                iter_cnts.append(module.cnt)
    else:
        print('no such mode')
        assert 0
    normalize_ranks_per_layer(layer_ranks)
    # print(layer_ranks)
    layer_ranks = np.asarray([np.asarray(layer_rank.cpu()) * (1 - 0.03 * layer)
                              for layer, layer_rank in enumerate(layer_ranks)])
    if mode == 'random':
        layers = len(layer_ranks)
        for layer in np.random.randint(layers, size=nums):
            length = len(layer_ranks[layer])
            id = np.random.randint(length)
            layer_ranks[layer][id] = 0
        masks = np.asarray([layer_rank != 0 for layer_rank in layer_ranks])
    # print(np.hstack(layer_ranks))
    elif mode == 'head_entropy':
        heads_entropy = []
        for attn_map, cnt in zip(attn_maps, iter_cnts):
            b, h, Nq, Nk = attn_map.shape
            attn_map = (attn_map.sum(dim=0) / (b * cnt)).data
            # print(attn_map.shape)
            entropy_map = torch.zeros(attn_map.shape)
            for ki in range(Nk):
                entropy_map[:, :, ki] = -attn_map[:, :, ki] * torch.log2(attn_map[:, :, ki])
            head_entropy = entropy_map.sum(dim=2).mean(dim=1)
            heads_entropy.append(np.asarray(head_entropy))
        # normalize_ranks_per_layer(heads_entropy)
        # print(heads_entropy)
        # print(heads_entropy)

        cut_layers = []
        alpha = 1
        for layer, head_entropy in enumerate(heads_entropy):
            if len(head_entropy) > 1:
                head_entropy *= alpha
                cut_layers.append(head_entropy)
                alpha += 0.02
        smallest = np.sort(np.hstack(cut_layers))[-nums]
        masks = []
        for head_entropy in heads_entropy:
            if len(head_entropy) <= 1:
                masks.append(head_entropy == head_entropy)
            else:
                masks.append(head_entropy < smallest)
        # print(masks)

    else:
        smallest = np.sort(np.hstack(layer_ranks))[nums]
        # print(smallest)
        masks = torch.tensor([layer_rank >= smallest for layer_rank in layer_ranks])
    # print(masks)
    if mode == 'head' or mode == 'head_entropy':
        prune_attention(model, masks)
    # for i in range(12):
    #     print(model.transformer.encoder_layers[i].attn.query.weight.data.shape)
        head_sets = [mask.sum() for mask in masks]
        dim_sets = [head_set * 64 for head_set in head_sets]
        print(head_sets)
        return head_sets, dim_sets

    if mode == 'attn' or mode == 'random':
        prune_tokens(model, masks)
        # for i in range(12):
        #     print(model.transformer.encoder_layers[i].attn.index.data.shape)
        channels = [mask.sum() + 1 for mask in masks]
        print(channels)
        return channels

