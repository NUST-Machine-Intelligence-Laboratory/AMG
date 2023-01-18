import torch
import torch.nn as nn
import logging
from functools import partial
from timm.models.layers import DropPath, PatchEmbed, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import _cfg, checkpoint_filter_fn, _load_weights
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg, load_pretrained

_logger = logging.getLogger(__name__)

import numpy as np
import math

default_cfgs = {'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
        'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
        'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth'),
}


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, N):
        flops = 0
        (in_dim, mlp_dim) = self.fc1.weight.data.shape
        flops += N * in_dim * mlp_dim
        flops += N * mlp_dim * in_dim
        return flops


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_tokens=1, num_heads=8, channel=None, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = 64
        self.in_dim = dim
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, (num_heads * self.head_dim) * 3, bias=qkv_bias)
        # print(self.qkv.weight.data.shape)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_heads * self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if channel is None:
            self.channel = num_patches + num_tokens
        else:
            self.channel = channel

        token_mask = torch.ones(num_patches + num_tokens)
        self.token_index = nn.Parameter((token_mask == 1), requires_grad=False)

        self.attn = None
        self.seq_ranks = None
        self.cnt = 0

    def forward(self, x, compute_taylor_attn=False, compute_head_entro=False):
        B, N, C = x.shape
        # B,N,C --> B,N,3C --> B,N,3,h,dv --> 3,B,h,N,dv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # B,h,N,dv
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        k = k[:, :, self.token_index, :]
        v = v[:, :, self.token_index, :]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if compute_taylor_attn:
            self.attn = attn

            attn.register_hook(self.compute_rank_attn)

        if compute_head_entro:
            if self.attn is None:
                self.attn = torch.zeros(attn.shape).cuda()
            if self.attn.shape[0] == attn.shape[0]:
                self.attn += attn
                self.cnt += 1

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def compute_rank_attn(self, grad):
        values = torch.sum((grad * self.attn), dim=0, keepdim=True) \
                      .sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)[0, 0, 0, 1:].data
        # print(values.shape)
        # print('attn')
        values = values / (self.attn.size(0) * self.attn.size(1) * self.attn.size(2))
        if self.seq_ranks is None:
            self.seq_ranks = torch.zeros(grad.size(3) - 1).cuda()

        self.seq_ranks += torch.abs(values)

    def reset_rank(self):
        self.attn = None
        self.seq_ranks = None
        self.cnt = 0

    def flops(self):
        flops = 0
        total_dim = self.head_dim * self.num_heads
        # q.k.v dot x
        flops += 3 * self.channel * total_dim * self.in_dim
        # attn = q matmul k.transpose
        flops += self.channel * total_dim * self.channel
        # softmax
        flops += self.num_heads * self.channel * self.channel
        # out = attn matmul v
        flops += self.channel * total_dim * self.channel
        # self.out dot out
        flops += self.channel * total_dim * self.in_dim
        return flops


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_patches, num_tokens=1, mlp_ratio=4., channel=None, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.num_patches = num_patches + num_tokens # if channel is not None else channel
        self.attn = Attention(dim, num_patches, num_heads=num_heads, num_tokens=num_tokens,
                              channel=channel, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, compute_taylor_attn=False, compute_head_entro=False):
        # token_keep = x[:, self.attn.token_index, :]
        # token_res = x[:, ~self.attn.token_index, :]
        x = x + self.drop_path(self.attn(self.norm1(x), compute_taylor_attn, compute_head_entro))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = torch.cat([x, token_res], dim=1)
        return x

    def flops(self):
        flops = 0
        flops += self.attn.flops()
        flops += self.mlp.flops(self.num_patches)
        return flops, self.attn.flops()


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, embed_dims=None, depth=12,
                 num_heads=12, mlp_ratio=4., channels=None, qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            channels (list): number of k,v patches
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.embed_dim = embed_dim
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if channels is None:
            channels = [num_patches + self.num_tokens] * depth
        assert len(channels) == depth

        if embed_dims is None:
            embed_dims = [embed_dim] * depth
        assert len(embed_dims) == depth

        if not isinstance(num_heads, list):
            num_heads = [num_heads] * depth
        assert len(num_heads) == depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    dim=embed_dims[i], num_heads=num_heads[i], num_patches=num_patches, num_tokens=self.num_tokens,
                    mlp_ratio=mlp_ratio, channel=channels[i], qkv_bias=qkv_bias, attn_drop=attn_drop_rate,
                    drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            )

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, compute_taylor_attn=False, compute_head_entro=False):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x, compute_taylor_attn, compute_head_entro)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, compute_attn=False, compute_entro=False):
        # print("fi{}".format(x.dtype))
        x = self.forward_features(x, compute_taylor_attn=compute_attn, compute_head_entro=compute_entro)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

    def flops(self):
        # embed_dim, in_dim, fw, fh = self.patch_embed.weight.data.shape
        # num_class, _ = self.head.weight.data.shape
        # flop_embedding = self.num_patches * in_dim * embed_dim * (fw * fh)
        # flop_classify = self.num_patches * num_class * embed_dim
        # print(flop_embedding, flop_classify)
        flops = 0
        attn_f = 0
        for layer in self.blocks:
            flops += layer.flops()[0]
            attn_f += layer.flops()[1]
        return flops, attn_f
        # return self.blocks.flops()  + flop_classify


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    kwargs.pop('pretrained_cfg')

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        pretrained_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


# @register_model
# def amg_vit_base_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg(url='https://storage.googleapis.com/vit_models/augreg/'
#             'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url='https://storage.googleapis.com/vit_models/augreg/'
#                 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
#             map_location="cpu", check_hash=True
#         )
#         # load_pretrained(
#         #     model,
#         #     num_classes=1000,
#         #     in_chans=kwargs.get('in_chans', 3),
#         #     filter_fn=None,
#         #     strict=False)
#         model.load_state_dict(checkpoint["model"], strict=False)
#     return model


@register_model
def amg_vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    if 'num_heads' in kwargs:
        model_kwargs = dict(patch_size=16, depth=12, embed_dim=768, **kwargs)
    else:
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, pretrained_strict=False, **model_kwargs)
    return model

@register_model
def amg_vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (ViT-Ti/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    if 'num_heads' in kwargs:
        model_kwargs = dict(patch_size=16, depth=12, embed_dim=384, **kwargs)
    else:
        model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, pretrained_strict=False, **model_kwargs)
    return model

@register_model
def amg_vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (ViT-Ti/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    if 'num_heads' in kwargs:
        model_kwargs = dict(patch_size=16, depth=12, embed_dim=192, **kwargs)
    else:
        model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, pretrained_strict=False, **model_kwargs)
    return model