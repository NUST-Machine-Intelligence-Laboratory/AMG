import torch
from timm.models import create_model


# path = '/home/maojunzhu/pycharmprojects/AMG/experiments/save/vti16-ImageNet-attn40_0_20_kd_20_20_lr1e-4_ImageNet_bs256_lr0.0001_wd0.001_220706_101355/checkpoints/best.pth'

path = '/home/maojunzhu/pycharmprojects/LA-Transformer/model/b16-attn25-head25-12-3-6-lr1e-5-wd1e-4-kd-2-20/net_best.pth'


def load_index(model, path):
    print('load index')
    if path.endswith('pth'):
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint['model']
        # print(state_dict.keys())
        # assert len(state_dict['transformer.encoder_layers.0.attn.index']) == \
        #        len(model.blocks[0].attn.token_index), 'token number must be the same'

        index_list = []
        for (name, param) in state_dict.items():
            if 'index' in name:
                index_list.append(param)
        for i, module in enumerate(model.blocks):
            module.attn.token_index.data = index_list[i]

        print(index_list)
        channels = [int(index.sum()) for index in index_list]

        print(channels)

        return channels

        # print(model.blocks[-1].attn.token_index)

    else:
        assert 0, 'only support .pth weight'


if __name__ == '__main__':
    model = create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=1000,
        channels=None,
        # embed_dims=emb_dims,
        # num_heads=num_heads,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    load_index(model, path)
