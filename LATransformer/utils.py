import os
import csv
import torch
import numpy as np
from collections import OrderedDict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


def save_network(network, epoch_label, name, best=False, channels=None, heads=None):
    if best:
        save_filename = 'net_best.pth'
    else:
        save_filename = 'net_last.pth'

    state = {
        'state_dict': network.cpu().state_dict(),
        'channels': channels,
        'heads': heads
    }
    save_path = os.path.join('./model', name, save_filename)
    torch.save(state, save_path)
    
    if torch.cuda.is_available():
        network.cuda()


def load_check(path):
    checkpoint = torch.load(path)
    if 'state_dict' in checkpoint.keys():
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if 'channels' in checkpoint.keys() and checkpoint['channels'] is not None and len(checkpoint['channels']) == 12:
        channels = np.asarray(checkpoint['channels'])
        print('channels:', channels)
    else:
        channels = None
    if 'heads' in checkpoint.keys() and checkpoint['heads'] is not None and len(checkpoint['heads']) == 12:
        heads = checkpoint['heads']
        print(heads)
    else:
        heads = None
    return state_dict, channels, heads

        
def get_id(img_path, mode='market'):
    camera_id = []
    labels = []
    if mode == 'market' or mode == 'o-duke':
        for path, v in img_path:
            #filename = path.split('/')[-1]
            filename = os.path.basename(path)
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2]=='-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))

    elif mode == 'o-reid':
        for path, v in img_path:
            filename = os.path.basename(path)
            label = filename[0:3]
            labels.append(int(label))
    else:
        assert 0, 'no supported dataset'
    return camera_id, labels