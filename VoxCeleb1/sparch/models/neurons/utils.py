import logging
import torch
import numpy as np
import random
import math
import os
import shutil
import sys
sys.path.append("/home/zeyang/Project/Spiking-LEAF/sparch/models/neurons")
import base

def count_parameters(model):
    param_sum = 0
    for p in model.parameters():
        if p.requires_grad:
            print(p.size(), p.numel())
            param_sum = param_sum + p.numel()
    return param_sum

def setup_logging(log_file='log.txt'):
    """Setup logging configuration"""
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def reset_states(model):
    for m in model.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, base.MemoryModule):
                print(f'Trying to call `reset()` of {m}, which is not base.MemoryModule')
            m.reset()

def temporal_loop_stack(input, op):
    output_current = []
    for t in range(input.size(0)):
        output_current.append(op(input[t]))
    return torch.stack(output_current, 0)

def seed_everything(seed=1234, is_cuda=False):
    """Some configurations for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if is_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(seed)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.5 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, dirname, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dirname, filename))
    if is_best:
        shutil.copyfile(os.path.join(dirname, filename), os.path.join(dirname, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def onehot(inp, nlable):
    onehot_x = inp.new_zeros(inp.shape[0], nlable+2)
    return onehot_x.scatter_(1, inp.long(), 1)