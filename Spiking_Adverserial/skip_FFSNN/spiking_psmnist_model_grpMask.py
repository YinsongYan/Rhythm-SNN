import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from skip_FFSNN.Hyperparameters import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algo = args.algo
thresh = args.thresh
lens = args.lens
decay = args.decay

output_size = args.out_size
input_size = args.in_size
cfg_fc = args.fc
skip_length = args.skip_length

# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update_skip(ops, x, mem, spike, mask):
    if algo == 'STBP':
        #temp = (ops(x) * mask).detach().cpu().numpy()
        mem = mem * decay * (1. - spike) + ops(x) * mask
    else:
        mem = mem.detach() * decay * (1. - spike.detach()) + ops(x) * mask  # STOP the gradient for TD
    spike = act_fun(mem) * mask # act_fun : approximation firing function
    return mem, spike

def mem_update_skip_woDecay(ops, x, mem, spike, mask):
    #temp = (ops(x) * mask).detach().cpu().numpy()
    mask = mask.expand(mem.size(0), -1)
    pre_mem = mem
    mem = mem * decay * (1. - spike) + ops(x)
    #tmp1 = (mem).detach().cpu().numpy()
    mem = torch.where(mask==0, pre_mem, mem)
    #tmp2 = (mem).detach().cpu().numpy()
    spike = act_fun(mem) * mask
    return mem, spike

def mem_update_pool(opts, x, mem, spike):
    if algo == 'STBP':
        mem = mem * (1. - spike) + opts(x, 2)
    else:
        mem = mem.detach() * (1. - spike.detach()) + opts(x, 2) # SDBP
    spike = act_fun(mem)
    return mem, spike

class FFSNN(nn.Module):

    def __init__(self):
        super(FFSNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.fc4 = nn.Linear(cfg_fc[2], output_size)

    def create_mask(self, dim=128, cycle=0):
        # n_group [2, 4, 6] [256/2=128, 513/3=171, 256/4=64]
        group_neu = int(dim/cycle)
        T = 784//self.input_size
        mask_ = []

        mask = None
        for group in range(cycle):
            mask_ = []
            for t in range(T):
                if t%cycle == 0:
                    mask_.append(1)
                else:
                    mask_.append(0)
            mask_ = torch.tensor(mask_)
            mask_ = mask_.expand(group_neu, -1)
            mask_ = torch.roll(mask_, group)
            if mask is None:
                mask = mask_
            else:
                mask = torch.cat((mask, mask_))
        #tmp1 = (mask).detach().numpy()
        return mask.to(device) # [H, T]

    def forward(self, input):
        time_window = 784//input_size
        N = input.size(0)

        h1_mem = h1_spike = torch.zeros(N, cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(N, cfg_fc[1], device=device)
        h3_mem = h3_spike = torch.zeros(N, cfg_fc[2], device=device)
        h4_mem = h4_spike = output_sum = torch.zeros(N, output_size, device=device)

        mask1 = self.create_mask(cfg_fc[0], skip_length[0])
        mask2 = self.create_mask(cfg_fc[1], skip_length[1])
        mask3 = self.create_mask(cfg_fc[2], skip_length[2])

        input = np.squeeze(input)
        input = input.view(N, input_size, -1)
        for step in range(time_window):   # input [N, 28, T]
            input_x = input[:, :, step]

            h1_mem, h1_spike = mem_update_skip_woDecay(self.fc1, input_x, h1_mem, h1_spike, mask1[:, step]) #mask[step] same shape as mem
            h2_mem, h2_spike = mem_update_skip_woDecay(self.fc2, h1_spike, h2_mem, h2_spike, mask2[:, step])
            h3_mem, h3_spike = mem_update_skip_woDecay(self.fc3, h2_spike, h3_mem, h3_spike, mask3[:, step])
            h4_mem = self.fc4(h3_spike)

            output_sum = output_sum + h4_mem # Accumulate mem of all time steps

        outputs = output_sum / time_window
        #outputs = h4_mem

        return outputs, None #n_nonzeros/n_neurons


