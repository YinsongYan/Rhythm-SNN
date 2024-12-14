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
skip_length_min = args.skip_length_min

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
def mem_update(ops, x, mem, spike, decay): # ops weight shape [32, 1, 3, 3], x [250, 1, 28, 28], mem [250, 32, 28, 28], spike [250, 32, 28, 28]
    mem = mem * decay * (1. - spike) + ops(x)   # mem: AddBackward, spike: ActFunBackward
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

def mem_update_skip(ops, x, mem, spike, mask):
    mem = mem * decay * (1. - spike) + ops(x) * mask
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

class mem_skip_update(nn.Module):
    def __init__(self):
        super(mem_skip_update, self).__init__()
    def forward(self, ops, x, mem, spike, mask):
        mask = mask.expand(mem.size(0), -1)
        pre_mem = mem
        mem = mem * decay * (1. - spike) + ops(x)
        mem = torch.where(mask == 0, pre_mem, mem)
        spike = act_fun(mem) * mask
        return mem, spike

def mem_update_pool(opts, x, mem, spike):
    mem = mem * (1. - spike) + opts(x, 2)
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
        T = 784//self.input_size
        mask_ = []#torch.ones(N, T, cfg_fc[0], device=device)
        for t in range(T):
            if t%cycle == 0:
                mask_.append(1)
            else:
                mask_.append(0)
        mask_ = torch.tensor(mask_)

        mask = []
        for n in range(dim):
            mask.append(torch.roll(mask_, n))
        mask = torch.stack(mask)
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

class FFSNN_v2(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    """
    def __init__(self, in_size=8, bias=True):
        super(FFSNN_v2, self).__init__()

        self.input_size = in_size
        self.stride = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, cfg_fc[0], bias=bias)
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], bias=bias)
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], bias=bias)
        self.fc4 = nn.Linear(cfg_fc[2], output_size, bias=bias)

    def create_mask(self, dim=128, cycle=0):
        T = 784//self.stride
        mask_ = []#torch.ones(N, T, cfg_fc[0], device=device)
        for t in range(T):
            if t%cycle == 0:
                mask_.append(1)
            else:
                mask_.append(0)
        mask_ = torch.tensor(mask_)

        mask = []
        for n in range(dim):
            mask.append(torch.roll(mask_, n))
        mask = torch.stack(mask)
        return mask.to(device) # [H, T]

    def forward(self, input):
        time_window = 784//self.stride
        N = input.size(0)

        h1_mem = h1_spike = h1_spike_sums = torch.zeros(N, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_spike_sums = torch.zeros(N, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_spike_sums = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, output_size, device=device)

        mask1 = self.create_mask(cfg_fc[0], skip_length[0])
        mask2 = self.create_mask(cfg_fc[1], skip_length[1])
        mask3 = self.create_mask(cfg_fc[2], skip_length[2])

        input = np.squeeze(input)
        input = input.view(N, -1) #[N, 784]
        for step in range(time_window):
            start_idx = step * self.stride
            if start_idx < (time_window - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            h1_mem, h1_spike = mem_update_skip_woDecay(self.fc1, input_x, h1_mem, h1_spike, mask1[:, step]) #mask[step] same shape as mem
            h2_mem, h2_spike = mem_update_skip_woDecay(self.fc2, h1_spike, h2_mem, h2_spike, mask2[:, step])
            h3_mem, h3_spike = mem_update_skip_woDecay(self.fc3, h2_spike, h3_mem, h3_spike, mask3[:, step])
            h4_mem = self.fc4(h3_spike)

            output_sum = output_sum + h4_mem # Accumulate mem of all time steps
            h1_spike_sums += h1_spike
            h2_spike_sums += h2_spike
            h3_spike_sums += h3_spike

        outputs = output_sum / time_window
        #outputs = h4_mem
        layer_fr = [h1_spike_sums.sum() / (torch.numel(h1_spike) * time_window),
                    h2_spike_sums.sum() / (torch.numel(h2_spike) * time_window),
                    h3_spike_sums.sum() / (torch.numel(h3_spike) * time_window)]
        layer_fr = torch.tensor(layer_fr)
        hidden_spk = [h1_spike_sums / time_window, h2_spike_sums / time_window, h3_spike_sums / time_window]
        return outputs, hidden_spk, layer_fr #n_nonzeros/n_neurons

class FFSNN_v2_bb(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    """
    def __init__(self, in_size=8):
        super(FFSNN_v2_bb, self).__init__()
        self.input_size = in_size
        self.stride = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.fc4 = nn.Linear(cfg_fc[2], output_size)

    def forward(self, input):
        time_window = 784//self.stride
        N = input.size(0)

        h1_mem = h1_spike = torch.zeros(input.size(0), cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(input.size(0), cfg_fc[1], device=device)
        h3_mem = h3_spike = torch.zeros(input.size(0), cfg_fc[2], device=device)
        h4_mem = h4_spike = output_sum = torch.zeros(input.size(0), output_size, device=device)

        input = np.squeeze(input)
        input = input.view(N, -1)  # [N, 784]
        for step in range(time_window):   # input [N, 28, T]

            if algo == 'noTD':  # if noTD, reset mem potential every time step
                h1_mem = h1_spike = torch.zeros(input.size(0), cfg_fc[0], device=device)
                h2_mem = h2_spike = torch.zeros(input.size(0), cfg_fc[1], device=device)
                h3_mem = h3_spike = torch.zeros(input.size(0), cfg_fc[2], device=device)

            start_idx = step * self.stride
            if start_idx < (time_window - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            h1_mem, h1_spike = mem_update(self.fc1, input_x, h1_mem, h1_spike, decay)#_cfg[0])
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike, decay)#_cfg[1])
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike, decay)#_cfg[2])
            #h4_mem, h4_spike = mem_update(self.fc4, h3_spike, h4_mem, h4_spike)
            h4_mem = self.fc4(h3_spike)
            output_sum = output_sum + h4_mem # Accumulate mem of all time steps


        outputs = output_sum / time_window
        #outputs = h4_mem

        return outputs, None, None


class FFSNN_mix(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    Mix skip_length within layers
    """
    def __init__(self, in_size=8, bias=True):
        super(FFSNN_mix, self).__init__()

        self.input_size = in_size
        self.stride = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, cfg_fc[0], bias=bias)
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], bias=bias)
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], bias=bias)
        self.fc4 = nn.Linear(cfg_fc[2], output_size, bias=bias)

        self.mask1 = self.create_mix_mask(cfg_fc[0], skip_length[0])
        self.mask2 = self.create_mix_mask(cfg_fc[1], skip_length[1])
        self.mask3 = self.create_mix_mask(cfg_fc[2], skip_length[2])

    def create_mix_mask(self, dim=128, max_cycle=0):
        T = 784//self.stride
        mask_ = []
        cnt = 1
        cycle = 1
        for t in range(1, T+1):
            if t % cycle == 0:
                mask_.append(1)
                cnt = cnt + 1 if (cnt < max_cycle) else 2
                cycle = cycle + cnt
            else:
                mask_.append(0)
        mask_ = torch.tensor(mask_)
        #tmp = mask_.detach().numpy()

        mask = []
        for n in range(dim):
            mask.append(torch.roll(mask_, n))
        mask = torch.stack(mask)
        #tmp2 = mask.detach().cpu().numpy()
        return mask.to(device)

    def forward(self, input):
        time_window = 784//self.stride
        N = input.size(0)

        h1_mem = h1_spike = torch.zeros(N, cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(N, cfg_fc[1], device=device)
        h3_mem = h3_spike = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, output_size, device=device)

        input = np.squeeze(input)
        input = input.view(N, -1) #[N, 784]
        for step in range(time_window):
            start_idx = step * self.stride
            if start_idx < (time_window - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            h1_mem, h1_spike = mem_update_skip_woDecay(self.fc1, input_x, h1_mem, h1_spike, self.mask1[:, step]) #mask[step] same shape as mem
            h2_mem, h2_spike = mem_update_skip_woDecay(self.fc2, h1_spike, h2_mem, h2_spike, self.mask2[:, step])
            h3_mem, h3_spike = mem_update_skip_woDecay(self.fc3, h2_spike, h3_mem, h3_spike, self.mask3[:, step])
            h4_mem = self.fc4(h3_spike)

            output_sum = output_sum + h4_mem # Accumulate mem of all time steps

        outputs = output_sum / time_window
        #outputs = h4_mem

        return outputs, None #n_nonzeros/n_neurons

class FFSNN_mix_v2(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    Mix skip_length within layers
    """
    def __init__(self, in_size=8, bias=True):
        super(FFSNN_mix_v2, self).__init__()
        self.input_size = in_size
        self.stride = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, cfg_fc[0], bias=bias)
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], bias=bias)
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], bias=bias)
        self.fc4 = nn.Linear(cfg_fc[2], output_size, bias=bias)

        self.mask1 = self.create_mix_mask(cfg_fc[0], skip_length[0])
        self.mask2 = self.create_mix_mask(cfg_fc[1], skip_length[1])
        self.mask3 = self.create_mix_mask(cfg_fc[2], skip_length[2])

    def create_mix_mask(self, dim=128, max_cycle=0):
        T = 784 // self.stride
        mask_cyc = []
        for cycle in range(2, max_cycle+1):
            mask_ = []
            for t in range(T):
                if t % cycle == 0:
                    mask_.append(1)
                else:
                    mask_.append(0)
            mask_ = torch.tensor(mask_)
            #tmp1 = mask_.detach().numpy()
            mask_cyc.append(mask_)
        mask_cyc = torch.stack(mask_cyc)
        #tmp2 = mask_cyc.detach().numpy()

        mask = mask_cyc
        for n in range(1, dim//(max_cycle-1) + 1):
            mask = torch.cat((mask, torch.roll(mask_cyc, n, 1)), 0)
            #tmp3 = mask.detach().numpy()
        return mask[: dim].to(device)  # [H, T]

    def forward(self, input):
        time_window = 784//self.stride
        N = input.size(0)

        h1_mem = h1_spike = h1_spike_sums = torch.zeros(N, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_spike_sums = torch.zeros(N, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_spike_sums = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, output_size, device=device)

        input = np.squeeze(input)
        input = input.view(N, -1) #[N, 784]
        for step in range(time_window):
            start_idx = step * self.stride
            if start_idx < (time_window - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            h1_mem, h1_spike = mem_update_skip_woDecay(self.fc1, input_x, h1_mem, h1_spike, self.mask1[:, step]) #mask[step] same shape as mem
            h2_mem, h2_spike = mem_update_skip_woDecay(self.fc2, h1_spike, h2_mem, h2_spike, self.mask2[:, step])
            h3_mem, h3_spike = mem_update_skip_woDecay(self.fc3, h2_spike, h3_mem, h3_spike, self.mask3[:, step])
            h4_mem = self.fc4(h3_spike)

            output_sum = output_sum + h4_mem # Accumulate mem of all time steps
            h1_spike_sums += h1_spike
            h2_spike_sums += h2_spike
            h3_spike_sums += h3_spike

        outputs = output_sum / time_window
        #outputs = h4_mem
        layer_fr = [h1_spike_sums.sum() / (torch.numel(h1_spike) * time_window),
                    h2_spike_sums.sum() / (torch.numel(h2_spike) * time_window),
                    h3_spike_sums.sum() / (torch.numel(h3_spike) * time_window)]
        layer_fr = torch.tensor(layer_fr)
        hidden_spk = [h1_spike_sums / time_window, h2_spike_sums / time_window, h3_spike_sums / time_window]
        return outputs, hidden_spk, layer_fr  # n_nonzeros/n_neurons


class FFSNN_mix_v3_in1(nn.Module):
    """
    Mix skip_length range within layers
    """
    def __init__(self):
        super(FFSNN_mix_v3_in1, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.fc4 = nn.Linear(cfg_fc[2], output_size)

        self.mask1 = self.create_mix_mask(cfg_fc[0], skip_length_min[0], skip_length[0])
        self.mask2 = self.create_mix_mask(cfg_fc[1], skip_length_min[1], skip_length[1])
        self.mask3 = self.create_mix_mask(cfg_fc[2], skip_length_min[2], skip_length[2])

    def create_mix_mask(self, dim=128, min_cycle=0, max_cycle=0):
        T = 784
        mask_cyc = []
        for cycle in range(min_cycle, max_cycle+1):
            mask_ = []
            for t in range(T):
                if t % cycle == 0:
                    mask_.append(1)
                else:
                    mask_.append(0)
            mask_ = torch.tensor(mask_)
            #tmp1 = mask_.detach().numpy()
            mask_cyc.append(mask_)
        mask_cyc = torch.stack(mask_cyc)
        #tmp2 = mask_cyc.detach().numpy()

        mask = mask_cyc
        for n in range(1, dim//(max_cycle-min_cycle+1) + 1):
            mask = torch.cat((mask, torch.roll(mask_cyc, n, 1)), 0)
            #tmp3 = mask.detach().numpy()
        return mask[: dim].to(device)  # [H, T]

    def forward(self, input):
        time_window = 784//input_size
        N = input.size(0)

        h1_mem = h1_spike = h1_spike_sums = torch.zeros(N, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_spike_sums = torch.zeros(N, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_spike_sums = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, output_size, device=device)

        input = np.squeeze(input)
        input = input.view(input.size(0), -1)
        #input = input.view(input.size(0), input_size, -1)
        for step in range(time_window):
            input_x = input[:, step, None]

            h1_mem, h1_spike = mem_update_skip_woDecay(self.fc1, input_x, h1_mem, h1_spike, self.mask1[:, step])
            h2_mem, h2_spike = mem_update_skip_woDecay(self.fc2, h1_spike, h2_mem, h2_spike, self.mask2[:, step])
            h3_mem, h3_spike = mem_update_skip_woDecay(self.fc3, h2_spike, h3_mem, h3_spike, self.mask3[:, step])
            h4_mem = self.fc4(h3_spike)

            output_sum = output_sum + h4_mem # Accumulate mem of all time steps
            h1_spike_sums += h1_spike
            h2_spike_sums += h2_spike
            h3_spike_sums += h3_spike

        outputs = output_sum / time_window
        #outputs = h4_mem
        layer_fr = [h1_spike_sums.sum() / (torch.numel(h1_spike) * time_window),
                    h2_spike_sums.sum() / (torch.numel(h2_spike) * time_window),
                    h3_spike_sums.sum() / (torch.numel(h3_spike) * time_window)]
        layer_fr = torch.tensor(layer_fr)
        hidden_spk = [h1_spike_sums / time_window, h2_spike_sums / time_window, h3_spike_sums / time_window]
        return outputs, hidden_spk, layer_fr #n_nonzeros/n_neurons

