import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FFSNN.Hyperparameters_psmnist import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algo = args.algo
thresh = args.thresh
lens = args.lens
decay = args.decay

output_size = args.out_size
input_size = args.in_size
cfg_fc = args.fc


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
# todo: remove decay if fix decay
def mem_update(ops, x, mem, spike, decay): # ops weight shape [32, 1, 3, 3], x [250, 1, 28, 28], mem [250, 32, 28, 28], spike [250, 32, 28, 28]
    if algo == 'STBP':
        #decay = torch.rand(mem.size(-1), device=device)
        #decay = 0.5*torch.ones_like(mem[-1], device=device)
        mem = mem * decay * (1. - spike) + ops(x)   # mem: AddBackward, spike: ActFunBackward
    else:
        #decay = torch.rand(mem.size(-1), device=device)
        #decay = 0.5*torch.ones_like(mem[-1], device=device)
        mem = mem.detach() * decay * (1. - spike.detach()) + ops(x)  # STOP the gradient for TD
    spike = act_fun(mem) # act_fun : approximation firing function
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

    def forward(self, input):
        time_window = 784//input_size

        h1_mem = h1_spike = torch.zeros(input.size(0), cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(input.size(0), cfg_fc[1], device=device)
        h3_mem = h3_spike = torch.zeros(input.size(0), cfg_fc[2], device=device)
        h4_mem = h4_spike = output_sum = torch.zeros(input.size(0), output_size, device=device)

        n_neurons = 0
        n_nonzeros= 0

        for step in range(time_window):   # input [N, 28, T]

            if algo == 'noTD':  # if noTD, reset mem potential every time step
                h1_mem = h1_spike = torch.zeros(input.size(0), cfg_fc[0], device=device)
                h2_mem = h2_spike = torch.zeros(input.size(0), cfg_fc[1], device=device)
                h3_mem = h3_spike = torch.zeros(input.size(0), cfg_fc[2], device=device)

            input = np.squeeze(input)
            input = input.view(input.size(0), input_size, -1)
            input_x = input[:, :, step]

            h1_mem, h1_spike = mem_update(self.fc1, input_x, h1_mem, h1_spike, decay)#_cfg[0])
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike, decay)#_cfg[1])
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike, decay)#_cfg[2])
            #h4_mem, h4_spike = mem_update(self.fc4, h3_spike, h4_mem, h4_spike)
            h4_mem = self.fc4(h3_spike)
            output_sum = output_sum + h4_mem # Accumulate mem of all time steps

            #### Calculate spike count ####
            '''
            n_nonzeros += torch.count_nonzero(h1_spike) \
                          + torch.count_nonzero(h2_spike) \
                          + torch.count_nonzero(h3_spike)

            n_neurons += torch.numel(h1_spike) \
                          + torch.numel(h2_spike) \
                          + torch.numel(h3_spike)
            '''

        outputs = output_sum / time_window
        #outputs = h4_mem

        return outputs, None #n_nonzeros/n_neurons

class FFSNN_v2(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    """
    def __init__(self, in_size=1):
        super(FFSNN_v2, self).__init__()

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

        h1_mem = h1_spike = h1_spike_sums = torch.zeros(input.size(0), cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_spike_sums = torch.zeros(input.size(0), cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_spike_sums = torch.zeros(input.size(0), cfg_fc[2], device=device)
        h4_mem = h4_spike = output_sum = torch.zeros(input.size(0), output_size, device=device)

        input = np.squeeze(input)
        input = input.view(N, -1)  # [N, 784]
        #input = input / 255.
        for step in range(time_window):   # input [N, 28, T]
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
            output_sum = output_sum + h4_mem
            h1_spike_sums += h1_spike
            h2_spike_sums += h2_spike
            h3_spike_sums += h3_spike

        outputs = output_sum / time_window
        #outputs = h4_mem
        layer_fr = [h1_spike_sums.sum()/(torch.numel(h1_spike)*time_window),
                    h2_spike_sums.sum()/(torch.numel(h2_spike)*time_window),
                    h3_spike_sums.sum()/(torch.numel(h3_spike)*time_window)]
        layer_fr = torch.tensor(layer_fr)
        hidden_spk = [h1_spike_sums/time_window, h2_spike_sums/time_window, h3_spike_sums/time_window]
        return outputs, hidden_spk, layer_fr #n_nonzeros/n_neurons
