import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FFSNN.Hyperparameters import args

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
        self.spike_recorder = []  # Initialize an empty list to store spikes

    def forward(self, input, record_spikes=False):
        time_window = 784//self.stride
        N = input.size(0)

        h1_mem = h1_spike = torch.zeros(input.size(0), cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(input.size(0), cfg_fc[1], device=device)
        h3_mem = h3_spike = torch.zeros(input.size(0), cfg_fc[2], device=device)
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
            output_sum = output_sum + h4_mem # Accumulate mem of all time steps

            if record_spikes:
                self.spike_recorder.append((h1_spike[0], h2_spike[0], h3_spike[0]))

        outputs = output_sum / time_window

        return outputs, None #n_nonzeros/n_neurons
    
    def collect_spikes(self):
        return self.spike_recorder


    def gradient(self, input, criterion, target):
        time_window = 784//self.stride
        N = input.size(0)

        h1_mem = h1_spike = torch.zeros(input.size(0), cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(input.size(0), cfg_fc[1], device=device)
        h3_mem = h3_spike = torch.zeros(input.size(0), cfg_fc[2], device=device)
        h4_mem = h4_spike = output_sum = torch.zeros(input.size(0), output_size, device=device)
        grads = {}

        input = np.squeeze(input)
        input = input.view(N, -1)  # [N, 784]
        #input = input / 255.
        for step in range(time_window):   # input [N, 28, T]
            grad_t = {}
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

            loss = criterion(output_sum, target)
            loss.backward(retain_graph=True)
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_t[name] = param.grad
            l1 = grad_t['fc1.weight'].t()
            l2 = grad_t['fc2.weight'].t()
            l3 = grad_t['fc3.weight'].t()
            # l1_sum += l1
            # l2_sum += l2
            # l3_sum += l3
            l_t = torch.cat([l1.mean(dim=0), l2.mean(dim=0), l3.mean(dim=0)], dim=0).cpu()
            print("step: ", step)
            grads[step] = l_t

        outputs = output_sum / time_window
        #outputs = h4_mem
        return outputs, grads #n_nonzeros/n_neurons

    def fire_rate(self, input):
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
            h4_mem = self.fc4(h3_spike)
            output_sum = output_sum + h4_mem # Accumulate mem of all time steps
            h1_spike_sums += h1_spike
            h2_spike_sums += h2_spike
            h3_spike_sums += h3_spike

        outputs = output_sum / time_window
        layer_fr = [h1_spike_sums.sum()/(torch.numel(h1_spike)*time_window),
                    h2_spike_sums.sum()/(torch.numel(h2_spike)*time_window),
                    h3_spike_sums.sum()/(torch.numel(h3_spike)*time_window)]
        layer_fr = torch.tensor(layer_fr)
        hidden_spk = [h1_spike_sums/time_window, h2_spike_sums/time_window, h3_spike_sums/time_window]
        return outputs, hidden_spk, layer_fr

class vis_FFSNN_v2(nn.Module):
    """
    Output data for visualization
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    """
    def __init__(self, in_size=8):
        super(vis_FFSNN_v2, self).__init__()

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
        output_sum = torch.zeros(input.size(0), output_size, device=device)

        input = np.squeeze(input)
        input_org = input
        input = input.view(N, -1)  # [N, 784]
        input_rec = input.detach() # record the input for visualization
        in_rec = []
        h1_rec = []
        h2_rec = []
        h3_rec = []
        out_rec = []

        for step in range(time_window):   # input [N, 28, T]
            start_idx = step * self.stride
            if start_idx < (time_window - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)
            in_rec.append(input_x)

            h1_mem, h1_spike = mem_update(self.fc1, input_x, h1_mem, h1_spike, decay)
            h1_rec.append(h1_spike)

            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike, decay)
            h2_rec.append(h2_spike)

            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike, decay)
            h3_rec.append(h3_spike)

            h4_mem = self.fc4(h3_spike)
            out_rec.append(h4_mem)
            output_sum = output_sum + h4_mem # Accumulate mem of all time steps
        in_rec = torch.stack(in_rec).permute(1,2,0)
        h1_rec = torch.stack(h1_rec).permute(1,2,0)
        h2_rec = torch.stack(h2_rec).permute(1,2,0)
        h3_rec = torch.stack(h3_rec).permute(1,2,0)
        out_rec = torch.stack(out_rec).permute(1,2,0)

        outputs = output_sum / time_window
        record = [input_org, input_rec, in_rec, h1_rec, h2_rec, h3_rec, out_rec]
        return outputs, record
