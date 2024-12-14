import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from skip_FFSNN.Hyperparameters_psmnist_cpy1 import args

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

phase_max = args.phase_max
cycle_min = args.cycle_min
cycle_max = args.cycle_max
duty_cycle_min = args.duty_cycle_min
duty_cycle_max = args.duty_cycle_max
trainable_ratio = args.trainable_ratio

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

        self.mask1 = self.create_mask(cfg_fc[0], skip_length[0])
        self.mask2 = self.create_mask(cfg_fc[1], skip_length[1])
        self.mask3 = self.create_mask(cfg_fc[2], skip_length[2])


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

        h1_mem = h1_spike = torch.zeros(N, cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(N, cfg_fc[1], device=device)
        h3_mem = h3_spike = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, output_size, device=device)

        input = np.squeeze(input)
        input = input.view(N, -1) #[N, 784]
        #input = input / 255.
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
    Mix skip_length for each neuron within the layers
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

class FFSNN_mix_v3(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    Mix skip_length range within layers. With Min and Max skip value.
    """
    def __init__(self, in_size=1, bias=True):
        super(FFSNN_mix_v3, self).__init__()
        self.input_size = in_size
        self.stride = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, cfg_fc[0], bias=bias)
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], bias=bias)
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], bias=bias)
        self.fc4 = nn.Linear(cfg_fc[2], output_size, bias=bias)

        self.mask1 = self.create_mix_mask(cfg_fc[0], skip_length_min[0], skip_length[0])
        self.mask2 = self.create_mix_mask(cfg_fc[1], skip_length_min[1], skip_length[1])
        self.mask3 = self.create_mix_mask(cfg_fc[2], skip_length_min[2], skip_length[2])

    def create_mix_mask(self, dim=128, min_cycle=0, max_cycle=0):
        T = 784 // self.stride
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

class FFSNN_general(nn.Module):
    """
    A general mask, incuding cycle, min_dc, max_dc.
    """
    def __init__(self, T = 784):
        super(FFSNN_general, self).__init__()
        self.T = T
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.fc4 = nn.Linear(cfg_fc[2], output_size)

        self.mask1 = self.create_general_mask(cfg_fc[0], cycle_min[0], cycle_max[0], duty_cycle_min[0], duty_cycle_max[0], phase_max[0], self.T)
        self.mask2 = self.create_general_mask(cfg_fc[1], cycle_min[1], cycle_max[1], duty_cycle_min[1], duty_cycle_max[1], phase_max[1], self.T)
        self.mask3 = self.create_general_mask(cfg_fc[2], cycle_min[2], cycle_max[2], duty_cycle_min[2], duty_cycle_max[2], phase_max[2], self.T)

    def create_general_mask(self, dim=128, c_min=4, c_max=8, min_dc=0.1, max_dc=0.9, phase_shift_max=0.5, T=784):
        mask = []
        # Create ranges for cycles and duty cycles
        dc_steps = torch.linspace(min_dc, max_dc, steps=dim) 
        cycles = torch.linspace(c_min, c_max, steps=dim)
        # Generate phase shifts within the specified maximum
        phase_shifts = torch.linspace(0, int(phase_shift_max * c_max), steps=dim)
        
        for cycle, dc, phase_shift in zip(cycles, dc_steps, phase_shifts):
            cycle = int(torch.round(cycle))
            on_length = int(torch.round(dc * cycle))
            off_length = cycle - on_length
            pattern = [1] * on_length + [0] * off_length

            phase_shift = int(torch.round(phase_shift))
            pattern = pattern[-phase_shift:] + pattern[:-phase_shift]  # Negative slicing for shifting right
            
            full_pattern = pattern * (T // cycle) + pattern[:T % cycle]  # Ensure the pattern fits exactly into T
            mask.append(full_pattern)
        
        mask = torch.tensor(mask, dtype=torch.float32)
        return mask.to(device)

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

    def gradient(self, input, criterion, target):
        time_window = 784//input_size
        N = input.size(0)

        h1_mem = h1_spike = torch.zeros(N, cfg_fc[0], device=device)
        h2_mem = h2_spike = torch.zeros(N, cfg_fc[1], device=device)
        h3_mem = h3_spike = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, output_size, device=device)
        grads = {}

        input = np.squeeze(input)
        input = input.view(input.size(0), -1)
        #input = input.view(input.size(0), input_size, -1)
        for step in range(time_window):
            # if step > 100:
            #     break
            grad_t = {}
            # l1_sum = 0
            # l2_sum = 0
            # l3_sum = 0
            input_x = input[:, step, None]

            h1_mem, h1_spike = mem_update_skip_woDecay(self.fc1, input_x, h1_mem, h1_spike, self.mask1[:, step])
            h2_mem, h2_spike = mem_update_skip_woDecay(self.fc2, h1_spike, h2_mem, h2_spike, self.mask2[:, step])
            h3_mem, h3_spike = mem_update_skip_woDecay(self.fc3, h2_spike, h3_mem, h3_spike, self.mask3[:, step])
            h4_mem = self.fc4(h3_spike)
            output_sum = output_sum + h4_mem 

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
            #print(l_t.shape)
        
        # print("l1 sum", l1_sum.sum(1))
        # print("l2 sum", l2_sum.sum(1))
        # print("l3 sum", l3_sum.sum(1))

        outputs = output_sum / time_window
        return outputs, grads

    def fire_rate(self, input):
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
