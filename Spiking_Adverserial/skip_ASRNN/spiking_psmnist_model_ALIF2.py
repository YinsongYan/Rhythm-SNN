import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skip_ASRNN.Hyperparameters_psmnist2 import args
from tools.lib import set_seed

set_seed(1111)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algo = args.algo
thresh = args.thresh
lens = args.lens
decay = args.decay

output_size = args.out_size
input_size = args.in_size
cfg_fc = args.fc
# skip_length = args.skip_length
# skip_length_min = args.skip_length_min

phase_max = args.phase_max
cycle_min = args.cycle_min
cycle_max = args.cycle_max
duty_cycle_min = args.duty_cycle_min
duty_cycle_max = args.duty_cycle_max

b_j0 = 0.01  # neural threshold baseline
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale


# cfg_fc = [256, 512, 128]


####################################################
# For noise study
####################################################
## Add mismatch to param
def _m(d, std_p):
    # for i, v in enumerate(d):
    d_mm = np.random.normal(loc=d, scale=std_p * abs(d))
    return d_mm


## Add noise to tensor
def add_noise(tensor, std_p):
    noise = 1 + std_p * torch.randn(tensor.shape).to(device)
    tensor_ = tensor * noise
    return tensor_


## Add mask to tensor
class Silence(torch.autograd.Function):
    # Define approximate firing function
    @staticmethod
    def forward(ctx, input, p):
        mask = torch.rand(size=input.shape).gt(p).float().to(device)
        output = input * mask
        ctx.save_for_backward(mask, )
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask, None


silence = Silence.apply


def Quantize(tensor, numBits=8):
    # print(min_float, max_float)
    min_float = tensor.min().item()
    max_float = tensor.max().item()
    tensor.clamp_(min_float, max_float)
    scale = (max_float - min_float) / (2 ** numBits - 1)
    min_float_adjusted = min_float + (-min_float) % scale
    tensor = (tensor - min_float_adjusted).div(scale).round().mul(scale) + min_float_adjusted
    return tensor


def W_Quantize(model, numBits, device):
    for name, m in model._modules.items():
        if hasattr(m, "_modules"):
            model._modules[name] = W_Quantize(m, numBits, device)
        if isinstance(m, nn.Conv2d):
            m.weight.data = Quantize(m.weight.data, numBits)
        elif isinstance(m, nn.Linear):
            m.weight.data = Quantize(m.weight.data, numBits)
    return model


####################################################


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


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()


act_fun_adp = ActFun_adp.apply


# membrane potential update
def mem_update(ops, x, mem,
               spike):  # ops weight shape [32, 1, 3, 3], x [250, 1, 28, 28], mem [250, 32, 28, 28], spike [250, 32, 28, 28]
    mem = mem * decay * (1. - spike) + ops(x)  # mem: AddBackward, spike: ActFunBackward
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike


class mem_skip_update(nn.Module):
    def __init__(self):
        super(mem_skip_update, self).__init__()

    def forward(self, x, mem, spike, mask):
        mask = mask.expand(mem.size(0), -1)
        pre_mem = mem
        mem = mem * decay * (1. - spike) + x
        mem = torch.where(mask == 0, pre_mem, mem)
        spike = act_fun(mem) * mask
        return mem, spike


def mem_update_pool(opts, x, mem, spike):
    mem = mem * (1. - spike) + opts(x, 2)
    spike = act_fun(mem)
    return mem, spike


def mem_update_hidden(x, mem, spike):
    mem = mem * decay * (1. - spike) + x
    spike = act_fun(mem)
    return mem, spike


def mem_update_hidden_skip_woDecay(x, mem, spike, mask):
    mask = mask.expand(mem.size(0), -1)
    pre_mem = mem
    mem = mem * decay * (1. - spike) + x
    mem = torch.where(mask == 0, pre_mem, mem)
    spike = act_fun(mem) * mask
    return mem, spike


def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b


def mem_update_adp_skip(inputs, mem, spike, tau_adp, b, tau_m, mask, dt=1, isAdapt=1):
    mask = mask.expand(mem.size(0), -1)
    pre_mem = mem

    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.
    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b
    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt

    mem = torch.where(mask == 0, pre_mem, mem)
    inputs_ = mem - B
    spike = act_fun_adp(inputs_) * mask  # act_fun : approximation firing function
    return mem, spike, B, b


class mem_adp_skip_update(nn.Module):
    def __init__(self):
        super(mem_adp_skip_update, self).__init__()

    def forward(self, inputs, mem, spike, tau_adp, b, tau_m, mask, dt=1, isAdapt=1):
        mask = mask.expand(mem.size(0), -1)
        pre_mem = mem

        alpha = torch.exp(-1. * dt / tau_m).cuda()
        ro = torch.exp(-1. * dt / tau_adp).cuda()
        # tau_adp is tau_adaptative which is learnable # add requiregredients
        if isAdapt:
            beta = 1.8
        else:
            beta = 0.
        b = ro * b + (1 - ro) * spike
        B = b_j0 + beta * b
        mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt

        mem = torch.where(mask == 0, pre_mem, mem)
        inputs_ = mem - B
        spike = act_fun_adp(inputs_) * mask  # act_fun : approximation firing function
        return mem, spike, B, b


def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem


class ASRNN_general(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    Mix skip_length within layers. With min and max skip value.
    A general mask, incuding cycle, min_dc, max_dc.
    """

    def __init__(self, in_size=1):
        super(ASRNN_general, self).__init__()
        self.input_size = in_size
        self.stride = input_size
        self.output_size = output_size
        self.T = 784 // self.stride

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        # self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2]) ## To make it only 2 RNN layers

        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        self.tau_adp_o = nn.Parameter(torch.Tensor(output_size))

        self.tau_m_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_m_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        self._initial_parameters()
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

        self.mask1 = self.create_general_mask(cfg_fc[0], cycle_min[0], cycle_max[0], duty_cycle_min[0],
                                              duty_cycle_max[0], phase_max[0], self.T)
        self.mask2 = self.create_general_mask(cfg_fc[1], cycle_min[1], cycle_max[1], duty_cycle_min[1],
                                              duty_cycle_max[1], phase_max[1], self.T)
        self.mask3 = self.create_general_mask(cfg_fc[2], cycle_min[2], cycle_max[2], duty_cycle_min[2],
                                              duty_cycle_max[2], phase_max[2], self.T)

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        # nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        # nn.init.constant_(self.h2h_3.bias, 0)

        nn.init.normal_(self.tau_adp_h1, 700, 25)
        nn.init.normal_(self.tau_adp_h2, 700, 25)
        nn.init.normal_(self.tau_adp_h3, 700, 25)
        nn.init.normal_(self.tau_adp_o, 700, 25)

        '''
        nn.init.normal_(self.tau_m_h1, 20., 5)
        nn.init.normal_(self.tau_m_h2, 20., 5)
        nn.init.normal_(self.tau_m_h3, 20., 5)
        nn.init.normal_(self.tau_m_o, 20., 5)
        '''
        nn.init.normal_(self.tau_m_h1, 5., 1)
        nn.init.normal_(self.tau_m_h2, 5., 1)
        nn.init.normal_(self.tau_m_h3, 5., 1)
        nn.init.normal_(self.tau_m_o, 5., 1)

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
        N = input.size(0)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = h1_spike_sums = torch.zeros(N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = h2_spike_sums = torch.zeros(N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = h3_spike_sums = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, self.output_size, device=device)

        input = input.squeeze()
        input = input.view(N, -1)  # [N, 784]

        for step in range(self.T):
            start_idx = step * self.stride
            if start_idx < (self.T - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp_skip(h1_input, h2h1_mem, h2h1_spike,
                                                                            self.tau_adp_h1, self.b_h1, self.tau_m_h1,
                                                                            self.mask1[:, step])
            h1_spike_sums += h2h1_spike

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp_skip(h2_input, h2h2_mem, h2h2_spike,
                                                                            self.tau_adp_h2, self.b_h2, self.tau_m_h2,
                                                                            self.mask2[:, step])
            h2_spike_sums += h2h2_spike

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp_skip(h3_input, h2h3_mem, h2h3_spike,
                                                                            self.tau_adp_h3, self.b_h3, self.tau_m_h3,
                                                                            self.mask3[:, step])
            h3_spike_sums += h2h3_spike

            h2o3_mem = self.h2o_3(h2h3_spike)
            output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            # h1_spike_sums += h2h1_spike
            # h2_spike_sums += h2h2_spike
            # h3_spike_sums += h2h3_spike

        outputs = output_sum / self.T

        layer_fr = [h1_spike_sums.sum() / (torch.numel(h2h1_spike) * self.T),
                    h2_spike_sums.sum() / (torch.numel(h2h2_spike) * self.T),
                    h3_spike_sums.sum() / (torch.numel(h2h3_spike) * self.T)]
        layer_fr = torch.tensor(layer_fr)
        hidden_spk = [h1_spike_sums / self.T, h2_spike_sums / self.T, h3_spike_sums / self.T]
        return outputs, hidden_spk, layer_fr  # n_nonzeros/n_neurons

    def gradient(self, input, criterion, target):
        N = input.size(0)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = torch.zeros(N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, self.output_size, device=device)
        grads = {}

        input = input.squeeze()
        input = input.view(N, -1)  # [N, 784]

        for step in range(self.T):
            l1, l2, l3, l_t = 0, 0, 0, 0
            grad_t = {}
            start_idx = step * self.stride
            if start_idx < (self.T - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp_skip(h1_input, h2h1_mem, h2h1_spike,
                                                                            self.tau_adp_h1, self.b_h1, self.tau_m_h1,
                                                                            self.mask1[:, step])

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp_skip(h2_input, h2h2_mem, h2h2_spike,
                                                                            self.tau_adp_h2, self.b_h2, self.tau_m_h2,
                                                                            self.mask2[:, step])

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp_skip(h3_input, h2h3_mem, h2h3_spike,
                                                                            self.tau_adp_h3, self.b_h3, self.tau_m_h3,
                                                                            self.mask3[:, step])

            h2o3_mem = self.h2o_3(h2h3_spike)
            output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.

            loss = criterion(output_sum, target)
            loss.backward(retain_graph=True)
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_t[name] = param.grad
            l1 = grad_t['i2h_1.weight'].t()
            l2 = grad_t['i2h_2.weight'].t()
            l3 = grad_t['i2h_3.weight'].t()
            # l1_sum += l1
            # l2_sum += l2
            # l3_sum += l3
            l_t = torch.cat([l1.mean(dim=0), l2.mean(dim=0), l3.mean(dim=0)], dim=0).cpu()
            print("step: ", step)
            grads[step] = l_t
            # print(l_t.shape)

        # print("l1 sum", l1_sum.sum(1))
        # print("l2 sum", l2_sum.sum(1))
        # print("l3 sum", l3_sum.sum(1))
        outputs = output_sum / self.T

        return outputs, grads

    def fire_rate(self, input):
        N = input.size(0)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = h1_spike_sums = torch.zeros(N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = h2_spike_sums = torch.zeros(N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = h3_spike_sums = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, self.output_size, device=device)

        input = input.squeeze()
        input = input.view(N, -1)  # [N, 784]

        for step in range(self.T):
            start_idx = step * self.stride
            if start_idx < (self.T - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp_skip(h1_input, h2h1_mem, h2h1_spike,
                                                                            self.tau_adp_h1, self.b_h1, self.tau_m_h1,
                                                                            self.mask1[:, step])

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp_skip(h2_input, h2h2_mem, h2h2_spike,
                                                                            self.tau_adp_h2, self.b_h2, self.tau_m_h2,
                                                                            self.mask2[:, step])

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp_skip(h3_input, h2h3_mem, h2h3_spike,
                                                                            self.tau_adp_h3, self.b_h3, self.tau_m_h3,
                                                                            self.mask3[:, step])

            h2o3_mem = self.h2o_3(h2h3_spike)
            output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            h1_spike_sums += h2h1_spike
            h2_spike_sums += h2h2_spike
            h3_spike_sums += h2h3_spike

        outputs = output_sum / self.T

        layer_fr = [h1_spike_sums.sum() / (torch.numel(h2h1_spike) * self.T),
                    h2_spike_sums.sum() / (torch.numel(h2h2_spike) * self.T),
                    h3_spike_sums.sum() / (torch.numel(h2h3_spike) * self.T)]
        layer_fr = torch.tensor(layer_fr)
        hidden_spk = [h1_spike_sums / self.T, h2_spike_sums / self.T, h3_spike_sums / self.T]
        return outputs, hidden_spk, layer_fr  # n_nonzeros/n_neurons


class Rhy_ASRNN_thermal(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    Mix skip_length within layers. With min and max skip value.
    A general mask, incuding cycle, min_dc, max_dc.
    """

    def __init__(self, in_size=1, noise_std=0.01):
        super(Rhy_ASRNN_thermal, self).__init__()
        self.noise_std = noise_std
        self.input_size = in_size
        self.stride = input_size
        self.output_size = output_size
        self.T = 784 // self.stride

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        # self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2]) ## To make it only 2 RNN layers

        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        self.tau_adp_o = nn.Parameter(torch.Tensor(output_size))

        self.tau_m_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_m_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        self._initial_parameters()
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

        self.mask1 = self.create_general_mask(cfg_fc[0], cycle_min[0], cycle_max[0], duty_cycle_min[0],
                                              duty_cycle_max[0], phase_max[0], self.T)
        self.mask2 = self.create_general_mask(cfg_fc[1], cycle_min[1], cycle_max[1], duty_cycle_min[1],
                                              duty_cycle_max[1], phase_max[1], self.T)
        self.mask3 = self.create_general_mask(cfg_fc[2], cycle_min[2], cycle_max[2], duty_cycle_min[2],
                                              duty_cycle_max[2], phase_max[2], self.T)

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        # nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        # nn.init.constant_(self.h2h_3.bias, 0)

        nn.init.normal_(self.tau_adp_h1, 700, 25)
        nn.init.normal_(self.tau_adp_h2, 700, 25)
        nn.init.normal_(self.tau_adp_h3, 700, 25)
        nn.init.normal_(self.tau_adp_o, 700, 25)

        '''
        nn.init.normal_(self.tau_m_h1, 20., 5)
        nn.init.normal_(self.tau_m_h2, 20., 5)
        nn.init.normal_(self.tau_m_h3, 20., 5)
        nn.init.normal_(self.tau_m_o, 20., 5)
        '''
        nn.init.normal_(self.tau_m_h1, 5., 1)
        nn.init.normal_(self.tau_m_h2, 5., 1)
        nn.init.normal_(self.tau_m_h3, 5., 1)
        nn.init.normal_(self.tau_m_o, 5., 1)

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
        N = input.size(0)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = h1_spike_sums = torch.zeros(N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = h2_spike_sums = torch.zeros(N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = h3_spike_sums = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, self.output_size, device=device)

        input = input.squeeze()
        input = input.view(N, -1)  # [N, 784]

        for step in range(self.T):
            start_idx = step * self.stride
            if start_idx < (self.T - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            input_x = add_noise(input_x, self.noise_std)  # apply noise to input
            h2h1_spike = add_noise(h2h1_spike, self.noise_std)
            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp_skip(h1_input, h2h1_mem, h2h1_spike,
                                                                            self.tau_adp_h1, self.b_h1, self.tau_m_h1,
                                                                            self.mask1[:, step])
            h1_spike_sums += h2h1_spike

            h2h1_spike = add_noise(h2h1_spike, self.noise_std)
            h2h2_spike = add_noise(h2h2_spike, self.noise_std)
            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp_skip(h2_input, h2h2_mem, h2h2_spike,
                                                                            self.tau_adp_h2, self.b_h2, self.tau_m_h2,
                                                                            self.mask2[:, step])
            h2_spike_sums += h2h2_spike

            h2h2_spike = add_noise(h2h2_spike, self.noise_std)
            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp_skip(h3_input, h2h3_mem, h2h3_spike,
                                                                            self.tau_adp_h3, self.b_h3, self.tau_m_h3,
                                                                            self.mask3[:, step])
            h3_spike_sums += h2h3_spike

            h2h3_spike = add_noise(h2h3_spike, self.noise_std)
            h2o3_mem = self.h2o_3(h2h3_spike)
            output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            # h1_spike_sums += h2h1_spike
            # h2_spike_sums += h2h2_spike
            # h3_spike_sums += h2h3_spike

        outputs = output_sum / self.T

        layer_fr = [h1_spike_sums.sum() / (torch.numel(h2h1_spike) * self.T),
                    h2_spike_sums.sum() / (torch.numel(h2h2_spike) * self.T),
                    h3_spike_sums.sum() / (torch.numel(h2h3_spike) * self.T)]
        layer_fr = torch.tensor(layer_fr)
        hidden_spk = [h1_spike_sums / self.T, h2_spike_sums / self.T, h3_spike_sums / self.T]
        return outputs, hidden_spk, layer_fr  # n_nonzeros/n_neurons


class Rhy_ASRNN_silence(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    Mix skip_length within layers. With min and max skip value.
    A general mask, incuding cycle, min_dc, max_dc.
    """

    def __init__(self, in_size=1, silence_p=0.0):
        super(Rhy_ASRNN_silence, self).__init__()
        self.silence_p = silence_p
        self.input_size = in_size
        self.stride = input_size
        self.output_size = output_size
        self.T = 784 // self.stride

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        # self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2]) ## To make it only 2 RNN layers

        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        self.tau_adp_o = nn.Parameter(torch.Tensor(output_size))

        self.tau_m_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_m_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        self._initial_parameters()
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

        self.mask1 = self.create_general_mask(cfg_fc[0], cycle_min[0], cycle_max[0], duty_cycle_min[0],
                                              duty_cycle_max[0], phase_max[0], self.T)
        self.mask2 = self.create_general_mask(cfg_fc[1], cycle_min[1], cycle_max[1], duty_cycle_min[1],
                                              duty_cycle_max[1], phase_max[1], self.T)
        self.mask3 = self.create_general_mask(cfg_fc[2], cycle_min[2], cycle_max[2], duty_cycle_min[2],
                                              duty_cycle_max[2], phase_max[2], self.T)

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        # nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        # nn.init.constant_(self.h2h_3.bias, 0)

        nn.init.normal_(self.tau_adp_h1, 700, 25)
        nn.init.normal_(self.tau_adp_h2, 700, 25)
        nn.init.normal_(self.tau_adp_h3, 700, 25)
        nn.init.normal_(self.tau_adp_o, 700, 25)

        '''
        nn.init.normal_(self.tau_m_h1, 20., 5)
        nn.init.normal_(self.tau_m_h2, 20., 5)
        nn.init.normal_(self.tau_m_h3, 20., 5)
        nn.init.normal_(self.tau_m_o, 20., 5)
        '''
        nn.init.normal_(self.tau_m_h1, 5., 1)
        nn.init.normal_(self.tau_m_h2, 5., 1)
        nn.init.normal_(self.tau_m_h3, 5., 1)
        nn.init.normal_(self.tau_m_o, 5., 1)

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
        N = input.size(0)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = h1_spike_sums = torch.zeros(N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = h2_spike_sums = torch.zeros(N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = h3_spike_sums = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, self.output_size, device=device)

        input = input.squeeze()
        input = input.view(N, -1)  # [N, 784]

        for step in range(self.T):
            start_idx = step * self.stride
            if start_idx < (self.T - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp_skip(h1_input, h2h1_mem, h2h1_spike,
                                                                            self.tau_adp_h1, self.b_h1, self.tau_m_h1,
                                                                            self.mask1[:, step])
            h2h1_spike = silence(h2h1_spike, self.silence_p)  # Apply silence noise

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp_skip(h2_input, h2h2_mem, h2h2_spike,
                                                                            self.tau_adp_h2, self.b_h2, self.tau_m_h2,
                                                                            self.mask2[:, step])
            h2h2_spike = silence(h2h2_spike, self.silence_p)  # Apply silence noise

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp_skip(h3_input, h2h3_mem, h2h3_spike,
                                                                            self.tau_adp_h3, self.b_h3, self.tau_m_h3,
                                                                            self.mask3[:, step])
            h2h3_spike = silence(h2h3_spike, self.silence_p)  # Apply silence noise

            h2o3_mem = self.h2o_3(h2h3_spike)
            output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            h1_spike_sums += h2h1_spike
            h2_spike_sums += h2h2_spike
            h3_spike_sums += h2h3_spike

        outputs = output_sum / self.T

        layer_fr = [h1_spike_sums.sum() / (torch.numel(h2h1_spike) * self.T),
                    h2_spike_sums.sum() / (torch.numel(h2h2_spike) * self.T),
                    h3_spike_sums.sum() / (torch.numel(h2h3_spike) * self.T)]
        layer_fr = torch.tensor(layer_fr)
        hidden_spk = [h1_spike_sums / self.T, h2_spike_sums / self.T, h3_spike_sums / self.T]
        return outputs, hidden_spk, layer_fr  # n_nonzeros/n_neurons


