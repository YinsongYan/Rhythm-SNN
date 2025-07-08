import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import sys

sys.path.append("..")
from lsnn.Hyperparameters import args


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# Hyperparameters
algo = args.algo
thresh = args.thresh
lens = args.lens
decay = args.decay

# Network architecture
output_size = args.out_size
input_size = args.in_size
cfg_fc = args.fc

# RhythmSNN initialized parameters
phase_max = args.phase_max
cycle_min = args.cycle_min
cycle_max = args.cycle_max
duty_cycle_min = args.duty_cycle_min
duty_cycle_max = args.duty_cycle_max


b_j0 = 0.01  # neural threshold baseline
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale


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
def mem_update_skip(ops, x, mem, spike, mask):
    mem = mem * decay * (1. - spike) + ops(x) * mask
    spike = act_fun(mem) * mask  # act_fun : approximation firing function
    return mem, spike


def mem_update_skip_woDecay(ops, x, mem, spike, mask):
    mask = mask.expand(mem.size(0), -1)
    pre_mem = mem
    mem = mem * decay * (1. - spike) + ops(x)
    mem = torch.where(mask == 0, pre_mem, mem)
    # tmp2 = (mem).detach().cpu().numpy()
    spike = act_fun(mem) * mask
    return mem, spike


def mem_update_NU_adp_skip(ops, input, mem, spike, tau_adp, b, mask, dt=1, isAdapt=1):
    inputs = ops(input)
    mask = mask.expand(mem.size(0), -1)
    pre_mem = mem
    tau_m_ = torch.tensor(tau_m, device=inputs.device) if not isinstance(tau_m, torch.Tensor) else tau_m
    tau_adp_ = torch.tensor(tau_adp, device=inputs.device) if not isinstance(tau_adp, torch.Tensor) else tau_adp
    alpha = torch.exp(-1. * dt / tau_m_).to(inputs.device)
    ro = torch.exp(-1. * dt / tau_adp_).to(inputs.device)
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


def mem_update_NU_adp(ops, input, mem, spike, tau_adp, b, dt=1, isAdapt=1):
    inputs = ops(input)
    tau_m_ = torch.tensor(tau_m, device=inputs.device) if not isinstance(tau_m, torch.Tensor) else tau_m
    tau_adp_ = torch.tensor(tau_adp, device=inputs.device) if not isinstance(tau_adp, torch.Tensor) else tau_adp
    alpha = torch.exp(-1. * dt / tau_m_).to(inputs.device)
    ro = torch.exp(-1. * dt / tau_adp_).to(inputs.device)
    # tau_adp is tau_adaptative which is learnable # add requiregredients
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


def mem_update_adp_DEXAT_skip(inputs, mem, spike, tau_a1, tau_a2, b1, b2, mask, dt=1, isAdapt=1):
    mask = mask.expand(mem.size(0), -1)
    pre_mem = mem
    tau_m_ = torch.tensor(tau_m, dtype=torch.float32) if not isinstance(tau_m, torch.Tensor) else tau_m
    tau_a1_ = torch.tensor(tau_a1, dtype=torch.float32) if not isinstance(tau_a1, torch.Tensor) else tau_a1
    tau_a2_ = torch.tensor(tau_a2, dtype=torch.float32) if not isinstance(tau_a2, torch.Tensor) else tau_a2

    alpha = torch.exp(-1. * dt / tau_m_).to(inputs.device)
    ro1 = torch.exp(-1. * dt / tau_a1_).to(inputs.device)
    ro2 = torch.exp(-1. * dt / tau_a2_).to(inputs.device)
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta1 = 1.8
        beta2 = 5 * beta1
    else:
        beta1 = 0.
        beta2 = 0.
    b1 = ro1 * b1 + (1 - ro1) * spike
    b2 = ro2 * b2 + (1 - ro2) * spike
    B = b_j0 + beta1 * b1 + beta2 * b2
    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt

    mem = torch.where(mask == 0, pre_mem, mem)
    inputs_ = mem - B
    spike = act_fun_adp(inputs_) * mask  # act_fun : approximation firing function
    return mem, spike, B, b1, b2


def mem_update_adp_DEXAT(inputs, mem, spike, tau_a1, tau_a2, b1, b2, dt=1, isAdapt=1):
    tau_m_ = torch.tensor(tau_m, dtype=torch.float32) if not isinstance(tau_m, torch.Tensor) else tau_m
    tau_a1_ = torch.tensor(tau_a1, dtype=torch.float32) if not isinstance(tau_a1, torch.Tensor) else tau_a1
    tau_a2_ = torch.tensor(tau_a2, dtype=torch.float32) if not isinstance(tau_a2, torch.Tensor) else tau_a2

    alpha = torch.exp(-1. * dt / tau_m_).to(inputs.device)
    ro1 = torch.exp(-1. * dt / tau_a1_).to(inputs.device)
    ro2 = torch.exp(-1. * dt / tau_a2_).to(inputs.device)
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta1 = 1.8
        beta2 = 5 * beta1
    else:
        beta1 = 0.
        beta2 = 0.
    b1 = ro1 * b1 + (1 - ro1) * spike
    b2 = ro2 * b2 + (1 - ro2) * spike
    B = b_j0 + beta1 * b1 + beta2 * b2
    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt

    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b1, b2


def mem_update_adp_skip(inputs, mem, spike, tau_adp, b, tau_m, mask, dt=1, isAdapt=1):
    mask = mask.expand(mem.size(0), -1)
    pre_mem = mem
    tau_m_ = torch.tensor(tau_m, dtype=torch.float32) if not isinstance(tau_m, torch.Tensor) else tau_m
    tau_adp_ = torch.tensor(tau_adp, dtype=torch.float32) if not isinstance(tau_adp, torch.Tensor) else tau_adp
    alpha = torch.exp(-1. * dt / tau_m_).to(inputs.device)
    ro = torch.exp(-1. * dt / tau_adp_).to(inputs.device)
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


def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
    if not isinstance(tau_m, torch.Tensor):
        tau_m_ = torch.tensor(tau_m, dtype=torch.float32)
    else:
        tau_m_ = tau_m
    if not isinstance(tau_adp, torch.Tensor):
        tau_adp_ = torch.tensor(tau_adp, dtype=torch.float32)
    else:
        tau_adp_ = tau_adp
    # tau_adp_ = torch.tensor(tau_adp, device=inputs.device) if not isinstance(tau_adp, torch.Tensor) else tau_adp
    alpha = torch.exp(-1. * dt / tau_m_).to(inputs.device)
    ro = torch.exp(-1. * dt / tau_adp_).to(inputs.device)
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




def mem_update_pool(opts, x, mem, spike):
    mem = mem * (1. - spike) + opts(x, 2)
    spike = act_fun(mem)
    return mem, spike





class SRNN_ALIF(nn.Module):
    def __init__(self, in_size=100, bias=True, n_out=2):
        super().__init__()
        self.input_size = in_size
        self.stride = in_size  # input_size
        self.output_size = n_out  # 1960  # output_siz

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])

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
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)

        nn.init.normal_(self.tau_adp_h1, 700, 25)
        nn.init.normal_(self.tau_adp_h2, 700, 25)
        nn.init.normal_(self.tau_adp_h3, 700, 25)
        nn.init.normal_(self.tau_adp_o, 700, 25)

        nn.init.normal_(self.tau_m_h1, 5., 1)
        nn.init.normal_(self.tau_m_h2, 5., 1)
        nn.init.normal_(self.tau_m_h3, 5., 1)
        nn.init.normal_(self.tau_m_o, 5., 1)

    def init_state(self, batch_size):
        h2h1_mem = h2h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(batch_size, cfg_fc[2], device=device)
        b_h1 = b_h2 = b_h3 = b_j0
        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        b_list = [b_h1, b_h2, b_h3]
        init_state_ = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       'b_list': b_list}
        return init_state_

    def forward(self, input, init_state):
        N = input.size(0)  # batch_size
        seq_len = input.size(1)  # seq_len = seq length * char_duration

        mem_list = init_state['mem_list']
        spike_list = init_state['spike_list']
        b_list = init_state['b_list']
        h2h1_mem = mem_list[0]
        h2h2_mem = mem_list[1]
        h2h3_mem = mem_list[2]
        h2h1_spike = spike_list[0]
        h2h2_spike = spike_list[1]
        h2h3_spike = spike_list[2]
        b_h1 = b_list[0]
        b_h2 = b_list[1]
        b_h3 = b_list[2]

        # output_sum = torch.zeros(11, N, self.output_size, device=device)
        outputs = torch.zeros(seq_len, N, self.output_size, device=device)

        hidden_spikes = torch.zeros(seq_len, N, cfg_fc[2], device=device)

        n_neurons = 0.
        n_nonzeros = 0.

        # input = input.unsqueeze(1)  # [bs] -> [bs, 1]
        # input = self.Embedding(input)  # [bs, 1, 128]
        #
        # # input = np.squeeze(input)
        # input = input.view(N, -1)  # [N, 128]

        # T = 128 // self.input_size  # 128 // 8 = 16
        T = seq_len
        for step in range(T):
            input_x = np.squeeze(input[:, step, :]).view(N, -1)  # .reshape(-1, self.input_size)  # [bs, 100]

            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_h1 = mem_update_adp(h1_input, h2h1_mem, h2h1_spike,
                                                                  self.tau_adp_h1, b_h1, self.tau_m_h1,
                                                                  )

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_h2 = mem_update_adp(h2_input, h2h2_mem, h2h2_spike,
                                                                  self.tau_adp_h2, b_h2, self.tau_m_h2,
                                                                  )

            h3_input = self.i2h_3(h2h2_spike) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_h3 = mem_update_adp(h3_input, h2h3_mem, h2h3_spike,
                                                                  self.tau_adp_h3, b_h3, self.tau_m_h3,
                                                                  )
            hidden_spikes[step] = h2h3_spike
            h2o3_mem = self.h2o_3(h2h3_mem)

            # output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            outputs[step] = h2o3_mem

        # output = output_sum / T  # [bs, 1979]
        output = outputs.permute(1, 0, 2)  # [seq_len, bs, 2] --> [bs, seq_len, 2]
        hidden_spike = hidden_spikes.permute(1, 0, 2)
        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        b_list = [b_h1, b_h2, b_h3]
        final_state = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       'b_list': b_list}

        #### Calculate spike count ####
        # n_nonzeros = torch.tensor(torch.nonzero(h2h1_spike).shape[0]) \
        #              + torch.tensor(torch.nonzero(h2h2_spike).shape[0]) \
        #              + torch.tensor(torch.nonzero(h2h3_spike).shape[0])
        n_nonzeros = torch.count_nonzero(h2h1_spike) \
                     + torch.count_nonzero(h2h2_spike) \
                     + torch.count_nonzero(h2h3_spike)
        n_neurons = torch.numel(h2h1_spike) \
                    + torch.numel(h2h2_spike) \
                    + torch.numel(h2h3_spike)
        fire_rate = n_nonzeros / n_neurons

        return output, final_state, fire_rate, hidden_spike  # n_nonzeros/n_neurons


class Gen_skip_SRNN_DEXAT_mix(nn.Module):
    def __init__(self, in_size=100, bias=True, n_out=2, tau_a1=30, tau_a2=300):
        super().__init__()
        # self.infeature_size = 256
        self.input_size = in_size
        self.stride = in_size  # input_size
        self.output_size = n_out  # 1960  # output_siz
        self.tau_a1 = tau_a1
        self.tau_a2 = tau_a2
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])

        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        # self.tau_adp_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        # self.tau_adp_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        # self.tau_adp_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        # self.tau_adp_o = nn.Parameter(torch.Tensor(output_size))
        #
        # self.tau_m_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        # self.tau_m_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        # self.tau_m_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        # self.tau_m_o = nn.Parameter(torch.Tensor(output_size))

        self._initial_parameters()
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

        self.mask1 = self.create_general_mask(cfg_fc[0], cycle_min[0], cycle_max[0], duty_cycle_min[0],
                                              duty_cycle_max[0], phase_max[0])
        self.mask2 = self.create_general_mask(cfg_fc[1], cycle_min[1], cycle_max[1], duty_cycle_min[1],
                                              duty_cycle_max[1], phase_max[1])
        self.mask3 = self.create_general_mask(cfg_fc[2], cycle_min[2], cycle_max[2], duty_cycle_min[2],
                                              duty_cycle_max[2], phase_max[2])

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)

        # nn.init.normal_(self.tau_adp_h1, 700, 25)
        # nn.init.normal_(self.tau_adp_h2, 700, 25)
        # nn.init.normal_(self.tau_adp_h3, 700, 25)
        # nn.init.normal_(self.tau_adp_o, 700, 25)
        #
        # nn.init.normal_(self.tau_m_h1, 20., 5)
        # nn.init.normal_(self.tau_m_h2, 20., 5)
        # nn.init.normal_(self.tau_m_h3, 20., 5)
        # nn.init.normal_(self.tau_m_o, 20., 5)

    def init_state(self, batch_size):
        h2h1_mem = h2h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(batch_size, cfg_fc[2], device=device)

        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]

        init_state_ = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       }
        return init_state_

    def create_general_mask(self, dim=128, c_min=4, c_max=8, min_dc=0.1, max_dc=0.9, phase_shift_max=0.5, T=8 * 784):
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

    def forward(self, input, init_state):
        N = input.size(0)  # batch_size
        seq_len = input.size(1)  # seq_len = seq length * char_duration

        mem_list = init_state['mem_list']
        spike_list = init_state['spike_list']

        h2h1_mem = mem_list[0]
        h2h2_mem = mem_list[1]
        h2h3_mem = mem_list[2]
        h2h1_spike = spike_list[0]
        h2h2_spike = spike_list[1]
        h2h3_spike = spike_list[2]
        b_h11 = b_h21 = b_h31 = b_j0
        b_h12 = b_h22 = b_h32 = b_j0

        # output_sum = torch.zeros(11, N, self.output_size, device=device)
        outputs = torch.zeros(seq_len, N, self.output_size, device=device)

        hidden_spikes = torch.zeros(seq_len, N, cfg_fc[2], device=device)
        hidden_mems = torch.zeros(seq_len, N, cfg_fc[2], device=device)
        hidden_thetas = torch.zeros(seq_len, N, cfg_fc[2], device=device)

        n_neurons = 0.
        n_nonzeros = 0.

        # input = input.unsqueeze(1)  # [bs] -> [bs, 1]
        # input = self.Embedding(input)  # [bs, 1, 128]
        #
        # # input = np.squeeze(input)
        # input = input.view(N, -1)  # [N, 128]

        # T = 128 // self.input_size  # 128 // 8 = 16
        T = seq_len
        for step in range(T):
            input_x = np.squeeze(input[:, step, :]).view(N, -1)  # .reshape(-1, self.input_size)  # [bs, 100]

            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_h11, b_h12 = mem_update_adp_DEXAT_skip(h1_input, h2h1_mem, h2h1_spike,
                                                                                     self.tau_a1, self.tau_a2, b_h11,
                                                                                     b_h12,
                                                                                     self.mask1[:, step])

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_h21, b_h22 = mem_update_adp_DEXAT_skip(h2_input, h2h2_mem, h2h2_spike,
                                                                                     self.tau_a1, self.tau_a2, b_h21,
                                                                                     b_h22,
                                                                                     self.mask2[:, step])

            h3_input = self.i2h_3(h2h2_spike) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_h31, b_h32 = mem_update_adp_DEXAT_skip(h3_input, h2h3_mem, h2h3_spike,
                                                                                     self.tau_a1, self.tau_a2, b_h31,
                                                                                     b_h32,
                                                                                     self.mask3[:, step])
            hidden_spikes[step] = h2h3_spike
            hidden_mems[step] = h2h3_mem
            hidden_thetas[step] = theta_h3

            h2o3_mem = self.h2o_3(h2h3_mem)

            # output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            outputs[step] = h2o3_mem

        # output = output_sum / T  # [bs, 1979]
        output = outputs.permute(1, 0, 2)  # [seq_len, bs, 2] --> [bs, seq_len, 2]
        hidden_spike = hidden_spikes.permute(1, 0, 2)
        hidden_mem = hidden_mems.permute(1, 0, 2)
        hidden_theta = hidden_thetas.permute(1, 0, 2)

        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        final_state = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       }

        #### Calculate spike count ####
        n_nonzeros = torch.tensor(torch.nonzero(h2h1_spike).shape[0]) \
                     + torch.tensor(torch.nonzero(h2h2_spike).shape[0]) \
                     + torch.tensor(torch.nonzero(h2h3_spike).shape[0])
        # n_nonzeros = torch.count_nonzero(h1_spike) \
        #               + torch.count_nonzero(h2_spike) \
        #               + torch.count_nonzero(h3_spike)
        n_neurons = torch.numel(h2h1_spike) \
                    + torch.numel(h2h2_spike) \
                    + torch.numel(h2h3_spike)
        # fire_rate = n_nonzeros / n_neurons
        fire_rate = torch.true_divide(n_nonzeros, n_neurons).float()

        return output, final_state, fire_rate, hidden_spike, hidden_mem, hidden_theta  # n_nonzeros/n_neurons


class SRNN_DEXAT(nn.Module):
    def __init__(self, in_size=100, bias=True, n_out=2, tau_a1=30, tau_a2=300):
        super().__init__()
        # self.infeature_size = 256
        self.input_size = in_size
        self.stride = in_size  # input_size
        self.output_size = n_out  # 1960  # output_siz
        self.tau_a1 = tau_a1
        self.tau_a2 = tau_a2
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])

        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        self._initial_parameters()
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)

        # nn.init.normal_(self.tau_adp_h1, 700, 25)
        # nn.init.normal_(self.tau_adp_h2, 700, 25)
        # nn.init.normal_(self.tau_adp_h3, 700, 25)
        # nn.init.normal_(self.tau_adp_o, 700, 25)
        #
        # nn.init.normal_(self.tau_m_h1, 20., 5)
        # nn.init.normal_(self.tau_m_h2, 20., 5)
        # nn.init.normal_(self.tau_m_h3, 20., 5)
        # nn.init.normal_(self.tau_m_o, 20., 5)

    def init_state(self, batch_size):
        h2h1_mem = h2h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(batch_size, cfg_fc[2], device=device)

        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]

        init_state_ = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       }
        return init_state_

    def forward(self, input, init_state):
        N = input.size(0)  # batch_size
        seq_len = input.size(1)  # seq_len = seq length * char_duration

        mem_list = init_state['mem_list']
        spike_list = init_state['spike_list']

        h2h1_mem = mem_list[0]
        h2h2_mem = mem_list[1]
        h2h3_mem = mem_list[2]
        h2h1_spike = spike_list[0]
        h2h2_spike = spike_list[1]
        h2h3_spike = spike_list[2]
        b_h11 = b_h21 = b_h31 = b_j0
        b_h12 = b_h22 = b_h32 = b_j0

        # output_sum = torch.zeros(11, N, self.output_size, device=device)
        outputs = torch.zeros(seq_len, N, self.output_size, device=device)

        hidden_spikes = torch.zeros(seq_len, N, cfg_fc[2], device=device)
        hidden_mems = torch.zeros(seq_len, N, cfg_fc[2], device=device)
        hidden_thetas = torch.zeros(seq_len, N, cfg_fc[2], device=device)

        n_neurons = 0.
        n_nonzeros = 0.

        # input = input.unsqueeze(1)  # [bs] -> [bs, 1]
        # input = self.Embedding(input)  # [bs, 1, 128]
        #
        # # input = np.squeeze(input)
        # input = input.view(N, -1)  # [N, 128]

        # T = 128 // self.input_size  # 128 // 8 = 16
        T = seq_len
        for step in range(T):
            input_x = np.squeeze(input[:, step, :]).view(N, -1)  # .reshape(-1, self.input_size)  # [bs, 100]

            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_h11, b_h12 = mem_update_adp_DEXAT(h1_input, h2h1_mem, h2h1_spike,
                                                                                self.tau_a1, self.tau_a2, b_h11, b_h12,
                                                                                )

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_h21, b_h22 = mem_update_adp_DEXAT(h2_input, h2h2_mem, h2h2_spike,
                                                                                self.tau_a1, self.tau_a2, b_h21, b_h22,
                                                                                )

            h3_input = self.i2h_3(h2h2_spike) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_h31, b_h32 = mem_update_adp_DEXAT(h3_input, h2h3_mem, h2h3_spike,
                                                                                self.tau_a1, self.tau_a2, b_h31, b_h32,
                                                                                )
            hidden_spikes[step] = h2h3_spike
            hidden_mems[step] = h2h3_mem
            hidden_thetas[step] = theta_h3
            h2o3_mem = self.h2o_3(h2h3_mem)

            # output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            outputs[step] = h2o3_mem

        # output = output_sum / T  # [bs, 1979]
        output = outputs.permute(1, 0, 2)  # [seq_len, bs, 2] --> [bs, seq_len, 2]
        hidden_spike = hidden_spikes.permute(1, 0, 2)
        hidden_mem = hidden_mems.permute(1, 0, 2)
        hidden_theta = hidden_thetas.permute(1, 0, 2)

        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        final_state = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       }

        #### Calculate spike count ####
        n_nonzeros = torch.tensor(torch.nonzero(h2h1_spike).shape[0]) \
                     + torch.tensor(torch.nonzero(h2h2_spike).shape[0]) \
                     + torch.tensor(torch.nonzero(h2h3_spike).shape[0])
        # n_nonzeros = torch.count_nonzero(h1_spike) \
        #               + torch.count_nonzero(h2_spike) \
        #               + torch.count_nonzero(h3_spike)
        n_neurons = torch.numel(h2h1_spike) \
                    + torch.numel(h2h2_spike) \
                    + torch.numel(h2h3_spike)
        # fire_rate = n_nonzeros / n_neurons
        fire_rate = torch.true_divide(n_nonzeros, n_neurons).float()

        return output, final_state, fire_rate, hidden_spike, hidden_mem, hidden_theta  # n_nonzeros/n_neurons


class Gen_skip_SRNN_ALIF_fix(nn.Module):
    def __init__(self, in_size=100, bias=True, n_out=2, tau_a=600):
        super().__init__()
        # self.infeature_size = 256
        self.input_size = in_size
        self.stride = in_size  # input_size
        self.output_size = n_out  # 1960  # output_siz
        self.tau_a = tau_a
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        self.tau_adp_h1 = self.tau_adp_h2 = self.tau_adp_h3 = self.tau_a
        self.tau_m_h1 = self.tau_m_h2 = self.tau_m_h3 = tau_m

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])

        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        self._initial_parameters()
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

        self.mask1 = self.create_general_mask(cfg_fc[0], cycle_min[0], cycle_max[0], duty_cycle_min[0],
                                              duty_cycle_max[0], phase_max[0])
        self.mask2 = self.create_general_mask(cfg_fc[1], cycle_min[1], cycle_max[1], duty_cycle_min[1],
                                              duty_cycle_max[1], phase_max[1])
        self.mask3 = self.create_general_mask(cfg_fc[2], cycle_min[2], cycle_max[2], duty_cycle_min[2],
                                              duty_cycle_max[2], phase_max[2])

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)

    def init_state(self, batch_size):
        h2h1_mem = h2h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(batch_size, cfg_fc[2], device=device)
        b_h1 = b_h2 = b_h3 = b_j0
        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        b_list = [b_h1, b_h2, b_h3]
        init_state_ = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       'b_list': b_list}
        return init_state_

    def create_general_mask(self, dim=128, c_min=4, c_max=8, min_dc=0.1, max_dc=0.9, phase_shift_max=0.5, T=8*784):
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

    def forward(self, input, init_state):
        N = input.size(0)  # batch_size
        seq_len = input.size(1)  # seq_len = seq length * char_duration

        mem_list = init_state['mem_list']
        spike_list = init_state['spike_list']
        b_list = init_state['b_list']
        h2h1_mem = mem_list[0]
        h2h2_mem = mem_list[1]
        h2h3_mem = mem_list[2]
        h2h1_spike = spike_list[0]
        h2h2_spike = spike_list[1]
        h2h3_spike = spike_list[2]
        b_h1 = b_list[0]
        b_h2 = b_list[1]
        b_h3 = b_list[2]

        # output_sum = torch.zeros(11, N, self.output_size, device=device)
        outputs = torch.zeros(seq_len, N, self.output_size, device=device)

        hidden_spikes = torch.zeros(seq_len, N, cfg_fc[2], device=device)
        hidden_mems = torch.zeros(seq_len, N, cfg_fc[2], device=device)
        hidden_thetas = torch.zeros(seq_len, N, cfg_fc[2], device=device)

        n_neurons = 0.
        n_nonzeros = 0.

        # input = input.unsqueeze(1)  # [bs] -> [bs, 1]
        # input = self.Embedding(input)  # [bs, 1, 128]
        #
        # # input = np.squeeze(input)
        # input = input.view(N, -1)  # [N, 128]

        # T = 128 // self.input_size  # 128 // 8 = 16
        T = seq_len
        for step in range(T):
            input_x = np.squeeze(input[:, step, :]).view(N, -1)  # .reshape(-1, self.input_size)  # [bs, 100]

            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_h1 = mem_update_adp_skip(h1_input, h2h1_mem, h2h1_spike,
                                                                       self.tau_adp_h1, b_h1, self.tau_m_h1,
                                                                       self.mask1[:, step])

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_h2 = mem_update_adp_skip(h2_input, h2h2_mem, h2h2_spike,
                                                                       self.tau_adp_h2, b_h2, self.tau_m_h2,
                                                                       self.mask2[:, step])

            h3_input = self.i2h_3(h2h2_spike) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_h3 = mem_update_adp_skip(h3_input, h2h3_mem, h2h3_spike,
                                                                       self.tau_adp_h3, b_h3, self.tau_m_h3,
                                                                       self.mask3[:, step])
            hidden_spikes[step] = h2h3_spike
            hidden_mems[step] = h2h3_mem
            hidden_thetas[step] = theta_h3
            h2o3_mem = self.h2o_3(h2h3_mem)

            # output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            outputs[step] = h2o3_mem

        # output = output_sum / T  # [bs, 1979]
        output = outputs.permute(1, 0, 2)  # [seq_len, bs, 2] --> [bs, seq_len, 2]
        hidden_spike = hidden_spikes.permute(1, 0, 2)
        hidden_mem = hidden_mems.permute(1, 0, 2)
        hidden_theta = hidden_thetas.permute(1, 0, 2)

        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        b_list = [b_h1, b_h2, b_h3]
        final_state = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       'b_list': b_list}

        #### Calculate spike count ####
        n_nonzeros = torch.tensor(torch.nonzero(h2h1_spike).shape[0]) \
                     + torch.tensor(torch.nonzero(h2h2_spike).shape[0]) \
                     + torch.tensor(torch.nonzero(h2h3_spike).shape[0])
        # n_nonzeros = torch.count_nonzero(h1_spike) \
        #               + torch.count_nonzero(h2_spike) \
        #               + torch.count_nonzero(h3_spike)
        n_neurons = torch.numel(h2h1_spike) \
                    + torch.numel(h2h2_spike) \
                    + torch.numel(h2h3_spike)
        # fire_rate = n_nonzeros / n_neurons
        fire_rate = torch.true_divide(n_nonzeros, n_neurons).float()

        return output, final_state, fire_rate, hidden_spike, hidden_mem, hidden_theta  # n_nonzeros/n_neurons


class SRNN_ALIF_fix(nn.Module):
    def __init__(self, in_size=100, bias=True, n_out=2, tau_a=600):
        super().__init__()
        self.input_size = in_size
        self.stride = in_size  # input_size
        self.output_size = n_out  # 1960  # output_siz
        self.tau_a = tau_a
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        self.tau_adp_h1 = self.tau_adp_h2 = self.tau_adp_h3 = self.tau_a
        self.tau_m_h1 = self.tau_m_h2 = self.tau_m_h3 = tau_m

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])

        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        self._initial_parameters()
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)

    def init_state(self, batch_size):
        h2h1_mem = h2h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(batch_size, cfg_fc[2], device=device)
        b_h1 = b_h2 = b_h3 = b_j0
        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        b_list = [b_h1, b_h2, b_h3]
        init_state_ = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       'b_list': b_list}
        return init_state_

    def forward(self, input, init_state):
        N = input.size(0)  # batch_size
        seq_len = input.size(1)  # seq_len = seq length * char_duration

        mem_list = init_state['mem_list']
        spike_list = init_state['spike_list']
        b_list = init_state['b_list']
        h2h1_mem = mem_list[0]
        h2h2_mem = mem_list[1]
        h2h3_mem = mem_list[2]
        h2h1_spike = spike_list[0]
        h2h2_spike = spike_list[1]
        h2h3_spike = spike_list[2]
        b_h1 = b_list[0]
        b_h2 = b_list[1]
        b_h3 = b_list[2]

        # output_sum = torch.zeros(11, N, self.output_size, device=device)
        outputs = torch.zeros(seq_len, N, self.output_size, device=device)

        hidden_spikes = torch.zeros(seq_len, N, cfg_fc[2], device=device)
        hidden_mems = torch.zeros(seq_len, N, cfg_fc[2], device=device)
        hidden_thetas = torch.zeros(seq_len, N, cfg_fc[2], device=device)

        n_neurons = 0.
        n_nonzeros = 0.

        # input = input.unsqueeze(1)  # [bs] -> [bs, 1]
        # input = self.Embedding(input)  # [bs, 1, 128]
        #
        # # input = np.squeeze(input)
        # input = input.view(N, -1)  # [N, 128]

        # T = 128 // self.input_size  # 128 // 8 = 16
        T = seq_len
        for step in range(T):
            input_x = np.squeeze(input[:, step, :]).view(N, -1)  # .reshape(-1, self.input_size)  # [bs, 100]

            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_h1 = mem_update_adp(h1_input, h2h1_mem, h2h1_spike,
                                                                  self.tau_adp_h1, b_h1, self.tau_m_h1,
                                                                  )

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_h2 = mem_update_adp(h2_input, h2h2_mem, h2h2_spike,
                                                                  self.tau_adp_h2, b_h2, self.tau_m_h2,
                                                                  )

            h3_input = self.i2h_3(h2h2_spike) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_h3 = mem_update_adp(h3_input, h2h3_mem, h2h3_spike,
                                                                  self.tau_adp_h3, b_h3, self.tau_m_h3,
                                                                  )
            hidden_spikes[step] = h2h3_spike
            hidden_mems[step] = h2h3_mem
            hidden_thetas[step] = theta_h3

            h2o3_mem = self.h2o_3(h2h3_mem)

            # output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            outputs[step] = h2o3_mem

        # output = output_sum / T  # [bs, 1979]
        output = outputs.permute(1, 0, 2)  # [seq_len, bs, 2] --> [bs, seq_len, 2]
        hidden_spike = hidden_spikes.permute(1, 0, 2)
        hidden_mem = hidden_mems.permute(1, 0, 2)
        hidden_theta = hidden_thetas.permute(1, 0, 2)

        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        b_list = [b_h1, b_h2, b_h3]
        final_state = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       'b_list': b_list}

        #### Calculate spike count ####
        n_nonzeros = torch.tensor(torch.nonzero(h2h1_spike).shape[0]) \
                     + torch.tensor(torch.nonzero(h2h2_spike).shape[0]) \
                     + torch.tensor(torch.nonzero(h2h3_spike).shape[0])
        # n_nonzeros = torch.count_nonzero(h2h1_spike) \
        #              + torch.count_nonzero(h2h2_spike) \
        #              + torch.count_nonzero(h2h3_spike)
        n_neurons = torch.numel(h2h1_spike) \
                    + torch.numel(h2h2_spike) \
                    + torch.numel(h2h3_spike)
        # fire_rate = n_nonzeros / n_neurons
        fire_rate = torch.true_divide(n_nonzeros, n_neurons).float()

        return output, final_state, fire_rate, hidden_spike, hidden_mem, hidden_theta  # n_nonzeros/n_neurons


class LSNN_ALIF(nn.Module):
    def __init__(self, in_size=100, bias=True, n_out=2):
        super().__init__()
        self.input_size = in_size
        self.stride = in_size  # input_size
        self.output_size = n_out  # 1960  # output_siz
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        # self.tau_adp_h1 = self.tau_adp_h2 = self.tau_adp_h3 = self.tau_a
        self.tau_m_h1 = self.tau_m_h2 = self.tau_m_h3 = tau_m

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])

        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        self._initial_parameters()
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)

        nn.init.constant_(self.tau_adp_h1, 600)
        nn.init.constant_(self.tau_adp_h2, 600)
        nn.init.constant_(self.tau_adp_h3, 600)

    def init_state(self, batch_size):
        h2h1_mem = h2h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(batch_size, cfg_fc[2], device=device)
        b_h1 = b_h2 = b_h3 = b_j0
        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        b_list = [b_h1, b_h2, b_h3]
        init_state_ = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       'b_list': b_list}
        return init_state_

    def forward(self, input, init_state):
        N = input.size(0)  # batch_size
        seq_len = input.size(1)  # seq_len = seq length * char_duration

        mem_list = init_state['mem_list']
        spike_list = init_state['spike_list']
        b_list = init_state['b_list']
        h2h1_mem = mem_list[0]
        h2h2_mem = mem_list[1]
        h2h3_mem = mem_list[2]
        h2h1_spike = spike_list[0]
        h2h2_spike = spike_list[1]
        h2h3_spike = spike_list[2]
        b_h1 = b_list[0]
        b_h2 = b_list[1]
        b_h3 = b_list[2]

        # output_sum = torch.zeros(11, N, self.output_size, device=device)
        outputs = torch.zeros(seq_len, N, self.output_size, device=device)

        hidden_spikes = torch.zeros(seq_len, N, cfg_fc[2], device=device)

        n_neurons = 0.
        n_nonzeros = 0.

        # input = input.unsqueeze(1)  # [bs] -> [bs, 1]
        # input = self.Embedding(input)  # [bs, 1, 128]
        #
        # # input = np.squeeze(input)
        # input = input.view(N, -1)  # [N, 128]

        # T = 128 // self.input_size  # 128 // 8 = 16
        T = seq_len
        for step in range(T):
            input_x = np.squeeze(input[:, step, :]).view(N, -1)  # .reshape(-1, self.input_size)  # [bs, 100]

            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_h1 = mem_update_adp(h1_input, h2h1_mem, h2h1_spike,
                                                                  self.tau_adp_h1, b_h1, self.tau_m_h1,
                                                                  )

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_h2 = mem_update_adp(h2_input, h2h2_mem, h2h2_spike,
                                                                  self.tau_adp_h2, b_h2, self.tau_m_h2,
                                                                  )

            h3_input = self.i2h_3(h2h2_spike) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_h3 = mem_update_adp(h3_input, h2h3_mem, h2h3_spike,
                                                                  self.tau_adp_h3, b_h3, self.tau_m_h3,
                                                                  )
            hidden_spikes[step] = h2h3_spike
            h2o3_mem = self.h2o_3(h2h3_mem)

            # output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            outputs[step] = h2o3_mem

        # output = output_sum / T  # [bs, 1979]
        output = outputs.permute(1, 0, 2)  # [seq_len, bs, 2] --> [bs, seq_len, 2]
        hidden_spike = hidden_spikes.permute(1, 0, 2)
        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        b_list = [b_h1, b_h2, b_h3]
        final_state = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       'b_list': b_list}

        #### Calculate spike count ####
        n_nonzeros = torch.count_nonzero(h2h1_spike) \
                     + torch.count_nonzero(h2h2_spike) \
                     + torch.count_nonzero(h2h3_spike)
        n_neurons = torch.numel(h2h1_spike) \
                    + torch.numel(h2h2_spike) \
                    + torch.numel(h2h3_spike)
        fire_rate = n_nonzeros / n_neurons

        return output, final_state, fire_rate, hidden_spike  # n_nonzeros/n_neurons


class Gen_skip_LSNN_ALIF(nn.Module):
    def __init__(self, in_size=100, bias=True, n_out=2):
        super().__init__()
        self.input_size = in_size
        self.stride = in_size  # input_size
        self.output_size = n_out  # 1960  # output_siz
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        # self.tau_adp_h1 = self.tau_adp_h2 = self.tau_adp_h3 = self.tau_a
        self.tau_m_h1 = self.tau_m_h2 = self.tau_m_h3 = tau_m

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])

        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        self._initial_parameters()
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

        self.mask1 = self.create_general_mask(cfg_fc[0], cycle_min[0], cycle_max[0], duty_cycle_min[0],
                                              duty_cycle_max[0], phase_max[0])
        self.mask2 = self.create_general_mask(cfg_fc[1], cycle_min[1], cycle_max[1], duty_cycle_min[1],
                                              duty_cycle_max[1], phase_max[1])
        self.mask3 = self.create_general_mask(cfg_fc[2], cycle_min[2], cycle_max[2], duty_cycle_min[2],
                                              duty_cycle_max[2], phase_max[2])

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)

        nn.init.constant_(self.tau_adp_h1, 600)
        nn.init.constant_(self.tau_adp_h2, 600)
        nn.init.constant_(self.tau_adp_h3, 600)

    def init_state(self, batch_size):
        h2h1_mem = h2h1_spike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(batch_size, cfg_fc[2], device=device)
        b_h1 = b_h2 = b_h3 = b_j0
        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        b_list = [b_h1, b_h2, b_h3]
        init_state_ = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       'b_list': b_list}
        return init_state_

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

    def forward(self, input, init_state):
        N = input.size(0)  # batch_size
        seq_len = input.size(1)  # seq_len = seq length * char_duration

        mem_list = init_state['mem_list']
        spike_list = init_state['spike_list']
        b_list = init_state['b_list']
        h2h1_mem = mem_list[0]
        h2h2_mem = mem_list[1]
        h2h3_mem = mem_list[2]
        h2h1_spike = spike_list[0]
        h2h2_spike = spike_list[1]
        h2h3_spike = spike_list[2]
        b_h1 = b_list[0]
        b_h2 = b_list[1]
        b_h3 = b_list[2]

        # output_sum = torch.zeros(11, N, self.output_size, device=device)
        outputs = torch.zeros(seq_len, N, self.output_size, device=device)

        hidden_spikes = torch.zeros(seq_len, N, cfg_fc[2], device=device)

        n_neurons = 0.
        n_nonzeros = 0.

        # input = input.unsqueeze(1)  # [bs] -> [bs, 1]
        # input = self.Embedding(input)  # [bs, 1, 128]
        #
        # # input = np.squeeze(input)
        # input = input.view(N, -1)  # [N, 128]

        # T = 128 // self.input_size  # 128 // 8 = 16
        T = seq_len
        for step in range(T):
            input_x = np.squeeze(input[:, step, :]).view(N, -1)  # .reshape(-1, self.input_size)  # [bs, 100]

            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_h1 = mem_update_adp_skip(h1_input, h2h1_mem, h2h1_spike,
                                                                       self.tau_adp_h1, b_h1, self.tau_m_h1,
                                                                       self.mask1[:, step])

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_h2 = mem_update_adp_skip(h2_input, h2h2_mem, h2h2_spike,
                                                                       self.tau_adp_h2, b_h2, self.tau_m_h2,
                                                                       self.mask2[:, step])

            h3_input = self.i2h_3(h2h2_spike) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_h3 = mem_update_adp_skip(h3_input, h2h3_mem, h2h3_spike,
                                                                       self.tau_adp_h3, b_h3, self.tau_m_h3,
                                                                       self.mask3[:, step])
            hidden_spikes[step] = h2h3_spike
            h2o3_mem = self.h2o_3(h2h3_mem)

            # output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
            outputs[step] = h2o3_mem

        # output = output_sum / T  # [bs, 1979]
        output = outputs.permute(1, 0, 2)  # [seq_len, bs, 2] --> [bs, seq_len, 2]
        hidden_spike = hidden_spikes.permute(1, 0, 2)
        mem_list = [h2h1_mem, h2h2_mem, h2h3_mem]
        spike_list = [h2h1_spike, h2h2_spike, h2h3_spike]
        b_list = [b_h1, b_h2, b_h3]
        final_state = {'mem_list': mem_list,
                       'spike_list': spike_list,
                       'b_list': b_list}

        #### Calculate spike count ####
        n_nonzeros = torch.count_nonzero(h2h1_spike) \
                     + torch.count_nonzero(h2h2_spike) \
                     + torch.count_nonzero(h2h3_spike)
        n_neurons = torch.numel(h2h1_spike) \
                    + torch.numel(h2h2_spike) \
                    + torch.numel(h2h3_spike)
        fire_rate = n_nonzeros / n_neurons

        return output, final_state, fire_rate, hidden_spike  # n_nonzeros/n_neurons





