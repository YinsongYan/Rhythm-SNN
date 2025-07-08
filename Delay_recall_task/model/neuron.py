import torch
import torch.nn as nn
import math
import sys
sys.path.append("..")
from model.Hyperparameters_recall import args
# from main_delayed_recll import args

"""
    Altered from https://github.com/byin-cwi/Efficient-spiking-networks
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

b_j0 = 0.01  # neural threshold baseline
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
lens = 0.5




# RhythmSNN initialized parameters
phase_max = args.phase_max
cycle_min = args.cycle_min
cycle_max = args.cycle_max
duty_cycle_min = args.duty_cycle_min
duty_cycle_max = args.duty_cycle_max





grads = []
def save_grad(name,step):
    def hook(grad):
        grads.append(grad)
    return hook

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        # temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        # temp =  gaussian(input, mu=0., sigma=lens)
        return grad_input * temp.float() * gamma


act_fun_adp = ActFun_adp.apply


def mem_update_adp(inputs, mem, spike, tau_adp, tau_m, b, dt=1, isAdapt=1):
    #     tau_adp = torch.FloatTensor([tau_adp])
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.
    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b
    # B = 0.3 + beta * b

    #mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    mem = mem * alpha + R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b


def mem_update_adp_skip(inputs, mem, spike, tau_adp, b, tau_m, mask, dt=1, isAdapt=1):
    mask = mask.expand(mem.size(0), -1)
    pre_mem = mem
    tau_m_ = torch.tensor(tau_m) if not isinstance(tau_m, torch.Tensor) else tau_m
    tau_adp_ = torch.tensor(tau_adp) if not isinstance(tau_adp, torch.Tensor) else tau_adp
    alpha = torch.exp(-1. * dt / tau_m_).to(inputs.device)
    ro = torch.exp(-1. * dt / tau_adp_).to(inputs.device)
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.
    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b
    # mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    mem = mem * alpha + R_m * inputs - B * spike * dt

    mem = torch.where(mask == 0, pre_mem, mem)
    inputs_ = mem - B
    spike = act_fun_adp(inputs_) * mask  # act_fun : approximation firing function
    return mem, spike, B, b



def mem_update_adp_DEXAT_skip(inputs, mem, spike, tau_a1, tau_a2, tau_m, b1, b2, mask, dt=1, isAdapt=1):
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
        beta1 = 0.0
        beta2 = 1.8
    else:
        beta1 = 0.
        beta2 = 0.
    b1 = ro1 * b1 + (1 - ro1) * spike
    b2 = ro2 * b2 + (1 - ro2) * spike
    B = b_j0 + beta1 * b1 + beta2 * b2
    # B = b_j0 + beta2 * b2
    # mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    mem = mem * alpha + R_m * inputs - B * spike * dt

    mem = torch.where(mask == 0, pre_mem, mem)
    inputs_ = mem - B
    spike = act_fun_adp(inputs_) * mask  # act_fun : approximation firing function
    return mem, spike, B, b1, b2


def mem_update_adp_DEXAT(inputs, mem, spike, tau_a1, tau_a2, tau_m, b1, b2, dt=1, isAdapt=1):
    tau_m_ = torch.tensor(tau_m, dtype=torch.float32) if not isinstance(tau_m, torch.Tensor) else tau_m
    tau_a1_ = torch.tensor(tau_a1, dtype=torch.float32) if not isinstance(tau_a1, torch.Tensor) else tau_a1
    tau_a2_ = torch.tensor(tau_a2, dtype=torch.float32) if not isinstance(tau_a2, torch.Tensor) else tau_a2

    alpha = torch.exp(-1. * dt / tau_m_).to(inputs.device)
    ro1 = torch.exp(-1. * dt / tau_a1_).to(inputs.device)
    ro2 = torch.exp(-1. * dt / tau_a2_).to(inputs.device)
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta1 = 0.36
        beta2 = 1.8
    else:
        beta1 = 0.
        beta2 = 0.
    b1 = ro1 * b1 + (1 - ro1) * spike
    b2 = ro2 * b2 + (1 - ro2) * spike
    B = b_j0 + beta1 * b1 + beta2 * b2
    # mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    mem = mem * alpha + R_m * inputs - B * spike * dt

    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b1, b2




def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem



class ALIF(nn.Module):
    def __init__(self, input_size=2, hidden_dims=[64, 64, 64], output_size=1, time_window=1):
        super(ALIF, self).__init__()

        self.time_window = time_window
        self.input_size = input_size
        self.output_size = output_size

        # self.relu = nn.ReLU()

        self.r1_dim = hidden_dims[0]
        self.r2_dim = hidden_dims[1]
        self.r3_dim = hidden_dims[2]
        self.i2h_1 = nn.Linear(input_size, self.r1_dim)
        self.h2h_1 = nn.Linear(self.r1_dim, self.r1_dim)

        self.i2h_2 = nn.Linear(self.r1_dim, self.r2_dim)
        self.h2h_2 = nn.Linear(self.r2_dim, self.r2_dim)

        self.i2h_3 = nn.Linear(self.r2_dim, self.r3_dim)
        self.h2o_3 = nn.Linear(self.r3_dim, self.output_size)
        self.act = nn.Sigmoid()

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(self.r3_dim))

        self.tau_m_h1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_m_h3 = nn.Parameter(torch.Tensor(self.r3_dim))

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


        nn.init.normal_(self.tau_adp_h1, 700, 25)
        nn.init.normal_(self.tau_adp_h2, 700, 25)
        nn.init.normal_(self.tau_adp_h3, 700, 25)

        nn.init.normal_(self.tau_m_h1, 20, 5)
        nn.init.normal_(self.tau_m_h2, 20, 5)
        nn.init.normal_(self.tau_m_h3, 20, 5)

        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = self.b_d1 = 0

    def compute_input_steps(self, seq_num):
        return int(seq_num / self.stride)

    def forward(self, input, task='duration'):
        batch_size, hid_dim, time_length = input.shape
        # print('batch_size, hid_dim, time_length:', batch_size, hid_dim, time_length)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        h2h2_mem = h2h2_spike = torch.rand(batch_size, self.r2_dim).cuda()
        h2h3_mem = h2h3_spike = torch.rand(batch_size, self.r3_dim).cuda()

        input = input
        input_steps = self.time_window

        out = []
        for i in range(input_steps):
            input_x = input[:, :, i]
            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp(h1_input, h2h1_mem, h2h1_spike, self.tau_adp_h1, self.tau_m_h1, self.b_h1)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp(h2_input, h2h2_mem, h2h2_spike, self.tau_adp_h2, self.tau_m_h2, self.b_h2)

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp(h3_input, h2h3_mem, h2h3_spike, self.tau_adp_h3, self.tau_m_h3, self.b_h3)

            h2o3_mem = self.act(self.h2o_3(h2h3_spike))
            out.append((h2o3_mem))
        output = torch.stack(out, dim=2)
        if task == 'duration':
            return output[:,:,-1] # duration
        elif task == 'syn':
            return output # Synchronization
        elif task == 'interval':
            return output[:, :, -1] # add
        elif task == 'recall':
            return output # Copy

    def get_grads(self, input):
        batch_size, seq_num, input_dim = input.shape
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        h2h2_mem = h2h2_spike = torch.rand(batch_size, self.r2_dim).cuda()
        h2h3_mem = h2h3_spike = torch.rand(batch_size, self.r3_dim).cuda()

        input = input
        input_steps = self.time_window
        out_ = []
        for i in range(input_steps):
            input_x = input[:, :, i]
            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp(h1_input, h2h1_mem, h2h1_spike, self.tau_adp_h1,
                                                                       self.tau_m_h1, self.b_h1)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp(h2_input, h2h2_mem, h2h2_spike, self.tau_adp_h2,
                                                                       self.tau_m_h2, self.b_h2)

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp(h3_input, h2h3_mem, h2h3_spike, self.tau_adp_h3,
                                                                       self.tau_m_h3, self.b_h3)

            h2o3_mem = self.act(self.h2o_3(h2h3_spike))
            h2h3_spike.register_hook(save_grad('grads', i))
            out_.append(h2o3_mem)
        out = torch.stack(out_, dim=2)
        return out, grads


class RhythmALIF(nn.Module):
    def __init__(self, input_size=2, hidden_dims=[64, 64, 64], output_size=1, time_window=1):
        super(RhythmALIF, self).__init__()

        self.time_window = time_window
        self.input_size = input_size
        self.output_size = output_size

        # self.relu = nn.ReLU()

        self.r1_dim = hidden_dims[0]
        self.r2_dim = hidden_dims[1]
        self.r3_dim = hidden_dims[2]
        self.i2h_1 = nn.Linear(input_size, self.r1_dim)
        self.h2h_1 = nn.Linear(self.r1_dim, self.r1_dim)

        self.i2h_2 = nn.Linear(self.r1_dim, self.r2_dim)
        self.h2h_2 = nn.Linear(self.r2_dim, self.r2_dim)

        self.i2h_3 = nn.Linear(self.r2_dim, self.r3_dim)
        self.h2o_3 = nn.Linear(self.r3_dim, self.output_size)
        self.act = nn.Sigmoid()

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(self.r3_dim))

        self.tau_m_h1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_m_h3 = nn.Parameter(torch.Tensor(self.r3_dim))

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

        nn.init.normal_(self.tau_adp_h1, 700, 25)
        nn.init.normal_(self.tau_adp_h2, 700, 25)
        nn.init.normal_(self.tau_adp_h3, 700, 25)

        nn.init.normal_(self.tau_m_h1, 20, 5)
        nn.init.normal_(self.tau_m_h2, 20, 5)
        nn.init.normal_(self.tau_m_h3, 20, 5)

        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = self.b_d1 = 0

        self.mask1 = self.create_general_mask(self.r1_dim, cycle_min[0], cycle_max[0], duty_cycle_min[0],
                                              duty_cycle_max[0], phase_max[0], time_window)
        self.mask2 = self.create_general_mask(self.r2_dim, cycle_min[1], cycle_max[1], duty_cycle_min[1],
                                              duty_cycle_max[1], phase_max[1], time_window)
        self.mask3 = self.create_general_mask(self.r3_dim, cycle_min[2], cycle_max[2], duty_cycle_min[2],
                                              duty_cycle_max[2], phase_max[2], time_window)

    def create_general_mask(self, dim=128, c_min=4, c_max=8, min_dc=0.1, max_dc=0.9, phase_shift_max=0.5, T=7*256):
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

    def compute_input_steps(self, seq_num):
        return int(seq_num / self.stride)

    def forward(self, input, task='duration'):
        batch_size, hid_dim, time_length = input.shape
        # print('batch_size, hid_dim, time_length:', batch_size, hid_dim, time_length)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        h2h2_mem = h2h2_spike = torch.rand(batch_size, self.r2_dim).cuda()
        h2h3_mem = h2h3_spike = torch.rand(batch_size, self.r3_dim).cuda()

        input = input
        input_steps = self.time_window

        out = []
        for i in range(input_steps):
            input_x = input[:, :, i]
            # h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            # h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp(h1_input, h2h1_mem, h2h1_spike, self.tau_adp_h1,
            #                                                            self.tau_m_h1, self.b_h1)
            #
            # h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            # h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp(h2_input, h2h2_mem, h2h2_spike, self.tau_adp_h2,
            #                                                            self.tau_m_h2, self.b_h2)
            #
            # h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            # h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp(h3_input, h2h3_mem, h2h3_spike, self.tau_adp_h3,
            #                                                            self.tau_m_h3, self.b_h3)
            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp_skip(h1_input, h2h1_mem, h2h1_spike, self.tau_adp_h1, self.b_h1, self.tau_m_h1, self.mask1[:, i])

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp_skip(h2_input, h2h2_mem, h2h2_spike, self.tau_adp_h2, self.b_h2, self.tau_m_h2, self.mask2[:, i])

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp_skip(h3_input, h2h3_mem, h2h3_spike, self.tau_adp_h3, self.b_h3, self.tau_m_h3, self.mask3[:, i])

            h2o3_mem = self.act(self.h2o_3(h2h3_spike))
            out.append((h2o3_mem))
        output = torch.stack(out, dim=2)
        if task == 'duration':
            return output[:, :, -1]  # duration
        elif task == 'syn':
            return output  # Synchronization
        elif task == 'interval':
            return output[:, :, -1]  # add
        elif task == 'recall':
            return output  # Copy

    def get_grads(self, input):
        batch_size, seq_num, input_dim = input.shape
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        h2h2_mem = h2h2_spike = torch.rand(batch_size, self.r2_dim).cuda()
        h2h3_mem = h2h3_spike = torch.rand(batch_size, self.r3_dim).cuda()

        input = input
        input_steps = self.time_window
        out_ = []
        for i in range(input_steps):
            input_x = input[:, :, i]
            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp_skip(h1_input, h2h1_mem, h2h1_spike,
                                                                            self.tau_adp_h1, self.b_h1, self.tau_m_h1,
                                                                            self.mask1[:, i])

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp_skip(h2_input, h2h2_mem, h2h2_spike,
                                                                            self.tau_adp_h2, self.b_h2, self.tau_m_h2,
                                                                            self.mask2[:, i])

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp_skip(h3_input, h2h3_mem, h2h3_spike,
                                                                            self.tau_adp_h3, self.b_h3, self.tau_m_h3,
                                                                            self.mask2[:, i])

            h2o3_mem = self.act(self.h2o_3(h2h3_spike))
            h2h3_spike.register_hook(save_grad('grads', i))
            out_.append(h2o3_mem)
        out = torch.stack(out_, dim=2)
        return out, grads




class DEXAT(nn.Module):
    def __init__(self, input_size=2, hidden_dims=[64, 64, 64], output_size=1, time_window=1):
        super(DEXAT, self).__init__()

        self.time_window = time_window
        self.input_size = input_size
        self.output_size = output_size

        # self.relu = nn.ReLU()

        self.r1_dim = hidden_dims[0]
        self.r2_dim = hidden_dims[1]
        self.r3_dim = hidden_dims[2]
        self.i2h_1 = nn.Linear(input_size, self.r1_dim)
        self.h2h_1 = nn.Linear(self.r1_dim, self.r1_dim)

        self.i2h_2 = nn.Linear(self.r1_dim, self.r2_dim)
        self.h2h_2 = nn.Linear(self.r2_dim, self.r2_dim)

        self.i2h_3 = nn.Linear(self.r2_dim, self.r3_dim)
        self.h2o_3 = nn.Linear(self.r3_dim, self.output_size)
        self.act = nn.Sigmoid()

        self.tau_adp_h11 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_h12 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_h21 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_adp_h22 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_h31 = nn.Parameter(torch.Tensor(self.r3_dim))
        self.tau_adp_h32 = nn.Parameter(torch.Tensor(self.r3_dim))

        self.tau_m_h1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_m_h3 = nn.Parameter(torch.Tensor(self.r3_dim))

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


        nn.init.normal_(self.tau_adp_h1, 700, 25)
        nn.init.normal_(self.tau_adp_h2, 700, 25)
        nn.init.normal_(self.tau_adp_h3, 700, 25)

        nn.init.normal_(self.tau_m_h1, 20, 5)
        nn.init.normal_(self.tau_m_h2, 20, 5)
        nn.init.normal_(self.tau_m_h3, 20, 5)

        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = self.b_d1 = 0

    def compute_input_steps(self, seq_num):
        return int(seq_num / self.stride)

    def forward(self, input, task='duration'):
        batch_size, hid_dim, time_length = input.shape
        # print('batch_size, hid_dim, time_length:', batch_size, hid_dim, time_length)
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        h2h2_mem = h2h2_spike = torch.rand(batch_size, self.r2_dim).cuda()
        h2h3_mem = h2h3_spike = torch.rand(batch_size, self.r3_dim).cuda()

        b_r11 = b_r21 = b_r31 = 0
        b_r12 = b_r22 = b_r32 = b_j0

        input = input
        input_steps = self.time_window

        out = []
        for i in range(input_steps):
            input_x = input[:, :, i]
            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_r11, b_r12 = mem_update_adp_DEXAT(h1_input, h2h1_mem, h2h1_spike, self.tau_adp_h11, self.tau_adp_h12, self.tau_m_h1, b_r11, b_r12)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_r21, b_r22 = mem_update_adp_DEXAT(h2_input, h2h2_mem, h2h2_spike, self.tau_adp_h21, self.tau_adp_h22, self.tau_m_h2, b_r21, b_r22)

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_r31, b_r32 = mem_update_adp_DEXAT(h3_input, h2h3_mem, h2h3_spike, self.tau_adp_h31, self.tau_adp_h32, self.tau_m_h3, b_r31, b_r32)

            h2o3_mem = self.act(self.h2o_3(h2h3_spike))
            out.append((h2o3_mem))
        output = torch.stack(out, dim=2)
        if task == 'duration':
            return output[:,:,-1] # duration
        elif task == 'syn':
            return output # Synchronization
        elif task == 'interval':
            return output[:, :, -1] # add
        elif task == 'recall':
            return output # Copy

    def get_grads(self, input):
        batch_size, seq_num, input_dim = input.shape
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        h2h2_mem = h2h2_spike = torch.rand(batch_size, self.r2_dim).cuda()
        h2h3_mem = h2h3_spike = torch.rand(batch_size, self.r3_dim).cuda()

        b_r11 = b_r21 = b_r31 = 0
        b_r12 = b_r22 = b_r32 = b_j0

        input = input
        input_steps = self.time_window
        out_ = []
        for i in range(input_steps):
            input_x = input[:, :, i]
            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_r11, b_r12 = mem_update_adp_DEXAT(h1_input, h2h1_mem, h2h1_spike,
                                                                                self.tau_adp_h11, self.tau_adp_h12,
                                                                                self.tau_m_h1, b_r11, b_r12)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_r21, b_r22 = mem_update_adp_DEXAT(h2_input, h2h2_mem, h2h2_spike,
                                                                                self.tau_adp_h21, self.tau_adp_h22,
                                                                                self.tau_m_h2, b_r21, b_r22)

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_r31, b_r32 = mem_update_adp_DEXAT(h3_input, h2h3_mem, h2h3_spike,
                                                                                self.tau_adp_h31, self.tau_adp_h32,
                                                                                self.tau_m_h3, b_r31, b_r32)

            h2o3_mem = self.act(self.h2o_3(h2h3_spike))
            h2h3_spike.register_hook(save_grad('grads', i))
            out_.append(h2o3_mem)
        out = torch.stack(out_, dim=2)
        return out, grads


class RhythmDEXAT(nn.Module):
    def __init__(self, input_size=2, hidden_dims=[64, 64, 64], output_size=1, time_window=1):
        super(RhythmDEXAT, self).__init__()

        self.time_window = time_window
        self.input_size = input_size
        self.output_size = output_size

        # self.relu = nn.ReLU()

        self.r1_dim = hidden_dims[0]
        self.r2_dim = hidden_dims[1]
        self.r3_dim = hidden_dims[2]
        self.i2h_1 = nn.Linear(input_size, self.r1_dim)
        self.h2h_1 = nn.Linear(self.r1_dim, self.r1_dim)

        self.i2h_2 = nn.Linear(self.r1_dim, self.r2_dim)
        self.h2h_2 = nn.Linear(self.r2_dim, self.r2_dim)

        self.i2h_3 = nn.Linear(self.r2_dim, self.r3_dim)
        self.h2o_3 = nn.Linear(self.r3_dim, self.output_size)
        self.act = nn.Sigmoid()

        self.tau_adp_h11 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_h12 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_h21 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_adp_h22 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_h31 = nn.Parameter(torch.Tensor(self.r3_dim))
        self.tau_adp_h32 = nn.Parameter(torch.Tensor(self.r3_dim))

        self.tau_m_h1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_m_h3 = nn.Parameter(torch.Tensor(self.r3_dim))

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


        nn.init.normal_(self.tau_adp_h1, 700, 25)
        nn.init.normal_(self.tau_adp_h2, 700, 25)
        nn.init.normal_(self.tau_adp_h3, 700, 25)

        nn.init.normal_(self.tau_m_h1, 20, 5)
        nn.init.normal_(self.tau_m_h2, 20, 5)
        nn.init.normal_(self.tau_m_h3, 20, 5)

        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = self.b_d1 = 0

        self.mask1 = self.create_general_mask(self.r1_dim, cycle_min[0], cycle_max[0], duty_cycle_min[0],
                                              duty_cycle_max[0], phase_max[0], time_window)
        self.mask2 = self.create_general_mask(self.r2_dim, cycle_min[1], cycle_max[1], duty_cycle_min[1],
                                              duty_cycle_max[1], phase_max[1], time_window)
        self.mask3 = self.create_general_mask(self.r3_dim, cycle_min[2], cycle_max[2], duty_cycle_min[2],
                                              duty_cycle_max[2], phase_max[2], time_window)

    def create_general_mask(self, dim=128, c_min=4, c_max=8, min_dc=0.1, max_dc=0.9, phase_shift_max=0.5, T=7 * 256):
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

    def compute_input_steps(self, seq_num):
        return int(seq_num / self.stride)

    def forward(self, input, task='duration'):
        batch_size, hid_dim, time_length = input.shape
        # print('batch_size, hid_dim, time_length:', batch_size, hid_dim, time_length)
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        h2h2_mem = h2h2_spike = torch.rand(batch_size, self.r2_dim).cuda()
        h2h3_mem = h2h3_spike = torch.rand(batch_size, self.r3_dim).cuda()

        b_r11 = b_r21 = b_r31 = 0
        b_r12 = b_r22 = b_r32 = b_j0

        input = input
        input_steps = self.time_window

        out = []
        for i in range(input_steps):
            input_x = input[:, :, i]
            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_r11, b_r12 = mem_update_adp_DEXAT(h1_input, h2h1_mem, h2h1_spike, self.tau_adp_h11, self.tau_adp_h12, self.tau_m_h1, b_r11, b_r12)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_r21, b_r22 = mem_update_adp_DEXAT(h2_input, h2h2_mem, h2h2_spike, self.tau_adp_h21, self.tau_adp_h22, self.tau_m_h2, b_r21, b_r22)

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_r31, b_r32 = mem_update_adp_DEXAT(h3_input, h2h3_mem, h2h3_spike, self.tau_adp_h31, self.tau_adp_h32, self.tau_m_h3, b_r31, b_r32)

            h2o3_mem = self.act(self.h2o_3(h2h3_spike))
            out.append((h2o3_mem))
        output = torch.stack(out, dim=2)
        if task == 'duration':
            return output[:,:,-1] # duration
        elif task == 'syn':
            return output # Synchronization
        elif task == 'interval':
            return output[:, :, -1] # add
        elif task == 'recall':
            return output # Copy

    def get_grads(self, input):
        batch_size, seq_num, input_dim = input.shape
        # self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0

        h2h1_mem = h2h1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        h2h2_mem = h2h2_spike = torch.rand(batch_size, self.r2_dim).cuda()
        h2h3_mem = h2h3_spike = torch.rand(batch_size, self.r3_dim).cuda()

        b_r11 = b_r21 = b_r31 = 0
        b_r12 = b_r22 = b_r32 = b_j0

        input = input
        input_steps = self.time_window
        out_ = []
        for i in range(input_steps):
            input_x = input[:, :, i]
            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, b_r11, b_r12 = mem_update_adp_DEXAT(h1_input, h2h1_mem, h2h1_spike,
                                                                                self.tau_adp_h11, self.tau_adp_h12,
                                                                                self.tau_m_h1, b_r11, b_r12)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, b_r21, b_r22 = mem_update_adp_DEXAT(h2_input, h2h2_mem, h2h2_spike,
                                                                                self.tau_adp_h21, self.tau_adp_h22,
                                                                                self.tau_m_h2, b_r21, b_r22)

            h3_input = self.i2h_3(h2h2_spike)  # + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, b_r31, b_r32 = mem_update_adp_DEXAT(h3_input, h2h3_mem, h2h3_spike,
                                                                                self.tau_adp_h31, self.tau_adp_h32,
                                                                                self.tau_m_h3, b_r31, b_r32)

            h2o3_mem = self.act(self.h2o_3(h2h3_spike))
            h2h3_spike.register_hook(save_grad('grads', i))
            out_.append(h2o3_mem)
        out = torch.stack(out_, dim=2)
        return out, grads



