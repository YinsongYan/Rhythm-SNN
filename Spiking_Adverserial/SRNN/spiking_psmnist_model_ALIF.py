import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from SRNN.Hyperparameters import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

b_j0 = 0.01  # neural threshold baseline
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
lens = 0.5


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

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b


def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem


class RNN_custom(nn.Module):
    def __init__(self, input_size, stride, hidden_dims, output_size, DC_f='adp-spike'):
        super(RNN_custom, self).__init__()
        self.DC_f = DC_f
        self.stride = stride
        self.input_size = input_size
        self.output_size = output_size

        self.r1_dim = hidden_dims[0]
        self.r2_dim = hidden_dims[1]
        self.d1_dim = hidden_dims[2]
        self.i2h = nn.Linear(input_size, self.r1_dim)
        self.h2h = nn.Linear(self.r1_dim, self.r1_dim)
        self.h2d = nn.Linear(self.r1_dim, self.r2_dim)
        self.d2d = nn.Linear(self.r2_dim, self.r2_dim)
        self.dense1 = nn.Linear(self.r2_dim, self.d1_dim)
        self.d2o = nn.Linear(self.d1_dim, self.output_size)

        self.tau_adp_r1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_r2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_adp_d1 = nn.Parameter(torch.Tensor(self.d1_dim))
        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))

        self.tau_m_r1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_m_r2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_m_d1 = nn.Parameter(torch.Tensor(self.d1_dim))
        self.tau_m_o = nn.Parameter(torch.Tensor(self.output_size))

        nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2d.weight)
        nn.init.xavier_uniform_(self.d2d.weight)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.d2o.weight)

        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.h2d.bias, 0)
        nn.init.constant_(self.d2d.bias, 0)
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.constant_(self.d2o.bias, 0)

        nn.init.normal_(self.tau_adp_r1, 700, 25)
        nn.init.normal_(self.tau_adp_r2, 700, 25)
        nn.init.normal_(self.tau_adp_o, 700, 25)
        nn.init.normal_(self.tau_adp_d1, 700, 25)

        nn.init.normal_(self.tau_m_r1, 20, 5)
        nn.init.normal_(self.tau_m_r2, 20, 5)
        nn.init.normal_(self.tau_m_o, 20, 5)
        nn.init.normal_(self.tau_m_d1, 20, 5)

        self.b_r1 = self.b_r2 = self.b_o = self.b_d1 = 0

    def compute_input_steps(self, seq_num):
        return int(seq_num / self.stride)

    def forward(self, input):
        N = input.size(0)
        input = input.squeeze()
        input = input.view(N, -1)  # [N, 784]
        batch_size, seq_num = input.shape
        self.b_r1 = self.b_r2 = self.b_o = self.b_d1 = b_j0

        r1_mem = r1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        r2_mem = r2_spike = torch.rand(batch_size, self.r2_dim).cuda()
        d1_mem = d1_spike = torch.rand(batch_size, self.d1_dim).cuda()
        d2o_spike = output_sumspike = d2o_mem = torch.rand(batch_size, self.output_size).cuda()

        #input = input / 255.
        input_steps = self.compute_input_steps(seq_num)
        r1_spikes = []
        r2_spikes = []
        d1_spikes = []
        d2_spikes = []
        for i in range(input_steps):
            start_idx = i * self.stride
            if start_idx < (seq_num - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)
            # print(input_x.shape)
            h_input = self.i2h(input_x.float()) + self.h2h(r1_spike)
            r1_mem, r1_spike, theta_r1, self.b_r1 = mem_update_adp(h_input, r1_mem, r1_spike, self.tau_adp_r1,
                                                                   self.tau_m_r1, self.b_r1)

            d_input = self.h2d(r1_spike) + self.d2d(r2_spike)
            r2_mem, r2_spike, theta_r2, self.b_r2 = mem_update_adp(d_input, r2_mem, r2_spike, self.tau_adp_r2,
                                                                   self.tau_m_r2, self.b_r2)

            d1_mem, d1_spike, theta_d1, self.b_d1 = mem_update_adp(self.dense1(r2_spike), d1_mem, d1_spike,
                                                                   self.tau_adp_d1, self.tau_m_d1, self.b_d1)

            if self.DC_f[:3] == 'adp':
                d2o_mem, d2o_spike, theta_o, self.b_o = mem_update_adp(self.d2o(d1_spike), d2o_mem, d2o_spike,
                                                                       self.tau_adp_o, self.tau_m_o, self.b_o)
            elif self.DC_f == 'integrator':
                d2o_mem = output_Neuron(self.d2o(d1_spike), d2o_mem, self.tau_m_o)
            if i >= 0:
                if self.DC_f == 'adp-mem':
                    output_sumspike = output_sumspike + F.softmax(d2o_mem, dim=1)
                elif self.DC_f == 'adp-spike':
                    output_sumspike = output_sumspike + d2o_spike
                elif self.DC_f == 'integrator':
                    output_sumspike = output_sumspike + F.softmax(d2o_mem, dim=1)
            r1_spikes.append(r1_spike.detach().cpu().numpy())
            r2_spikes.append(r2_spike.detach().cpu().numpy())
            d1_spikes.append(d1_spike.detach().cpu().numpy())
            d2_spikes.append(d2o_spike.detach().cpu().numpy())
        return output_sumspike #, [r1_spikes, r2_spikes, d1_spikes, d2_spikes]

class RNN_custom_keras(nn.Module):
    def __init__(self, input_size, stride, hidden_dims, output_size, DC_f='adp-spike'):
        super(RNN_custom_keras, self).__init__()
        self.DC_f = DC_f
        self.stride = stride
        self.input_size = input_size
        self.output_size = output_size

        self.r1_dim = hidden_dims[0]
        self.r2_dim = hidden_dims[1]
        self.d1_dim = hidden_dims[2]
        self.i2h = nn.Linear(input_size, self.r1_dim)
        self.h2h = nn.Linear(self.r1_dim, self.r1_dim)
        self.h2d = nn.Linear(self.r1_dim, self.r2_dim)
        self.d2d = nn.Linear(self.r2_dim, self.r2_dim)
        self.dense1 = nn.Linear(self.r2_dim, self.d1_dim)
        self.d2o = nn.Linear(self.d1_dim, self.output_size)

        self.tau_adp_r1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_adp_r2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_adp_d1 = nn.Parameter(torch.Tensor(self.d1_dim))
        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))

        self.tau_m_r1 = nn.Parameter(torch.Tensor(self.r1_dim))
        self.tau_m_r2 = nn.Parameter(torch.Tensor(self.r2_dim))
        self.tau_m_d1 = nn.Parameter(torch.Tensor(self.d1_dim))
        self.tau_m_o = nn.Parameter(torch.Tensor(self.output_size))

        nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2d.weight)
        nn.init.xavier_uniform_(self.d2d.weight)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.d2o.weight)

        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.h2d.bias, 0)
        nn.init.constant_(self.d2d.bias, 0)
        nn.init.constant_(self.dense1.bias, 0)
        nn.init.constant_(self.d2o.bias, 0)

        nn.init.normal_(self.tau_adp_r1, 700, 25)
        nn.init.normal_(self.tau_adp_r2, 700, 25)
        nn.init.normal_(self.tau_adp_o, 700, 25)
        nn.init.normal_(self.tau_adp_d1, 700, 25)

        nn.init.normal_(self.tau_m_r1, 20, 5)
        nn.init.normal_(self.tau_m_r2, 20, 5)
        nn.init.normal_(self.tau_m_o, 20, 5)
        nn.init.normal_(self.tau_m_d1, 20, 5)

        self.b_r1 = self.b_r2 = self.b_o = self.b_d1 = 0

    def compute_input_steps(self, seq_num):
        return int(seq_num / self.stride)

    def forward(self, input):
        N = input.size(0)
        #input = input.squeeze()
        #input = input.view(N, -1)  # [N, 784]
        batch_size, seq_num, input_dim = input.shape
        self.b_r1 = self.b_r2 = self.b_o = self.b_d1 = b_j0

        r1_mem = r1_spike = torch.rand(batch_size, self.r1_dim).cuda()
        r2_mem = r2_spike = torch.rand(batch_size, self.r2_dim).cuda()
        d1_mem = d1_spike = torch.rand(batch_size, self.d1_dim).cuda()
        d2o_spike = output_sumspike = d2o_mem = torch.rand(batch_size, self.output_size).cuda()

        input = input / 255.
        input_steps = self.compute_input_steps(seq_num)
        r1_spikes = []
        r2_spikes = []
        d1_spikes = []
        d2_spikes = []
        for i in range(input_steps):
            start_idx = i * self.stride
            if start_idx < (seq_num - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size, :].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:, :].reshape(-1, self.input_size)
            # print(input_x.shape)
            h_input = self.i2h(input_x.float()) + self.h2h(r1_spike)
            r1_mem, r1_spike, theta_r1, self.b_r1 = mem_update_adp(h_input, r1_mem, r1_spike, self.tau_adp_r1,
                                                                   self.tau_m_r1, self.b_r1)

            d_input = self.h2d(r1_spike) + self.d2d(r2_spike)
            r2_mem, r2_spike, theta_r2, self.b_r2 = mem_update_adp(d_input, r2_mem, r2_spike, self.tau_adp_r2,
                                                                   self.tau_m_r2, self.b_r2)

            d1_mem, d1_spike, theta_d1, self.b_d1 = mem_update_adp(self.dense1(r2_spike), d1_mem, d1_spike,
                                                                   self.tau_adp_d1, self.tau_m_d1, self.b_d1)

            if self.DC_f[:3] == 'adp':
                d2o_mem, d2o_spike, theta_o, self.b_o = mem_update_adp(self.d2o(d1_spike), d2o_mem, d2o_spike,
                                                                       self.tau_adp_o, self.tau_m_o, self.b_o)
            elif self.DC_f == 'integrator':
                d2o_mem = output_Neuron(self.d2o(d1_spike), d2o_mem, self.tau_m_o)
            if i >= 0:
                if self.DC_f == 'adp-mem':
                    output_sumspike = output_sumspike + F.softmax(d2o_mem, dim=1)
                elif self.DC_f == 'adp-spike':
                    output_sumspike = output_sumspike + d2o_spike
                elif self.DC_f == 'integrator':
                    output_sumspike = output_sumspike + F.softmax(d2o_mem, dim=1)
            r1_spikes.append(r1_spike.detach().cpu().numpy())
            r2_spikes.append(r2_spike.detach().cpu().numpy())
            d1_spikes.append(d1_spike.detach().cpu().numpy())
            d2_spikes.append(d2o_spike.detach().cpu().numpy())
        return output_sumspike #, [r1_spikes, r2_spikes, d1_spikes, d2_spikes]