import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from SNN_layers.spike_neuron import *
from SNN_layers.spike_dense import *


#Vanilla SRNN layer
class spike_rnn_test_origin(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 1,dt = 1,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_rnn_test_origin, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim+output_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)


    def set_neuron_state(self,batch_size):
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):

        #concat the the forward inputs with the recurrent inputs 
        k_input = torch.cat((input_spike.float(),self.spike),1)
        #synaptic inputs
        d_input = self.dense(k_input)
        self.mem,self.spike = mem_update_pra(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike


# Gen Rhy SRNN layer
class Rhy_spike_rnn_test_origin(nn.Module):
    def __init__(self, input_dim, output_dim,
                 tau_minitializer='uniform', low_m=0, high_m=4, vth=1, dt=1,
                 cycle_min=2, cycle_max=10,
                 duty_cycle_min=0.10, duty_cycle_max=0.5, phase_max=0.5,
                 device='cpu', bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(Rhy_spike_rnn_test_origin, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim + output_dim, output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp_h = nn.Parameter(torch.Tensor(self.output_dim))

        self.rhy_mask = self.create_general_mask(output_dim, cycle_min, cycle_max, duty_cycle_min, duty_cycle_max,
                                                 phase_max)

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)

        # nn.init.uniform_(self.tau_adp_h, 0, 2)
        nn.init.constant_(self.tau_adp_h, 0)
        self.b_h = 0.3


    def set_neuron_state(self, batch_size):
        self.mem = Variable(torch.rand(batch_size, self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size, self.output_dim)).to(self.device)
        # self.v_th = Variable(torch.ones(batch_size, self.output_dim) * self.vth).to(self.device)
        # self.b_h = Variable(0.01 * torch.ones(batch_size, self.output_dim) * self.vth).to(self.device)

    def create_general_mask(self, dim=128, c_min=4, c_max=8, min_dc=0.1, max_dc=0.9, phase_shift_max=0.5, T=784*2):
        mask = []
        # Create ranges for cycles and duty cycles
        dc_steps = torch.linspace(min_dc, max_dc, steps=dim)
        cycles = torch.linspace(c_min, c_max, steps=dim)
        # Generate phase shifts within the specified maximum
        phase_shifts = torch.linspace(0, int(phase_shift_max * c_max), steps=dim)

        for cycle, dc, phase_shift in zip(cycles, dc_steps, phase_shifts):
            cycle = int(torch.ceil(cycle))
            on_length = int(torch.ceil(dc * cycle))
            off_length = cycle - on_length
            pattern = [1] * on_length + [0] * off_length

            phase_shift = int(torch.round(phase_shift))
            pattern = pattern[-phase_shift:] + pattern[:-phase_shift]  # Negative slicing for shifting right

            full_pattern = pattern * (T // cycle) + pattern[:T % cycle]  # Ensure the pattern fits exactly into T
            mask.append(full_pattern)

        mask = torch.tensor(mask, dtype=torch.float32)
        return mask.to(self.device)  # .to(device)

    def forward(self, input_spike, time):
        self.b_h = 0.3
        # concat the forward inputs with the recurrent inputs
        k_input = torch.cat((input_spike.float(), self.spike), 1)
        # synaptic inputs
        d_input = self.dense(k_input)

        mask = self.rhy_mask[:, time]

        # self.mem, self.spike = mem_update_pra_rhythm(d_input, self.mem, self.spike, self.v_th, self.tau_m, mask, self.dt,
        #                                       device=self.device)

        self.mem, self.spike, theta_h, self.b_h = mem_update_adp_rhythm(d_input, self.mem, self.spike, self.tau_adp_h, self.b_h,
                                                                   self.tau_m, mask, self.dt, device=self.device)

        return self.mem, self.spike




#vanilla SRNN with the noreset LIF neuron
class spike_rnn_test_origin_noreset(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 0.5,dt = 1,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_rnn_test_origin_noreset, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.is_adaptive = is_adaptive
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim+output_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)


    def set_neuron_state(self,batch_size):
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):

        #concat the the forward inputs with the recurrent inputs 
        k_input = torch.cat((input_spike.float(),self.spike),1)
        #synaptic inputs
        d_input = self.dense(k_input)
        self.mem,self.spike = mem_update_pra_noreset(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike





#DH-SRNN layer
class spike_rnn_test_denri_wotanh_R(nn.Module):
    def __init__(self,input_dim,output_dim,tau_minitializer = 'uniform',low_m = 0,high_m = 4,
                 tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth = 0.5,dt = 4,branch = 4,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            tau_ninitializer(str): the method of initialization of tau_n
            low_n(float): the low limit of the init values of tau_n
            high_n(float): the upper limit of the init values of tau_n
            vth(float): threshold
            branch(int): the number of dendritic branches
        """
        super(spike_rnn_test_denri_wotanh_R, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt
        mask_rate = 1/branch
        self.pad = ((input_dim+output_dim)//branch*branch+branch-(input_dim+output_dim)) % branch
        self.dense = nn.Linear(input_dim+output_dim+self.pad,output_dim*branch)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim,branch))
        #the number of dendritic branch
        self.branch = branch

        self.create_mask()
        
        # timing factor of membrane potential
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)
        # timing factor of dendritic branches       
        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n,low_n,high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n,low_n)

    
    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.tau_m,self.tau_n]
    
    #init
    def set_neuron_state(self,batch_size):

        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.d_input = Variable(torch.zeros(batch_size,self.output_dim,self.branch)).to(self.device)

        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)

    #create connection pattern
    def create_mask(self):
        input_size = self.input_dim+self.output_dim+self.pad
        self.mask = torch.zeros(self.output_dim*self.branch,input_size).to(self.device)
        for i in range(self.output_dim):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                self.mask[i*self.branch+j,seq[j*input_size // self.branch:(j+1)*input_size // self.branch]] = 1
    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data*self.mask
    def forward(self,input_spike):
        # timing factor of dendritic branches
        beta = torch.sigmoid(self.tau_n)
        padding = torch.zeros(input_spike.size(0),self.pad).to(self.device)
        k_input = torch.cat((input_spike.float(),self.spike,padding),1)
        #update dendritic currents 
        self.d_input = beta*self.d_input+(1-beta)*self.dense(k_input).reshape(-1,self.output_dim,self.branch)
        #summation of dendritic currents
        l_input = (self.d_input).sum(dim=2,keepdim=False)
        
        #update membrane potential and generate spikes
        self.mem,self.spike = mem_update_pra(l_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike


# Rhythm DH-SRNN layer
class Rhy_spike_rnn_test_denri_wotanh_R(nn.Module):
    def __init__(self, input_dim, output_dim, tau_minitializer='uniform', low_m=0, high_m=4,
                 tau_ninitializer='uniform', low_n=0, high_n=4, vth=0.5, dt=4, branch=4, cycle_min=3, cycle_max=10,
                 duty_cycle_min=0.02, duty_cycle_max=0.5, phase_max=0.5,
                 device='cpu', bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            tau_ninitializer(str): the method of initialization of tau_n
            low_n(float): the low limit of the init values of tau_n
            high_n(float): the upper limit of the init values of tau_n
            vth(float): threshold
            branch(int): the number of dendritic branches
        """
        super(Rhy_spike_rnn_test_denri_wotanh_R, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt
        mask_rate = 1 / branch
        self.pad = ((input_dim + output_dim) // branch * branch + branch - (input_dim + output_dim)) % branch
        self.dense = nn.Linear(input_dim + output_dim + self.pad, output_dim * branch)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim, branch))
        # the number of dendritic branch
        self.branch = branch

        self.create_mask()

        # self.rhy_mask = self.create_mix_mask(output_dim, skip_length_min, skip_length)
        self.rhy_mask = self.create_general_mask(output_dim, cycle_min, cycle_max, duty_cycle_min, duty_cycle_max,
                                                 phase_max)


        # timing factor of membrane potential
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)
        # timing factor of dendritic branches
        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n, low_n, high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n, low_n)

    def parameters(self):
        return [self.dense.weight, self.dense.bias, self.tau_m, self.tau_n]

    # init
    def set_neuron_state(self, batch_size):

        self.mem = Variable(torch.rand(batch_size, self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size, self.output_dim)).to(self.device)
        self.d_input = Variable(torch.zeros(batch_size, self.output_dim, self.branch)).to(self.device)

        self.v_th = Variable(torch.ones(batch_size, self.output_dim) * self.vth).to(self.device)

    # create connection pattern
    def create_mask(self):
        input_size = self.input_dim + self.output_dim + self.pad
        self.mask = torch.zeros(self.output_dim * self.branch, input_size).to(self.device)
        for i in range(self.output_dim):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                self.mask[
                    i * self.branch + j, seq[j * input_size // self.branch:(j + 1) * input_size // self.branch]] = 1

    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data * self.mask

    def create_general_mask(self, dim=128, c_min=4, c_max=8, min_dc=0.1, max_dc=0.9, phase_shift_max=0.5, T=784*2):
        mask = []
        # Create ranges for cycles and duty cycles
        dc_steps = torch.linspace(min_dc, max_dc, steps=dim)
        cycles = torch.linspace(c_min, c_max, steps=dim)
        # Generate phase shifts within the specified maximum
        phase_shifts = torch.linspace(0, int(phase_shift_max * c_max), steps=dim)

        for cycle, dc, phase_shift in zip(cycles, dc_steps, phase_shifts):
            cycle = int(torch.ceil(cycle))
            on_length = int(torch.ceil(dc * cycle))
            off_length = cycle - on_length
            pattern = [1] * on_length + [0] * off_length

            phase_shift = int(torch.round(phase_shift))
            pattern = pattern[-phase_shift:] + pattern[:-phase_shift]  # Negative slicing for shifting right

            full_pattern = pattern * (T // cycle) + pattern[:T % cycle]  # Ensure the pattern fits exactly into T
            mask.append(full_pattern)

        mask = torch.tensor(mask, dtype=torch.float32)
        return mask.to(self.device)  # .to(device)

    def forward(self, input_spike, time):
        # timing factor of dendritic branches
        beta = torch.sigmoid(self.tau_n)
        padding = torch.zeros(input_spike.size(0), self.pad).to(self.device)
        k_input = torch.cat((input_spike.float(), self.spike, padding), 1)
        # update dendritic currents
        self.d_input = beta * self.d_input + (1 - beta) * self.dense(k_input).reshape(-1, self.output_dim, self.branch)
        # summation of dendritic currents
        l_input = (self.d_input).sum(dim=2, keepdim=False)

        mask = self.rhy_mask[:, time]
        # update membrane potential and generate spikes
        self.mem, self.spike = mem_update_pra_rhythm(l_input, self.mem, self.spike, self.v_th, self.tau_m, mask, self.dt,
                                              device=self.device)

        return self.mem, self.spike
