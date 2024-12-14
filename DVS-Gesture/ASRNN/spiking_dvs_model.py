import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ASRNN.Hyperparameters import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len =  args.seq_len
input_size = args.in_size
output_size = args.out_size
cfg_fc = args.fc


#b_j0 = 0.01  # neural threshold baseline
b_j0 = 0.1 
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
#lens = 0.5
lens = args.lens

###########################################################################################
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

def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
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

###########################################################################################


thresh = args.thresh
decay = args.decay

###########################################################################################
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

def mem_update(ops, x, mem, spike): # ops weight shape [32, 1, 3, 3], x [250, 1, 28, 28], mem [250, 32, 28, 28], spike [250, 32, 28, 28]
    mem = mem * decay * (1. - spike) + ops(x)   # mem: AddBackward, spike: ActFunBackward
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike
    
def mem_update_hidden(x, mem, spike):
    mem = mem * decay * (1. - spike) + x
    spike = act_fun(mem)
    return mem, spike 


def output_Neuron(inputs, mem, tau_m, dt=1):
    """
    The read out neuron is leaky integrator without spike
    """
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem

#################################################################
class SRNN_ALIF(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    """
    def __init__(self,):
        super(SRNN_ALIF, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.T = seq_len

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        #self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(self.input_size, cfg_fc[1])
        #self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[0] + cfg_fc[1], cfg_fc[2])
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

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)

        self._initial_parameters()
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def _initial_parameters(self):
        #nn.init.orthogonal_(self.h2h_1.weight)
        #nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        #nn.init.constant_(self.h2h_1.bias, 0)
        #nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)

        nn.init.normal_(self.tau_adp_h1, 200, 50) # 700, 25
        nn.init.normal_(self.tau_adp_h2, 200, 50)
        nn.init.normal_(self.tau_adp_h3, 200, 50)
        nn.init.normal_(self.tau_adp_o, 200, 50)

        nn.init.normal_(self.tau_m_h1, 20., 5)  # 20, 5
        nn.init.normal_(self.tau_m_h2, 20., 5)
        nn.init.normal_(self.tau_m_h3, 20., 5)
        nn.init.normal_(self.tau_m_o, 20., 5)

    def forward(self, input):
        N = input.size(0) # [N, T, 2, 128, 128]

        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        h2h1_mem = h2h1_spike = torch.zeros(N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, self.output_size, device=device)

        for step in range(self.T):
            input_x = input[:,step,:,:,:]
            x_down = F.max_pool2d(input_x[ :,:,:,: ],kernel_size=4,stride=4) # [N, 2, 32, 32]

            h1_input = self.i2h_1(x_down[:,0,:,:].view(N,self.input_size)) # + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp(h1_input, h2h1_mem, h2h1_spike,
                                                                       self.tau_adp_h1, self.b_h1, self.tau_m_h1)

            h2_input = self.i2h_2(x_down[:,1,:,:].view(N,self.input_size)) # + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp(h2_input, h2h2_mem, h2h2_spike,
                                                                       self.tau_adp_h2, self.b_h2, self.tau_m_h2)

            h2h1_spike = self.dp1(h2h1_spike)
            h2h2_spike = self.dp2(h2h2_spike)
            
            h3_input = torch.cat((h2h1_spike, h2h2_spike),dim=-1)
            h3_input = self.i2h_3(h3_input) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp(h3_input, h2h3_mem, h2h3_spike,
                                                                       self.tau_adp_h3, self.b_h3, self.tau_m_h3)
            h2h3_spike = self.dp3(h2h3_spike)
            
            h2o3_mem = self.h2o_3(h2h3_spike)
            #mem_output = output_Neuron(self.h2o(spike_layer2), mem_output, self.tau_m_o)

        #     output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
        # outputs = output_sum / self.T
        outputs = h2o3_mem
        return outputs


class SRNN_ALIF_outNeu(nn.Module):
    """
    Using Output Neuron 
    """
    def __init__(self,):
        super(SRNN_ALIF_outNeu, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.T = seq_len

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        #self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(self.input_size, cfg_fc[1])
        #self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[0] + cfg_fc[1], cfg_fc[2])
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

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)

        self._initial_parameters()
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def _initial_parameters(self):
        #nn.init.orthogonal_(self.h2h_1.weight)
        #nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        #nn.init.constant_(self.h2h_1.bias, 0)
        #nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)

        nn.init.normal_(self.tau_adp_h1, 200, 50) # 700, 25
        nn.init.normal_(self.tau_adp_h2, 200, 50)
        nn.init.normal_(self.tau_adp_h3, 200, 50)
        nn.init.normal_(self.tau_adp_o, 200, 50)

        nn.init.normal_(self.tau_m_h1, 20., 5)  # 20, 5
        nn.init.normal_(self.tau_m_h2, 20., 5)
        nn.init.normal_(self.tau_m_h3, 20., 5)
        nn.init.normal_(self.tau_m_o, 20., 5)

    def forward(self, input):
        N = input.size(0) # [N, T, 2, 128, 128]

        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        h2h1_mem = h2h1_spike = torch.zeros(N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(N, cfg_fc[2], device=device)
        mem_output = output_sum = torch.zeros(N, self.output_size, device=device)

        for step in range(self.T):
            input_x = input[:,step,:,:,:]
            x_down = F.max_pool2d(input_x[ :,:,:,: ],kernel_size=4,stride=4) # [N, 2, 32, 32]

            h1_input = self.i2h_1(x_down[:,0,:,:].view(N,self.input_size)) # + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp(h1_input, h2h1_mem, h2h1_spike,
                                                                       self.tau_adp_h1, self.b_h1, self.tau_m_h1)

            h2_input = self.i2h_2(x_down[:,1,:,:].view(N,self.input_size)) # + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp(h2_input, h2h2_mem, h2h2_spike,
                                                                       self.tau_adp_h2, self.b_h2, self.tau_m_h2)

            h2h1_spike = self.dp1(h2h1_spike)
            h2h2_spike = self.dp2(h2h2_spike)
            
            h3_input = torch.cat((h2h1_spike, h2h2_spike),dim=-1)
            h3_input = self.i2h_3(h3_input) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp(h3_input, h2h3_mem, h2h3_spike,
                                                                       self.tau_adp_h3, self.b_h3, self.tau_m_h3)
            h2h3_spike = self.dp3(h2h3_spike)
            
            h2o3_mem = self.h2o_3(h2h3_spike)
            mem_output = output_Neuron(h2o3_mem, mem_output, self.tau_m_o)

            output_sum = output_sum + mem_output  # Using output layer's mem potential to make decision.
        outputs = output_sum / self.T

        return outputs


class SRNN_ALIF_1Adapt(nn.Module):
    """
    Only last layer is adaptive
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    """
    def __init__(self,):
        super(SRNN_ALIF_1Adapt, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.T = seq_len

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])

        self.i2h_2 = nn.Linear(self.input_size, cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[0] + cfg_fc[1], cfg_fc[2])
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

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)

        self._initial_parameters()
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def _initial_parameters(self):
        #nn.init.orthogonal_(self.h2h_1.weight)
        #nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)

        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        #nn.init.constant_(self.h2h_1.bias, 0)
        #nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)

        nn.init.normal_(self.tau_adp_h1, 200, 50) # 700, 25
        nn.init.normal_(self.tau_adp_h2, 200, 50)
        nn.init.normal_(self.tau_adp_h3, 200, 50)
        nn.init.normal_(self.tau_adp_o, 200, 50)

        nn.init.normal_(self.tau_m_h1, 20., 5)  # 20, 5
        nn.init.normal_(self.tau_m_h2, 20., 5)
        nn.init.normal_(self.tau_m_h3, 20., 5)
        nn.init.normal_(self.tau_m_o, 20., 5)

    def forward(self, input):
        N = input.size(0) # [N, T, 2, 128, 128]

        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        h2h1_mem = h2h1_spike = torch.zeros(N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(N, cfg_fc[2], device=device)
        output_sum = torch.zeros(N, self.output_size, device=device)

        for step in range(self.T):
            input_x = input[:,step,:,:,:]
            x_down = F.max_pool2d(input_x[ :,:,:,: ],kernel_size=4,stride=4) # [N, 2, 32, 32]

            h1_input = self.i2h_1(x_down[:,0,:,:].view(N,self.input_size)) # + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_adp(h1_input, h2h1_mem, h2h1_spike,
                                                                       self.tau_adp_h1, self.b_h1, self.tau_m_h1, isAdapt=0)

            h2_input = self.i2h_2(x_down[:,1,:,:].view(N,self.input_size)) # + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_adp(h2_input, h2h2_mem, h2h2_spike,
                                                                       self.tau_adp_h2, self.b_h2, self.tau_m_h2, isAdapt=0)

            h2h1_spike = self.dp1(h2h1_spike)
            h2h2_spike = self.dp2(h2h2_spike)
            
            h3_input = torch.cat((h2h1_spike, h2h2_spike),dim=-1)
            h3_input = self.i2h_3(h3_input) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_adp(h3_input, h2h3_mem, h2h3_spike,
                                                                       self.tau_adp_h3, self.b_h3, self.tau_m_h3)
            h2h3_spike = self.dp3(h2h3_spike)
            
            h2o3_mem = self.h2o_3(h2h3_spike)
            #mem_output = output_Neuron(self.h2o(spike_layer2), mem_output, self.tau_m_o)

        #     output_sum = output_sum + h2o3_mem  # Using output layer's mem potential to make decision.
        # outputs = output_sum / self.T
        outputs = h2o3_mem
        return outputs


#################################################################
class SRNN(nn.Module):
    def __init__(self):
        super(SRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.T = seq_len
        
        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        #self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(self.input_size, cfg_fc[1])
        #self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[0] + cfg_fc[1], cfg_fc[2])
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])
        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)

    def forward(self, input):
        N = input.size(0)
        h2h1_mem = h2h1_spike = torch.zeros(N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(N, cfg_fc[2], device=device)
        h2o3_mem = output_sum = torch.zeros(N, self.output_size, device=device)

        for step in range(self.T):
            input_x = input[:,step,:,:,:]
            x_down = F.max_pool2d(input_x[ :,:,:,: ],kernel_size=4,stride=4)

            h1_input = self.i2h_1(x_down[:,0,:,:].view(N,self.input_size)) #+ self.h2h_1(h2h1_spike)  # h_input [200,128] : [64-->128] + [128-->128]
            h2h1_mem, h2h1_spike = mem_update_hidden(h1_input, h2h1_mem, h2h1_spike)
            
            h2_input = self.i2h_2(x_down[:,1,:,:].view(N,self.input_size)) #+ self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike = mem_update_hidden(h2_input, h2h2_mem, h2h2_spike)
            
            h2h1_spike = self.dp1(h2h1_spike)
            h2h2_spike = self.dp2(h2h2_spike)

            h3_input = torch.cat((h2h1_spike, h2h2_spike),dim=-1)
            h3_input = self.i2h_3(h3_input) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike = mem_update_hidden(h3_input, h2h3_mem, h2h3_spike)
    
            h2h3_spike = self.dp3(h2h3_spike)
            h2o3_mem = self.h2o_3(h2h3_spike)

        #     output_sum = output_sum + h2o3_mem  
        # outputs = output_sum / self.T
        outputs = h2o3_mem
        return outputs

