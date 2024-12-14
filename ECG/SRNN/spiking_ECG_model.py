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


#################################################################
class SRNN_ALIF(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    """
    def __init__(self, in_size, hidden_size, output_size):
        super(SRNN_ALIF, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #self.sub_seq_length = sub_seq_length

        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

        self.tau_adp_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))
        
        self.tau_m_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tau_m_o = nn.Parameter(torch.Tensor(self.output_size))

        self._initial_parameters()
        self.b_h = self.b_o = 0


    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2o.weight)
        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.h2o.bias, 0)

        nn.init.constant_(self.tau_adp_h, 7) #7
        nn.init.constant_(self.tau_adp_o, 100)
        nn.init.constant_(self.tau_m_h, 20) #7
        nn.init.constant_(self.tau_m_o, 20)

    def forward(self, input):
        self.b_h = self.b_o = b_j0
        batch_size, seq_num, input_dim = input.shape
        hidden_mem = hidden_spike = (torch.rand(batch_size, self.hidden_size)*b_j0).cuda()
        output_mem = output_spike = out_spike = (torch.rand(batch_size, self.output_size)*b_j0).cuda()

        outputs = []
        for step in range(seq_num):
            input_x = input[:, step, :]

            #################   update states  #########################
            h_input = self.i2h(input_x.float()) + self.h2h(hidden_spike)
            hidden_mem, hidden_spike, theta_h, self.b_h = mem_update_adp(h_input,hidden_mem, hidden_spike, self.tau_adp_h,
                                                                         self.tau_m_h, self.b_h,isAdapt=0)

            o_input = self.h2o(hidden_spike)
            output_mem, output_spike, theta_o, self.b_o = mem_update_adp(o_input,output_mem,output_spike, self.tau_adp_o, 
                                                                         self.tau_m_o, self.b_o,isAdapt=1)
            
            #################   classification  #########################
            #if step >= self.sub_seq_length:
            output_sumspike = output_mem 
            output_sumspike = F.log_softmax(output_sumspike,dim=1) # [N, 6]
            outputs.append(output_sumspike)
        outputs = torch.stack(outputs).permute(1,2,0) #[N, 6, T]

        return outputs
    
    def predict(self, input):
        self.b_h = self.b_o = b_j0
        batch_size, seq_num, input_dim = input.shape
        hidden_mem = hidden_spike = (torch.rand(batch_size, self.hidden_size)*b_j0).cuda()
        output_mem = output_spike = out_spike = (torch.rand(batch_size, self.output_size)*b_j0).cuda()

        predictions = []
        for step in range(seq_num):
            input_x = input[:, step, :]

            #################   update states  #########################
            h_input = self.i2h(input_x.float()) + self.h2h(hidden_spike)
            hidden_mem, hidden_spike, theta_h, self.b_h = mem_update_adp(h_input,hidden_mem, hidden_spike, self.tau_adp_h,
                                                                         self.tau_m_h, self.b_h,isAdapt=0)

            o_input = self.h2o(hidden_spike)
            output_mem, output_spike, theta_o, self.b_o = mem_update_adp(o_input,output_mem,output_spike, self.tau_adp_o, 
                                                                         self.tau_m_o, self.b_o,isAdapt=1)
            
            #################   classification  #########################
            #if step >= self.sub_seq_length:
            output_sumspike = output_mem 
            output_sumspike = F.log_softmax(output_sumspike,dim=1) # [N, 6]
            predictions.append(output_sumspike.data.cpu().numpy())
        
        predictions = torch.tensor(predictions)

        return predictions
    

class SRNN(nn.Module):
    def __init__(self, in_size, hidden_size, output_size):
        super(SRNN, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        batch_size, seq_num, input_dim = input.shape
        h2h_mem = h2h_spike = torch.zeros(batch_size, self.hidden_size, device=device)
        h2o_mem = h2o_spike = output_sum = torch.zeros(batch_size, self.output_size, device=device)

        outputs = []
        for step in range(seq_num):
            input_x = input[:, step, :]

            h_input = self.i2h(input_x.float()) + self.h2h(h2h_spike)
            h2h_mem, h2h_spike = mem_update_hidden(h_input, h2h_mem, h2h_spike)
            h2o_mem, h2o_spike = mem_update(self.h2o, h2h_spike, h2o_mem, h2o_spike)
            mem_sum = h2o_mem 
            mem_sum = F.log_softmax(mem_sum,dim=1) # [N, 6]
            outputs.append(mem_sum)
        outputs = torch.stack(outputs).permute(1,2,0) #[N, 6, T]

        return outputs
