import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from skip_FFSNN.Hyperparameters import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algo = args.algo
thresh = args.thresh
lens = args.lens
decay = args.decay

output_size = args.out_size
input_size = args.in_size
cfg_fc = args.fc
skip_length = args.skip_length

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
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

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

class FFSNN_grp(nn.Module):
    def __init__(self):
        super(FFSNN_grp, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # n_group [2, 4, 6] [256/2=128, 513/3=171, 256/4=64]
        self.sub_fc = [int(cfg_fc[0]/skip_length[0]),
                       int(cfg_fc[1]/skip_length[1]),
                       int(cfg_fc[2]/skip_length[2])]

        self.fc1_in1 = nn.Linear(input_size, self.sub_fc[0])
        self.fc1_in2 = nn.Linear(input_size, self.sub_fc[0])

        self.fc2_11 = nn.Linear(self.sub_fc[0], self.sub_fc[1])
        self.fc2_22 = nn.Linear(self.sub_fc[0], self.sub_fc[1])
        self.fc2_13 = nn.Linear(self.sub_fc[0], self.sub_fc[1])
        self.fc2_24 = nn.Linear(self.sub_fc[0], self.sub_fc[1])

        self.fc3_111 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_222 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_133 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_244 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_115 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_226 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_131 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_242 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_113 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_224 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_135 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_246 = nn.Linear(self.sub_fc[1], self.sub_fc[2])

        self.fc4_111o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_222o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_133o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_244o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_115o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_226o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_131o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_242o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_113o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_224o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_135o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_246o = nn.Linear(self.sub_fc[2], output_size)

    def forward(self, input):
        time_window = 784//input_size
        N = input.size(0)

        h1_in1_mem = h1_in1_spike = torch.zeros(N, self.sub_fc[0], device=device)
        h1_in2_mem = h1_in2_spike = torch.zeros(N, self.sub_fc[0], device=device)

        h2_11_mem = h2_11_spike = torch.zeros(N, self.sub_fc[1], device=device)
        h2_22_mem = h2_22_spike = torch.zeros(N, self.sub_fc[1], device=device)
        h2_13_mem = h2_13_spike = torch.zeros(N, self.sub_fc[1], device=device)
        h2_24_mem = h2_24_spike = torch.zeros(N, self.sub_fc[1], device=device)

        h3_111_mem = h3_111_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_222_mem = h3_222_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_133_mem = h3_133_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_244_mem = h3_244_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_115_mem = h3_115_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_226_mem = h3_226_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_131_mem = h3_131_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_242_mem = h3_242_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_113_mem = h3_113_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_224_mem = h3_224_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_135_mem = h3_135_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_246_mem = h3_246_spike = torch.zeros(N, self.sub_fc[2], device=device)

        output_sum = h4_mem = torch.zeros(N, output_size, device=device)

        input = np.squeeze(input)
        input = input.view(N, input_size, -1)
        for step in range(time_window):   # input [N, 28, T]
            input_x = input[:, :, step]

            if step%12 == 0:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_11_mem, h2_11_spike = mem_update(self.fc2_11, h1_in1_spike, h2_11_mem, h2_11_spike)
                h3_111_mem, h3_111_spike = mem_update(self.fc3_111, h2_11_spike, h3_111_mem, h3_111_spike)
                h4_mem = self.fc4_111o(h3_111_spike)
            elif step%12 == 1:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_22_mem, h2_22_spike = mem_update(self.fc2_22, h1_in2_spike, h2_22_mem, h2_22_spike)
                h3_222_mem, h3_222_spike = mem_update(self.fc3_222, h2_22_spike, h3_222_mem, h3_222_spike)
                h4_mem = self.fc4_222o(h3_222_spike)
            elif step%12 == 2:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_13_mem, h2_13_spike = mem_update(self.fc2_13, h1_in1_spike, h2_13_mem, h2_13_spike)
                h3_133_mem, h3_133_spike = mem_update(self.fc3_133, h2_13_spike, h3_133_mem, h3_133_spike)
                h4_mem = self.fc4_133o(h3_133_spike)
            elif step%12 == 3:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_24_mem, h2_24_spike = mem_update(self.fc2_24, h1_in2_spike, h2_24_mem, h2_24_spike)
                h3_244_mem, h3_244_spike = mem_update(self.fc3_244, h2_24_spike, h3_244_mem, h3_244_spike)
                h4_mem = self.fc4_244o(h3_244_spike)
            elif step%12 == 4:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_11_mem, h2_11_spike = mem_update(self.fc2_11, h1_in1_spike, h2_11_mem, h2_11_spike)
                h3_115_mem, h3_115_spike = mem_update(self.fc3_115, h2_11_spike, h3_115_mem, h3_115_spike)
                h4_mem = self.fc4_115o(h3_115_spike)
            elif step%12 == 5:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_22_mem, h2_22_spike = mem_update(self.fc2_22, h1_in2_spike, h2_22_mem, h2_22_spike)
                h3_226_mem, h3_226_spike = mem_update(self.fc3_226, h2_22_spike, h3_226_mem, h3_226_spike)
                h4_mem = self.fc4_226o(h3_226_spike)
            elif step%12 == 6:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_13_mem, h2_13_spike = mem_update(self.fc2_13, h1_in1_spike, h2_13_mem, h2_13_spike)
                h3_131_mem, h3_131_spike = mem_update(self.fc3_131, h2_13_spike, h3_131_mem, h3_131_spike)
                h4_mem = self.fc4_131o(h3_131_spike)
            elif step%12 == 7:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_24_mem, h2_24_spike = mem_update(self.fc2_24, h1_in2_spike, h2_24_mem, h2_24_spike)
                h3_242_mem, h3_242_spike = mem_update(self.fc3_242, h2_24_spike, h3_242_mem, h3_242_spike)
                h4_mem = self.fc4_242o(h3_242_spike)
            elif step%12 == 8:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_11_mem, h2_11_spike = mem_update(self.fc2_11, h1_in1_spike, h2_11_mem, h2_11_spike)
                h3_113_mem, h3_113_spike = mem_update(self.fc3_113, h2_11_spike, h3_113_mem, h3_113_spike)
                h4_mem = self.fc4_113o(h3_113_spike)
            elif step%12 == 9:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_22_mem, h2_22_spike = mem_update(self.fc2_22, h1_in2_spike, h2_22_mem, h2_22_spike)
                h3_224_mem, h3_224_spike = mem_update(self.fc3_224, h2_22_spike, h3_224_mem, h3_224_spike)
                h4_mem = self.fc4_226o(h3_224_spike)
            elif step%12 == 10:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_13_mem, h2_13_spike = mem_update(self.fc2_13, h1_in1_spike, h2_13_mem, h2_13_spike)
                h3_135_mem, h3_135_spike = mem_update(self.fc3_135, h2_13_spike, h3_135_mem, h3_135_spike)
                h4_mem = self.fc4_135o(h3_135_spike)
            elif step%12 == 11:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_24_mem, h2_24_spike = mem_update(self.fc2_24, h1_in2_spike, h2_24_mem, h2_24_spike)
                h3_246_mem, h3_246_spike = mem_update(self.fc3_246, h2_24_spike, h3_246_mem, h3_246_spike)
                h4_mem = self.fc4_246o(h3_246_spike)

            output_sum = output_sum + h4_mem # Accumulate mem of all time steps

        outputs = output_sum / time_window
        #outputs = h4_mem

        return outputs, None #n_nonzeros/n_neurons

class FFSNN_grp_234(nn.Module):
    def __init__(self):
        super(FFSNN_grp_234, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sub_fc = [int(cfg_fc[0]/skip_length[0]),
                       int(cfg_fc[1]/skip_length[1]),
                       int(cfg_fc[2]/skip_length[2])]

        self.fc1_in1 = nn.Linear(input_size, self.sub_fc[0])
        self.fc1_in2 = nn.Linear(input_size, self.sub_fc[0])

        self.fc2_11 = nn.Linear(self.sub_fc[0], self.sub_fc[1])
        self.fc2_22 = nn.Linear(self.sub_fc[0], self.sub_fc[1])
        self.fc2_13 = nn.Linear(self.sub_fc[0], self.sub_fc[1])
        self.fc2_21 = nn.Linear(self.sub_fc[0], self.sub_fc[1])
        self.fc2_12 = nn.Linear(self.sub_fc[0], self.sub_fc[1])
        self.fc2_23 = nn.Linear(self.sub_fc[0], self.sub_fc[1])

        self.fc3_111 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_222 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_133 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_214 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_121 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_232 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_113 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_224 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_131 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_212 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_123 = nn.Linear(self.sub_fc[1], self.sub_fc[2])
        self.fc3_234 = nn.Linear(self.sub_fc[1], self.sub_fc[2])

        self.fc4_111o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_222o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_133o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_214o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_121o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_232o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_113o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_224o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_131o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_212o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_123o = nn.Linear(self.sub_fc[2], output_size)
        self.fc4_234o = nn.Linear(self.sub_fc[2], output_size)

    def forward(self, input):
        time_window = 784//input_size
        N = input.size(0)

        h1_in1_mem = h1_in1_spike = torch.zeros(N, self.sub_fc[0], device=device)
        h1_in2_mem = h1_in2_spike = torch.zeros(N, self.sub_fc[0], device=device)

        h2_11_mem = h2_11_spike = torch.zeros(N, self.sub_fc[1], device=device)
        h2_22_mem = h2_22_spike = torch.zeros(N, self.sub_fc[1], device=device)
        h2_13_mem = h2_13_spike = torch.zeros(N, self.sub_fc[1], device=device)
        h2_21_mem = h2_21_spike = torch.zeros(N, self.sub_fc[1], device=device)
        h2_12_mem = h2_12_spike = torch.zeros(N, self.sub_fc[1], device=device)
        h2_23_mem = h2_23_spike = torch.zeros(N, self.sub_fc[1], device=device)

        h3_111_mem = h3_111_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_222_mem = h3_222_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_133_mem = h3_133_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_214_mem = h3_214_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_121_mem = h3_121_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_232_mem = h3_232_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_113_mem = h3_113_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_224_mem = h3_224_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_131_mem = h3_131_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_212_mem = h3_212_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_123_mem = h3_123_spike = torch.zeros(N, self.sub_fc[2], device=device)
        h3_234_mem = h3_234_spike = torch.zeros(N, self.sub_fc[2], device=device)

        output_sum = h4_mem = torch.zeros(N, output_size, device=device)

        input = np.squeeze(input)
        input = input.view(N, input_size, -1)
        for step in range(time_window):   # input [N, 28, T]
            input_x = input[:, :, step]

            if step%12 == 0:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_11_mem, h2_11_spike = mem_update(self.fc2_11, h1_in1_spike, h2_11_mem, h2_11_spike)
                h3_111_mem, h3_111_spike = mem_update(self.fc3_111, h2_11_spike, h3_111_mem, h3_111_spike)
                h4_mem = self.fc4_111o(h3_111_spike)
            elif step%12 == 1:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_22_mem, h2_22_spike = mem_update(self.fc2_22, h1_in2_spike, h2_22_mem, h2_22_spike)
                h3_222_mem, h3_222_spike = mem_update(self.fc3_222, h2_22_spike, h3_222_mem, h3_222_spike)
                h4_mem = self.fc4_222o(h3_222_spike)
            elif step%12 == 2:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_13_mem, h2_13_spike = mem_update(self.fc2_13, h1_in1_spike, h2_13_mem, h2_13_spike)
                h3_133_mem, h3_133_spike = mem_update(self.fc3_133, h2_13_spike, h3_133_mem, h3_133_spike)
                h4_mem = self.fc4_133o(h3_133_spike)
            elif step%12 == 3:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_21_mem, h2_21_spike = mem_update(self.fc2_21, h1_in2_spike, h2_21_mem, h2_21_spike)
                h3_214_mem, h3_214_spike = mem_update(self.fc3_214, h2_21_spike, h3_214_mem, h3_214_spike)
                h4_mem = self.fc4_214o(h3_214_spike)
            elif step%12 == 4:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_12_mem, h2_12_spike = mem_update(self.fc2_12, h1_in1_spike, h2_12_mem, h2_12_spike)
                h3_121_mem, h3_121_spike = mem_update(self.fc3_121, h2_12_spike, h3_121_mem, h3_121_spike)
                h4_mem = self.fc4_121o(h3_121_spike)
            elif step%12 == 5:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_23_mem, h2_23_spike = mem_update(self.fc2_23, h1_in2_spike, h2_23_mem, h2_23_spike)
                h3_232_mem, h3_232_spike = mem_update(self.fc3_232, h2_23_spike, h3_232_mem, h3_232_spike)
                h4_mem = self.fc4_232o(h3_232_spike)
            elif step%12 == 6:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_11_mem, h2_11_spike = mem_update(self.fc2_11, h1_in1_spike, h2_11_mem, h2_11_spike)
                h3_113_mem, h3_113_spike = mem_update(self.fc3_113, h2_11_spike, h3_113_mem, h3_113_spike)
                h4_mem = self.fc4_113o(h3_113_spike)
            elif step%12 == 7:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_22_mem, h2_22_spike = mem_update(self.fc2_22, h1_in2_spike, h2_22_mem, h2_22_spike)
                h3_224_mem, h3_224_spike = mem_update(self.fc3_224, h2_22_spike, h3_224_mem, h3_224_spike)
                h4_mem = self.fc4_224o(h3_224_spike)
            elif step%12 == 8:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_13_mem, h2_13_spike = mem_update(self.fc2_13, h1_in1_spike, h2_13_mem, h2_13_spike)
                h3_131_mem, h3_131_spike = mem_update(self.fc3_131, h2_13_spike, h3_131_mem, h3_131_spike)
                h4_mem = self.fc4_131o(h3_131_spike)
            elif step%12 == 9:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_21_mem, h2_21_spike = mem_update(self.fc2_21, h1_in2_spike, h2_21_mem, h2_21_spike)
                h3_212_mem, h3_212_spike = mem_update(self.fc3_212, h2_21_spike, h3_212_mem, h3_212_spike)
                h4_mem = self.fc4_212o(h3_212_spike)
            elif step%12 == 10:
                h1_in1_mem, h1_in1_spike = mem_update(self.fc1_in1, input_x, h1_in1_mem, h1_in1_spike)
                h2_12_mem, h2_12_spike = mem_update(self.fc2_12, h1_in1_spike, h2_12_mem, h2_12_spike)
                h3_123_mem, h3_123_spike = mem_update(self.fc3_123, h2_12_spike, h3_123_mem, h3_123_spike)
                h4_mem = self.fc4_123o(h3_123_spike)
            elif step%12 == 11:
                h1_in2_mem, h1_in2_spike = mem_update(self.fc1_in2, input_x, h1_in2_mem, h1_in2_spike)
                h2_23_mem, h2_23_spike = mem_update(self.fc2_23, h1_in2_spike, h2_23_mem, h2_23_spike)
                h3_234_mem, h3_234_spike = mem_update(self.fc3_234, h2_23_spike, h3_234_mem, h3_234_spike)
                h4_mem = self.fc4_234o(h3_234_spike)

            output_sum = output_sum + h4_mem # Accumulate mem of all time steps

        outputs = output_sum / time_window
        #outputs = h4_mem

        return outputs, None #n_nonzeros/n_neurons

