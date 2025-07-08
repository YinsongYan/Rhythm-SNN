import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.3
lens = 0.2

grads = []
def save_grad(name,step):
    def hook(grad):
        grads.append(grad)
    return hook

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,thresh_t):  # input = membrane potential- threshold
        ctx.save_for_backward(input,thresh_t)
        return input.gt(thresh_t).float()  # is firing ???
    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input,thresh_t = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input-thresh_t) < lens
        return grad_input * temp.float(), -grad_input * temp.float()

act_fun_adp = ActFun_adp.apply



class TESNN(nn.Module):

    def __init__(self, in_size=1, out_size=1, time_window=1, te_type=None, hidden_size=[64, 64], beta = 0.02, decay = 0.5):
        super(TESNN, self).__init__()
        self.input_size = in_size
        self.output_size = out_size
        self.T = time_window
        self.hidden_size = hidden_size
        self.i2h_1 = nn.Linear(self.input_size, self.hidden_size[0])
        self.i2h_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h2h_1 = nn.Linear(self.hidden_size[0], self.hidden_size[0])
        self.h2o_3 = nn.Linear(self.hidden_size[1], self.output_size)

        self.co2 = beta
        self.decay = decay

        self.te = te_type

        if self.te == 'TE-R':
            dt = 0.02 / (2e1 ** (torch.arange(self.hidden_size[-1]).float().to(device) / self.hidden_size[-1]))
            #dt = 0.1 / (1e2 ** (torch.arange(self.hidden_size[-1]).float().to(device) / self.hidden_size[-1]))
            self.dt = dt.to(device)
            self.fre = nn.Parameter((torch.rand(self.hidden_size[-1], device=device)+0.99) * math.pi * 2)
        elif self.te == 'TE-N':
            print('learn')
            self.ce = nn.Parameter(torch.zeros(self.hidden_size[-1], self.T, device=device))
            nn.init.normal_(self.ce, 0.01, 0.01)
        else:
            print('lif')
            self.ce = torch.zeros(self.hidden_size[-1], self.T, device=device)
        self.time_idx = torch.arange(self.T).unsqueeze(0).to(device)

    def mem_update_hidden_wv(self, x, mem, spike, thresh_t, co1):
        if self.te != 'LIF':
            thresh_t = thresh_t + mem * co1 - (thresh_t - thresh) * self.co2

        mem = mem * self.decay * (1. - spike) + x
        spike = act_fun_adp(mem, thresh_t)
        return mem, spike, thresh_t


    def forward(self, input, task='duration'):
        if self.te == 'TE-R':
            fre_complex = (1j * (self.fre * self.dt)).unsqueeze(1)
            self.ce = torch.exp(fre_complex * self.time_idx).real * 0.1

        out = []
        N = input.size(0)
        h2h1_mem = h2h1_spike = torch.zeros(N, self.hidden_size[0], device=device)

        h2h2_mem = h2h2_spike = torch.zeros(N, self.hidden_size[1], device=device)


        thresh1 = torch.ones(N, self.hidden_size[0], device=device) * thresh
        thresh2 = torch.ones(N, self.hidden_size[1], device=device) * thresh

        for step in range(self.T):
            input_x=input[:,:,step]

            h1_input = self.i2h_1(input_x.float()) + self.h2h_1(h2h1_spike)

            h2h1_mem, h2h1_spike, thresh1 = self.mem_update_hidden_wv(h1_input, h2h1_mem, h2h1_spike,
                                                                        thresh1, self.ce[:self.hidden_size[0], step])

            h2_input = self.i2h_2((h2h1_spike))
            h2h2_mem, h2h2_spike, thresh2 = self.mem_update_hidden_wv(h2_input, h2h2_mem, h2h2_spike,
                                                                        thresh2, self.ce[:self.hidden_size[-1], step])

            h2o3_mem = self.h2o_3(h2h2_spike)
            out.append(h2o3_mem)

        output = torch.stack(out, dim=2)
        if task == 'duration':
            return output[:,:,-1]  # Duration
        elif task == 'syn':
            return output # Synchronization
        elif task == 'interval':
            return output[:, :, -1]  # Interval
        elif task == 'recall':
            return output # Delayed recall

    def visualize(self, input, task='duration'):
        if self.te == 'TE-R':
            fre_complex = (1j * (self.fre * self.dt)).unsqueeze(1)
            self.ce = torch.exp(fre_complex * self.time_idx).real * 0.1
        out = []
        N = input.size(0)
        h2h1_mem = h2h1_spike = torch.zeros(N, self.hidden_size[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(N, self.hidden_size[1], device=device)


        thresh1 = torch.ones(N, self.hidden_size[0], device=device) * thresh
        thresh2 = torch.ones(N, self.hidden_size[1], device=device) * thresh
        hid1 = []
        hid2 = []
        for step in range(self.T):
            input_x=input[:,:,step]


            h1_input = self.i2h_1(input_x.float())+ self.h2h_1(h2h1_spike)

            h2h1_mem, h2h1_spike, thresh1 = self.mem_update_hidden_wv(h1_input, h2h1_mem, h2h1_spike,
                                                                        thresh1, self.ce[:self.hidden_size[0], step])

            h2_input = self.i2h_2((h2h1_spike))
            h2h2_mem, h2h2_spike, thresh2 = self.mem_update_hidden_wv(h2_input, h2h2_mem, h2h2_spike,
                                                                        thresh2, self.ce[:self.hidden_size[-1], step])

            h2o3_mem = self.h2o_3(h2h2_spike)
            out.append(h2o3_mem) # set2

            hid1.append(h2h1_spike)
            hid2.append(h2h2_spike)

        hidden_spike1=torch.stack(hid1,dim=2)
        hidden_spike2=torch.stack(hid2,dim=2)
        output = torch.stack(out, dim=2)

        if task == 'duration':
            return output, hidden_spike1, hidden_spike2  # Duration
        elif task == 'syn':
            return output, hidden_spike1, hidden_spike2 # Synchronization
        elif task == 'interval':
            return output.clamp(0,2), hidden_spike1, hidden_spike2  # Interval
        elif task == 'recall':
            return output, hidden_spike1, hidden_spike2 # Delayed recall

    def get_grads(self, input):
        if self.te == 'TE-R':
            fre_complex = (1j * (self.fre * self.dt)).unsqueeze(1)
            self.ce = torch.exp(fre_complex * self.time_idx).real * 0.1

        out = []
        N = input.size(0)
        h2h1_mem = h2h1_spike = torch.zeros(N, self.hidden_size[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(N, self.hidden_size[1], device=device)


        thresh1 = torch.ones(N, self.hidden_size[0], device=device) * thresh
        thresh2 = torch.ones(N, self.hidden_size[1], device=device) * thresh

        for step in range(self.T):
            input_x=input[:,:,step]


            h1_input = self.i2h_1(input_x.float())+ self.h2h_1(h2h1_spike)

            h2h1_mem, h2h1_spike, thresh1 = self.mem_update_hidden_wv(h1_input, h2h1_mem, h2h1_spike,
                                                                        thresh1, self.ce[:self.hidden_size[0], step])

            h2_input = self.i2h_2((h2h1_spike))
            h2h2_mem, h2h2_spike, thresh2 = self.mem_update_hidden_wv(h2_input, h2h2_mem, h2h2_spike,
                                                                        thresh2, self.ce[:self.hidden_size[-1], step])

            h2o3_mem = self.h2o_3(h2h2_spike)
            h2h2_spike.register_hook(save_grad('grads', step))
            out.append(h2o3_mem)
        output = torch.stack(out, dim=2)

        return output, grads

