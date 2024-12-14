import torch
import torch.nn as nn
import torch.nn.functional as F
from LSNN.Hyperparameters import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
algo = args.algo
thresh = args.thresh # 0.3
lens = args.lens #0.3
decay = args.decay # 0.3

output_size = args.out_size
input_size = args.in_size
cfg_fc = args.fc

class ActFun(torch.autograd.Function):
    # Define approximate firing function
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        # function
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

# membrane potential update
def mem_combine(x1, x2, mem, spike):
    mem = mem * decay * (1. - spike) + x1 + x2
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike

def mem_update(opts, x, mem, spike):
    if algo == "STBP" :
        mem = mem * decay * (1. - spike) + opts(x) # STBP
    else:
        mem = mem.detach() * decay * (1. - spike.detach()) + opts(x) # SDBP
    spike = act_fun(mem)
    return mem, spike

def mem_update_hidden(x, mem, spike):
    if algo == "STBP":
        mem = mem * decay * (1. - spike) + x # STBP
    else:
        mem = mem.detach() * decay * (1. - spike.detach()) + x
    spike = act_fun(mem)
    return mem, spike

act_fun = ActFun.apply


'''
STEP 3a_v2: CREATE Adaptative spike MODEL CLASS
'''
b_j0 = 0.01  # neural threshold baseline
tau_m = 5  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale


# define approximate firing function
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
        return gamma * grad_input * temp.float()


act_fun_adp = ActFun_adp.apply
# membrane potential update

tau_m = torch.FloatTensor([tau_m])


def mem_update_adp(ops, x, mem, spike, tau_adp, b, dt=1, isAdapt=1):
    alpha = torch.exp(-1. * dt / tau_m).cuda()
    ro = torch.exp(-1. * dt / tau_adp).cuda()
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    if algo == 'LSNN':
        mem = mem * alpha + (1 - alpha) * R_m * ops(x) - B * spike * dt
    else:
        mem = mem.detach() * alpha + (1 - alpha) * R_m * ops(x) - B * spike.detach() * dt  # Block the Temporal BP

    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b


def mem_update_NU_adp(inputs, mem, spike, tau_adp, b, dt=1, isAdapt=1):
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

    if algo == 'LSNN':
        mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    else:
        mem = mem.detach() * alpha + (1 - alpha) * R_m * inputs - B * spike.detach() * dt  # Block the Temporal BP

    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b


class LSNN(nn.Module):

    def __init__(self):
        super(LSNN, self).__init__()

        self.input_size = input_size  # num_encode*input_dim
        self.hidden_size = 800
        self.output_size = output_size
        #self.encode_size = input_size
        self.decode_step = 50

        self.i2h = nn.Linear(self.input_size, self.hidden_size)  # [1,300]
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)  # [300,300]
        self.h2o = nn.Linear(self.hidden_size, self.output_size)  # [300, 10]

        self.tau_adp_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))

        nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2o.weight)
        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.tau_adp_h, 700)
        nn.init.constant_(self.tau_adp_o, 700)
        self.b_h = self.b_o = 0

    def forward(self, input ):
        self.b_h = self.b_o = b_j0
        self.N = input.size(0)
        self.T = 784 // self.input_size
        h2h_mem = h2h_spike = torch.zeros(self.N, self.hidden_size, device=device)
        h2o_mem = h2o_spike = output_sumspike = torch.zeros(self.N, self.output_size, device=device)

        decode_input = torch.zeros(self.N, self.input_size, self.decode_step, device=device)

        input = input.view(self.N, self.input_size, -1)

        input = torch.cat((input, decode_input), dim=2) #[N, T, F]

        for step in range(self.T + self.decode_step):  # seq_num = 784 (pixel by pixel)

            input_x = input[:, :, step]
            ####################################################################
            h_input = self.i2h(input_x) + self.h2h(h2h_spike)  # h_input [N,128] : [80-->128] + [128-->128]
            h2h_mem, h2h_spike, theta_h, self.b_h = mem_update_NU_adp(h_input,
                                                                     h2h_mem, h2h_spike, self.tau_adp_h, self.b_h)

            h2o_mem, h2o_spike, theta_o, self.b_o = mem_update_adp(self.h2o, h2h_spike,
                                                                   h2o_mem, h2o_spike, self.tau_adp_o, self.b_o)

            if step >= self.T:
                output_sumspike = output_sumspike + h2o_spike

            #output_sumspike = output_sumspike + h2o_spike

        return output_sumspike

class LSNN_multi(nn.Module):

    def __init__(self):
        super(LSNN_multi, self).__init__()

        self.input_size = input_size  # num_encode*input_dim
        #self.hidden_size = 300
        self.output_size = output_size
        #self.encode_size = input_size
        self.decode_step = 50

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])  # [1,256]
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])  # [256,256]

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])  # [256,512]
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])  # [512,512]

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])  # [512,256]
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])  # [256,256]
        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)  # [256, 10]

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))

        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))

        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)
        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)
        nn.init.constant_(self.tau_adp_h1, 700)
        nn.init.constant_(self.tau_adp_h2, 700)
        nn.init.constant_(self.tau_adp_h3, 700)
        nn.init.constant_(self.tau_adp_o, 700)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def forward(self, input ):
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        self.N = input.size(0)
        self.T = 784 // self.input_size


        h2h1_mem = h2h1_spike = torch.zeros(self.N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(self.N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(self.N, cfg_fc[2], device=device)
        h2o3_mem = h2o3_spike = output_sumspike = torch.zeros(self.N, self.output_size, device=device)

        input = input.view(self.N, self.input_size, -1)

        for step in range(self.T ):

            input_x = input[:, :, step]
            ####################################################################
            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_NU_adp(h1_input,
                                                                          h2h1_mem, h2h1_spike,
                                                                          self.tau_adp_h1, self.b_h1)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_NU_adp(h2_input,
                                                                          h2h2_mem, h2h2_spike,
                                                                          self.tau_adp_h2, self.b_h2)

            h3_input = self.i2h_3(h2h2_spike) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_NU_adp(h3_input,
                                                                          h2h3_mem, h2h3_spike,
                                                                          self.tau_adp_h3, self.b_h3)

            #h2o3_mem, h2o3_spike, theta_o, self.b_o = mem_update_adp(self.h2o_3, h2h3_spike,
            #                                                       h2o3_mem, h2o3_spike, self.tau_adp_o, self.b_o)

            h2o3_mem = self.h2o_3(h2h3_spike)

            output_sumspike = output_sumspike + h2o3_mem

        return output_sumspike

class LSNN_multi_2RNN(nn.Module):
    """
    Similar method with paper "Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks"
    """
    def __init__(self, in_size=8):
        super(LSNN_multi_2RNN, self).__init__()
        self.input_size = in_size
        self.stride = input_size
        self.output_size = output_size
        self.T = 784 // self.stride

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])
        #self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])
        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))
        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))

        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        #nn.init.xavier_uniform_(self.h2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        #nn.init.orthogonal_(self.h2h_3.weight)
        #nn.init.xavier_uniform_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)
        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        #nn.init.constant_(self.h2h_3.bias, 0)
        nn.init.constant_(self.tau_adp_h1, 700)
        nn.init.constant_(self.tau_adp_h2, 700)
        nn.init.constant_(self.tau_adp_h3, 700)
        nn.init.constant_(self.tau_adp_o, 700)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def forward(self, input ):
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        self.N = input.size(0)

        #'''
        h2h1_mem = h2h1_spike = torch.zeros(self.N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(self.N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(self.N, cfg_fc[2], device=device)
        '''
        h2h1_mem = torch.rand(self.N, cfg_fc[0], device=device)
        h2h2_mem = torch.rand(self.N, cfg_fc[1], device=device)
        h2h3_mem = torch.rand(self.N, cfg_fc[2], device=device)
        h2h1_spike = torch.zeros(self.N, cfg_fc[0], device=device)
        h2h2_spike = torch.zeros(self.N, cfg_fc[1], device=device)
        h2h3_spike = torch.zeros(self.N, cfg_fc[2], device=device)
        '''

        output_sumspike = torch.zeros(self.N, self.output_size, device=device)

        input = input.view(self.N, -1)

        for step in range(self.T):
            start_idx = step * self.stride
            if start_idx < (self.T - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            ####################################################################
            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_NU_adp(h1_input,
                                                                          h2h1_mem, h2h1_spike,
                                                                          self.tau_adp_h1, self.b_h1)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_NU_adp(h2_input,
                                                                          h2h2_mem, h2h2_spike,
                                                                          self.tau_adp_h2, self.b_h2)

            h3_input = self.i2h_3(h2h2_spike) #+ self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_NU_adp(h3_input,
                                                                          h2h3_mem, h2h3_spike,
                                                                          self.tau_adp_h3, self.b_h3)

            h2o3_mem = self.h2o_3(h2h3_spike)
            output_sumspike = output_sumspike + h2o3_mem

        outputs = output_sumspike / self.T

        return outputs, None

    def gradient(self, input, criterion, target):
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        self.N = input.size(0)

        #'''
        h2h1_mem = h2h1_spike = torch.zeros(self.N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(self.N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(self.N, cfg_fc[2], device=device)
        '''
        h2h1_mem = torch.rand(self.N, cfg_fc[0], device=device)
        h2h2_mem = torch.rand(self.N, cfg_fc[1], device=device)
        h2h3_mem = torch.rand(self.N, cfg_fc[2], device=device)
        h2h1_spike = torch.zeros(self.N, cfg_fc[0], device=device)
        h2h2_spike = torch.zeros(self.N, cfg_fc[1], device=device)
        h2h3_spike = torch.zeros(self.N, cfg_fc[2], device=device)
        '''

        output_sumspike = torch.zeros(self.N, self.output_size, device=device)
        grads = {}

        input = input.view(self.N, -1)

        for step in range(self.T):
            grad_t = {}
            start_idx = step * self.stride
            if start_idx < (self.T - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            ####################################################################
            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_NU_adp(h1_input,
                                                                          h2h1_mem, h2h1_spike,
                                                                          self.tau_adp_h1, self.b_h1)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_NU_adp(h2_input,
                                                                          h2h2_mem, h2h2_spike,
                                                                          self.tau_adp_h2, self.b_h2)

            h3_input = self.i2h_3(h2h2_spike) #+ self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_NU_adp(h3_input,
                                                                          h2h3_mem, h2h3_spike,
                                                                          self.tau_adp_h3, self.b_h3)

            h2o3_mem = self.h2o_3(h2h3_spike)
            output_sumspike = output_sumspike + h2o3_mem

            loss = criterion(output_sumspike, target)
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
            #print(l_t.shape)
        
        # print("l1 sum", l1_sum.sum(1))
        # print("l2 sum", l2_sum.sum(1))
        # print("l3 sum", l3_sum.sum(1))

        outputs = output_sumspike / self.T

        return outputs, grads

    def fire_rate(self, input):
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        self.N = input.size(0)

        #'''
        h2h1_mem = h2h1_spike = h1_spike_sums = torch.zeros(self.N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = h2_spike_sums = torch.zeros(self.N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = h3_spike_sums = torch.zeros(self.N, cfg_fc[2], device=device)
        output_sumspike = torch.zeros(self.N, self.output_size, device=device)

        input = input.view(self.N, -1)

        for step in range(self.T):
            start_idx = step * self.stride
            if start_idx < (self.T - self.input_size):
                input_x = input[:, start_idx:start_idx + self.input_size].reshape(-1, self.input_size)
            else:
                input_x = input[:, -self.input_size:].reshape(-1, self.input_size)

            ####################################################################
            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_NU_adp(h1_input,
                                                                          h2h1_mem, h2h1_spike,
                                                                          self.tau_adp_h1, self.b_h1)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_NU_adp(h2_input,
                                                                          h2h2_mem, h2h2_spike,
                                                                          self.tau_adp_h2, self.b_h2)

            h3_input = self.i2h_3(h2h2_spike) #+ self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_NU_adp(h3_input,
                                                                          h2h3_mem, h2h3_spike,
                                                                          self.tau_adp_h3, self.b_h3)

            h2o3_mem = self.h2o_3(h2h3_spike)
            output_sumspike = output_sumspike + h2o3_mem
            h1_spike_sums += h2h1_spike
            h2_spike_sums += h2h2_spike
            h3_spike_sums += h2h3_spike

        outputs = output_sumspike / self.T

        layer_fr = [h1_spike_sums.sum()/(torch.numel(h2h1_spike)*self.T),
                    h2_spike_sums.sum()/(torch.numel(h2h2_spike)*self.T),
                    h3_spike_sums.sum()/(torch.numel(h2h3_spike)*self.T)]
        layer_fr = torch.tensor(layer_fr)
        hidden_spk = [h1_spike_sums/self.T, h2_spike_sums/self.T, h3_spike_sums/self.T]
        return outputs, hidden_spk, layer_fr #n_nonzeros/n_neurons




class LSNN_noBP(nn.Module):

    def __init__(self):
        super(LSNN_noBP, self).__init__()

        self.input_size = input_size  # num_encode*input_dim
        #self.hidden_size = 300
        self.output_size = output_size
        #self.encode_size = input_size
        self.decode_step = 50

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])  # [1,256]
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])  # [256,256]

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])  # [256,512]
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])  # [512,512]

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])  # [512,256]
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])  # [256,256]
        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)  # [256, 10]

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))

        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))

        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)
        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)
        nn.init.constant_(self.tau_adp_h1, 700)
        nn.init.constant_(self.tau_adp_h2, 700)
        nn.init.constant_(self.tau_adp_h3, 700)
        nn.init.constant_(self.tau_adp_o, 700)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def forward(self, input ):
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        self.N = input.size(0)
        self.T = 784 // self.input_size


        h2h1_mem = h2h1_spike = torch.zeros(self.N, cfg_fc[0], device=device)
        h2h2_mem = h2h2_spike = torch.zeros(self.N, cfg_fc[1], device=device)
        h2h3_mem = h2h3_spike = torch.zeros(self.N, cfg_fc[2], device=device)
        h2o3_mem = h2o3_spike = output_sumspike = torch.zeros(self.N, self.output_size, device=device)

        input = input.view(self.N, self.input_size, -1)

        for step in range(self.T ):

            input_x = input[:, :, step]
            ####################################################################
            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_NU_adp(h1_input,
                                                                          h2h1_mem, h2h1_spike,
                                                                          self.tau_adp_h1, self.b_h1)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_NU_adp(h2_input,
                                                                          h2h2_mem, h2h2_spike,
                                                                          self.tau_adp_h2, self.b_h2)

            h3_input = self.i2h_3(h2h2_spike) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_NU_adp(h3_input,
                                                                          h2h3_mem, h2h3_spike,
                                                                          self.tau_adp_h3, self.b_h3)

            #h2o3_mem, h2o3_spike, theta_o, self.b_o = mem_update_adp(self.h2o_3, h2h3_spike,
            #                                                       h2o3_mem, h2o3_spike, self.tau_adp_o, self.b_o)

            h2o3_mem = self.h2o_3(h2h3_spike)

            output_sumspike = output_sumspike + h2o3_mem

        return output_sumspike


class LSNN_noTD(nn.Module):

    def __init__(self):
        super(LSNN_noTD, self).__init__()

        self.input_size = input_size  # num_encode*input_dim
        # self.hidden_size = 300
        self.output_size = output_size
        # self.encode_size = input_size
        self.decode_step = 50

        self.i2h_1 = nn.Linear(self.input_size, cfg_fc[0])  # [1,256]
        self.h2h_1 = nn.Linear(cfg_fc[0], cfg_fc[0])  # [256,256]

        self.i2h_2 = nn.Linear(cfg_fc[0], cfg_fc[1])  # [256,512]
        self.h2h_2 = nn.Linear(cfg_fc[1], cfg_fc[1])  # [512,512]

        self.i2h_3 = nn.Linear(cfg_fc[1], cfg_fc[2])  # [512,256]
        self.h2h_3 = nn.Linear(cfg_fc[2], cfg_fc[2])  # [256,256]
        self.h2o_3 = nn.Linear(cfg_fc[2], self.output_size)  # [256, 10]

        self.tau_adp_h1 = nn.Parameter(torch.Tensor(cfg_fc[0]))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(cfg_fc[1]))
        self.tau_adp_h3 = nn.Parameter(torch.Tensor(cfg_fc[2]))

        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))

        nn.init.orthogonal_(self.h2h_1.weight)
        nn.init.xavier_uniform_(self.i2h_1.weight)
        nn.init.orthogonal_(self.h2h_2.weight)
        nn.init.xavier_uniform_(self.i2h_2.weight)
        nn.init.orthogonal_(self.h2h_3.weight)
        nn.init.xavier_uniform_(self.i2h_3.weight)
        nn.init.xavier_uniform_(self.h2o_3.weight)
        nn.init.constant_(self.i2h_1.bias, 0)
        nn.init.constant_(self.h2h_1.bias, 0)
        nn.init.constant_(self.i2h_2.bias, 0)
        nn.init.constant_(self.h2h_2.bias, 0)
        nn.init.constant_(self.i2h_3.bias, 0)
        nn.init.constant_(self.h2h_3.bias, 0)
        nn.init.constant_(self.tau_adp_h1, 700)
        nn.init.constant_(self.tau_adp_h2, 700)
        nn.init.constant_(self.tau_adp_h3, 700)
        nn.init.constant_(self.tau_adp_o, 700)
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = 0

    def forward(self, input):
        self.b_h1 = self.b_h2 = self.b_h3 = self.b_o = b_j0
        self.N = input.size(0)
        self.T = 784 // self.input_size

        # h2h1_mem = h2h1_spike = torch.zeros(self.N, cfg_fc[0], device=device)
        # h2h2_mem = h2h2_spike = torch.zeros(self.N, cfg_fc[1], device=device)
        # h2h3_mem = h2h3_spike = torch.zeros(self.N, cfg_fc[2], device=device)
        h2o3_mem = h2o3_spike = output_sumspike = torch.zeros(self.N, self.output_size, device=device)

        input = input.view(self.N, self.input_size, -1)

        for step in range(self.T):
            #### Clear mem potential for every step ######
            h2h1_mem = h2h1_spike = torch.zeros(self.N, cfg_fc[0], device=device)
            h2h2_mem = h2h2_spike = torch.zeros(self.N, cfg_fc[1], device=device)
            h2h3_mem = h2h3_spike = torch.zeros(self.N, cfg_fc[2], device=device)

            input_x = input[:, :, step]
            ####################################################################
            h1_input = self.i2h_1(input_x) + self.h2h_1(h2h1_spike)
            h2h1_mem, h2h1_spike, theta_h1, self.b_h1 = mem_update_NU_adp(h1_input,
                                                                          h2h1_mem, h2h1_spike,
                                                                          self.tau_adp_h1, self.b_h1)

            h2_input = self.i2h_2(h2h1_spike) + self.h2h_2(h2h2_spike)
            h2h2_mem, h2h2_spike, theta_h2, self.b_h2 = mem_update_NU_adp(h2_input,
                                                                          h2h2_mem, h2h2_spike,
                                                                          self.tau_adp_h2, self.b_h2)

            h3_input = self.i2h_3(h2h2_spike) + self.h2h_3(h2h3_spike)
            h2h3_mem, h2h3_spike, theta_h3, self.b_h3 = mem_update_NU_adp(h3_input,
                                                                          h2h3_mem, h2h3_spike,
                                                                          self.tau_adp_h3, self.b_h3)

            # h2o3_mem, h2o3_spike, theta_o, self.b_o = mem_update_adp(self.h2o_3, h2h3_spike,
            #                                                       h2o3_mem, h2o3_spike, self.tau_adp_o, self.b_o)

            h2o3_mem = self.h2o_3(h2h3_spike)

            output_sumspike = output_sumspike + h2o3_mem

        return output_sumspike