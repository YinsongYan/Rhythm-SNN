import torch
import torch.nn as nn
import torch.nn.functional as F
from skip_LSNN.Hyperparameters import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
algo = args.algo
lens = args.lens #0.3

skip_length = args.skip_length
skip_length_min = args.skip_length_min

'''
STEP 3a_v2: CREATE Adaptative spike MODEL CLASS
'''
b_j0 = 0.01  # neural threshold baseline
tau_m = 5 #20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1  #
gamma = .5  # gradient scale
print('tau_m', tau_m)


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

def mem_update_NU_adp_skip(inputs, mem, spike, tau_adp, b, mask, dt=1, isAdapt=1):
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

########################################################################
class LSNN_mix(nn.Module):
    """
    Mix skip_length within layers. With min and max skip value.
    """
    def __init__(self, in_size, hidden_size, output_size):
        super(LSNN_mix, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

        self.tau_adp_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tau_adp_o = nn.Parameter(torch.Tensor(self.output_size))

        self._initial_parameters()
        self.b_h = self.b_o = 0

        self.mask1 = self.create_mix_mask(self.hidden_size, skip_length_min[0], skip_length[0])
        self.mask2 = self.create_mix_mask(self.output_size, skip_length_min[1], skip_length[1])

    def _initial_parameters(self):
        nn.init.orthogonal_(self.h2h.weight)
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2o.weight)
        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)

        nn.init.constant_(self.tau_adp_h, 7) # 700
        nn.init.constant_(self.tau_adp_o, 100) # 700

    def create_mix_mask(self, dim=128, min_cycle=0, max_cycle=0):
        T = 1301
        mask_cyc = []
        for cycle in range(min_cycle, max_cycle+1):
            mask_ = []
            for t in range(T):
                if t % cycle == 0:
                    mask_.append(1)
                else:
                    mask_.append(0)
            mask_ = torch.tensor(mask_)
            #tmp1 = mask_.detach().numpy()
            mask_cyc.append(mask_)
        mask_cyc = torch.stack(mask_cyc)
        #tmp2 = mask_cyc.detach().numpy()

        mask = mask_cyc
        for n in range(1, dim//(max_cycle-min_cycle+1) + 1):
            mask = torch.cat((mask, torch.roll(mask_cyc, n, 1)), 0)
            #tmp3 = mask.detach().numpy()
        return mask[: dim].to(device)  # [H, T]

    def forward(self, input):
        self.b_h = self.b_o = b_j0
        batch_size, seq_num, input_dim = input.shape
        #h2h_mem = h2h_spike = torch.zeros(batch_size, self.hidden_size, device=device)
        #h2o_mem = h2o_spike = output_sumspike = torch.zeros(batch_size, self.output_size, device=device)
        h2h_mem = h2h_spike = torch.rand(batch_size, self.hidden_size, device=device)*b_j0
        h2o_mem = h2o_spike = output_sumspike = torch.rand(batch_size, self.output_size, device=device)*b_j0

        outputs = []
        for step in range(seq_num):  # seq_num = 784 (pixel by pixel)
            input_x = input[:, step, :]

            ####################################################################
            h_input = self.i2h(input_x.float()) + self.h2h(h2h_spike)  # h_input [N,128] : [80-->128] + [128-->128]
            h2h_mem, h2h_spike, theta_h, self.b_h = mem_update_NU_adp_skip(h_input,
                                                                     h2h_mem, h2h_spike, self.tau_adp_h, self.b_h, self.mask1[:, step])

            o_input = self.h2o(h2h_spike)
            h2o_mem, h2o_spike, theta_o, self.b_o = mem_update_NU_adp_skip(o_input,
                                                                   h2o_mem, h2o_spike, self.tau_adp_o, self.b_o, self.mask2[:, step])
            
            #################   classification  #########################
            #if step >= self.sub_seq_length:
            output_sumspike = h2o_mem 
            output_sumspike = F.log_softmax(output_sumspike,dim=1) # [N, 6]
            outputs.append(output_sumspike)
        outputs = torch.stack(outputs).permute(1,2,0) #[N, 6, T]

        return outputs
    
