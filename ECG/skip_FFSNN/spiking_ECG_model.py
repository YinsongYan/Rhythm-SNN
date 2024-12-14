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
skip_length_min = args.skip_length_min

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


#################################################################
class FFSNN_mix(nn.Module):
    """
    Mix skip_length range within layers. With Min and Max skip value.
    """
    def __init__(self,in_size, hidden_size, output_size):
        super(FFSNN_mix, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size , self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        self.mask1 = self.create_mix_mask(self.hidden_size, skip_length_min[0], skip_length[0])
        self.mask2 = self.create_mix_mask(self.output_size, skip_length_min[1], skip_length[1])

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
        batch_size, seq_num, input_dim = input.shape
        h1_mem = h1_spike = torch.zeros(batch_size, self.hidden_size, device=device)
        h2_mem = h2_spike = output_sum = torch.zeros(batch_size, self.output_size, device=device)

        outputs = []
        for step in range(seq_num): 
            input_x = input[:, step, :] 

            h1_mem, h1_spike = mem_update_skip_woDecay(self.fc1, input_x.float(), h1_mem, h1_spike, self.mask1[:, step])
            h2_mem, h2_spike = mem_update_skip_woDecay(self.fc2, h1_spike, h2_mem, h2_spike, self.mask2[:, step])
            #h2_mem = self.fc2(h1_spike)
            
            #################   classification  #########################
            #if step >= self.sub_seq_length:
            output_sum = h2_mem 
            output_sum = F.log_softmax(output_sum,dim=1) # [N, 6]
            outputs.append(output_sum)
        outputs = torch.stack(outputs).permute(1,2,0) #[N, 6, T]

        return outputs
