import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FFSNN.Hyperparameters import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algo = args.algo
thresh = args.thresh
lens = args.lens
decay = args.decay


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
def mem_update(ops, x, mem, spike): # ops weight shape [32, 1, 3, 3], x [250, 1, 28, 28], mem [250, 32, 28, 28], spike [250, 32, 28, 28]
    mem = mem * decay * (1. - spike) + ops(x)   # mem: AddBackward, spike: ActFunBackward
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike


#################################################################
class FFSNN(nn.Module):
    def __init__(self,in_size, hidden_size, output_size):
        super(FFSNN, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size , self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        batch_size, seq_num, input_dim = input.shape
        h1_mem = h1_spike = torch.zeros(batch_size, self.hidden_size, device=device)
        h2_mem = h2_spike = output_sum = torch.zeros(batch_size, self.output_size, device=device)

        outputs = []
        for step in range(seq_num): 
            input_x = input[:, step, :] 

            h1_mem, h1_spike = mem_update(self.fc1, input_x.float(), h1_mem, h1_spike)
            h2_mem = self.fc2(h1_spike)
            
            #################   classification  #########################
            #if step >= self.sub_seq_length:
            output_sum = h2_mem 
            output_sum = F.log_softmax(output_sum,dim=1) # [N, 6]
            outputs.append(output_sum)
        outputs = torch.stack(outputs).permute(1,2,0) #[N, 6, T]

        return outputs