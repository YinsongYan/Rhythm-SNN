
import torch.nn as nn


DROP_RATE = 0.01

grads = []
def save_grad(name):
    def hook(grad):
        grads.append(grad)
    return hook

class lstm(nn.Module):
    def __init__(self, INPUT_SIZE=1, OUT_SIZE=1, LAYERS = 1, HIDDEN_SIZE=64):
        super(lstm, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            dropout=DROP_RATE,
            batch_first=True
        )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, OUT_SIZE)
        self.h_s = None
        self.h_c = None

    def forward(self, x, task='duration'):
        r_out, (self.h_s,self.h_c) = self.rnn(x.transpose(1,2))
        output = self.hidden_out(r_out)
        if task == 'duration':
            return output[:,-1,:]
        elif task == 'syn':
            return output.permute(0,2,1)
        elif task == 'interval':
            return output[:, -1, :]
        elif task == 'recall':
            return output.permute(0,2,1)

    def get_grads(self, x):
        r_out, (self.h_s,self.h_c) = self.rnn(x.transpose(1,2))
        r_out.register_hook(save_grad('grads'))
        output = self.hidden_out(r_out)

        return output.permute(0,2,1), grads
