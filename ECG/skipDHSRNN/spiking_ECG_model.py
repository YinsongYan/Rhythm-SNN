import torch
import torch.nn as nn
from skipDHSRNN.SNN_layers.spike_dense import *
from skipDHSRNN.SNN_layers.spike_neuron import *
from skipDHSRNN.SNN_layers.spike_rnn import *
from skipDHSRNN.Hyperparameters import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
algo = args.algo
thresh = args.thresh
lens = args.lens
decay = args.decay

output_size = args.out_size
input_size = args.in_size
cfg_fc = args.fc

phase_max = args.phase_max
cycle_min = args.cycle_min
cycle_max = args.cycle_max
duty_cycle_min = args.duty_cycle_min
duty_cycle_max = args.duty_cycle_max


# create the network
class DHSRNN(nn.Module):
    def __init__(self, T= 500, low_n = 0, high_n = 4, branch=4):
        super(DHSRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.T = T

        self.rnn_1 = spike_rnn_test_denri_wotanh_R(self.input_size, cfg_fc[0],
                                                       tau_ninitializer='uniform', low_n=low_n, high_n=low_n, vth=thresh, branch=branch,
                                                       dt=1, device=device, bias=True)

        self.dense_2 = spike_dense_test_origin(cfg_fc[0], self.output_size,
                                               vth=0.5, dt=1, device=device, bias=True)


    def forward(self, input):
        batch_size, seq_num, input_dim = input.shape

        self.rnn_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)

        outputs = []
        for i in range(seq_num):
            input_x = input[:, i, :] 
            
            mem_layer1, spike_layer1 = self.rnn_1.forward(input_x)
            mem_layer2, spike_layer2 = self.dense_2.forward(spike_layer1)

            #################   classification  #########################

            output_sum = mem_layer2 
            output_sum = F.log_softmax(output_sum,dim=1) # [N, 6]
            outputs.append(output_sum)
        outputs = torch.stack(outputs).permute(1,2,0) #[N, 6, T]

        return outputs



class rnn_test(nn.Module):
    def __init__(self, T= 500, low_n = 0, high_n = 4, branch=4):
        super(rnn_test, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.T = T

        self.rnn_1 = Rhy_spike_rnn_test_denri_wotanh_R(self.input_size, cfg_fc[0], tau_minitializer='uniform', low_m=0, high_m=0,
                                                       tau_ninitializer='uniform', low_n=low_n, high_n=low_n, vth=thresh, branch=branch,
                                                       dt=1, cycle_min=cycle_min[0], cycle_max=cycle_max[0], duty_cycle_min=duty_cycle_min[0], 
                                                       duty_cycle_max=duty_cycle_max[0], phase_max=phase_max[0], T=T,
                                                       device=device, bias=True)

        self.dense_2 = spike_dense_test_origin(cfg_fc[0], self.output_size,
                                               vth=0.5, dt=1, device=device, bias=True)


    def forward(self, input):
        batch_size, seq_num, input_dim = input.shape

        self.rnn_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)

        outputs = []
        for i in range(seq_num):
            input_x = input[:, i, :] 
            
            mem_layer1, spike_layer1 = self.rnn_1.forward(input_x, i)
            mem_layer2, spike_layer2 = self.dense_2.forward(spike_layer1)

            #################   classification  #########################

            output_sum = mem_layer2 
            output_sum = F.log_softmax(output_sum,dim=1) # [N, 6]
            outputs.append(output_sum)
        outputs = torch.stack(outputs).permute(1,2,0) #[N, 6, T]

        return outputs



class DHSFNN(nn.Module):
    def __init__(self, T= 500, low_n = 0, high_n = 4, branch=4):
        super(DHSFNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.T = T

        self.fsnn_1 = spike_dense_test_denri_wotanh_R(self.input_size, cfg_fc[0],
                                                       tau_ninitializer='uniform', low_n=low_n, high_n=low_n, vth=0.5, branch=branch,
                                                       dt=1, device=device, bias=True)

        self.dense_2 = spike_dense_test_origin(cfg_fc[0], self.output_size,
                                               vth=0.5, dt=1, device=device, bias=True)


    def forward(self, input):
        batch_size, seq_num, input_dim = input.shape

        self.fsnn_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)

        outputs = []
        for i in range(seq_num):
            input_x = input[:, i, :] 
            
            mem_layer1, spike_layer1 = self.fsnn_1.forward(input_x)
            mem_layer2, spike_layer2 = self.dense_2.forward(spike_layer1)

            #################   classification  #########################

            output_sum = mem_layer2 
            output_sum = F.log_softmax(output_sum,dim=1) # [N, 6]
            outputs.append(output_sum)
        outputs = torch.stack(outputs).permute(1,2,0) #[N, 6, T]

        return outputs

