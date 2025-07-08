import argparse

parser = argparse.ArgumentParser(description='model')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int,metavar='N',   # 256
					help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--algo', default='model', type=str, metavar='N',
					help='algorithmn for learning')
parser.add_argument('--te', default='TE-R', type=str, choices=['ALIF', 'RhythmALIF','LIF','DEXAT','RhythmDEXAT','LSTM'],
					help='algorithmn for learning')
parser.add_argument('--seq_len', default=100, type=int, metavar='N',
					help='sequence length')
parser.add_argument('--max_duration', default=10, type=int, metavar='N',
					help='max duration')
parser.add_argument('--beta', default=0.005, type=float, metavar='N',
					help='Decay factor of V')
parser.add_argument('--thresh', default=0.3, type=float, metavar='N',
					help='threshold of the neuron model')
parser.add_argument('--lens', default=0.2, type=float, metavar='N',
					help='lens of surrogate function')
parser.add_argument('--decay', default=0.5, type=float, metavar='N',
					help='decay of the neuron model')
parser.add_argument('--in_size', default=2, type=int, metavar='N',
					help='model input size')
parser.add_argument('--out_size', default=1, type=int, metavar='N',
					help='model output size')

parser.add_argument('--phase_max', nargs='*', default=[0.5, 0.5, 0.5], type=float, metavar='N',
					help='maximum repeating cycle')
parser.add_argument('--cycle_min', nargs='*', default=[1, 1, 1], type=int, metavar='N',
					help='minimum repeating cycle')
parser.add_argument('--cycle_max', nargs='*', default=[10, 10, 10], type=int, metavar='N',
					help='maximum repeating cycle')
parser.add_argument('--duty_cycle_min', nargs='*', default=[0.8, 0.8, 0.8], type=float, metavar='L',
					help='min duty cycle for each layer')
parser.add_argument('--duty_cycle_max', nargs='*', default=[0.9, 0.9, 0.9], type=float, metavar='L',
					help='max duty cycle for each layer')

parser.add_argument('--fc', nargs= '+', default=[256, 256], type=int, metavar='N',
					help='model architecture')
parser.add_argument('--grad-clip', type=float, default=0.01)
parser.add_argument('--chkp_path', default='', type=str, metavar='PATH',
					help='path to save the training model (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
					help='path to save the training record (default: none)')
parser.add_argument('--seed', type=int, default=1112)  # default=1111, 1110, 1112

args = parser.parse_args()