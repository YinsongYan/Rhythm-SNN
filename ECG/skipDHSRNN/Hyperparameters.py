import argparse

parser = argparse.ArgumentParser(description='PyTorch IMDB Training')

parser.add_argument('--epochs', default=250, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,metavar='N',
					help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--algo', default='SRNN', type=str, metavar='N',
					help='algorithmn for learning')
parser.add_argument('--thresh', default=0.5, type=float, metavar='N',
					help='threshold of the neuron model')
parser.add_argument('--lens', default=0.5, type=float, metavar='N',
					help='lens of surrogate function')
parser.add_argument('--decay', default=0.5, type=float, metavar='N',
					help='decay of the neuron model')
parser.add_argument('--in_size', default=4, type=int, metavar='N',
					help='model input size')
parser.add_argument('--out_size', default=6, type=int, metavar='N',
					help='model output size')
parser.add_argument('--branch', default=1, type=int, metavar='N',
					help='branch of neuron model')
parser.add_argument('--low_n', default=0, type=int, metavar='N',
					help='branch of neuron model')
parser.add_argument('--high_n', default=4, type=int, metavar='N',
					help='branch of neuron model')

parser.add_argument('--phase_max', nargs='*', default=[0.0, 0.0], type=float, metavar='N',
					help='maximum repeating cycle')
parser.add_argument('--cycle_min', nargs='*', default=[10, 10], type=int, metavar='N',
					help='minimum repeating cycle')
parser.add_argument('--cycle_max', nargs='*', default=[50, 50], type=int, metavar='N',
					help='maximum repeating cycle')
parser.add_argument('--duty_cycle_min', nargs='*', default=[0.95, 0.95], type=float, metavar='L',
					help='min duty cycle for each layer')
parser.add_argument('--duty_cycle_max', nargs='*', default=[1.0, 1.0], type=float, metavar='L',
					help='max duty cycle for each layer')
					
parser.add_argument('--fc', nargs= '+', default=[36], type=int, metavar='N',
					help='model architecture')
parser.add_argument('--num_steps', default=1301, type=int, metavar='N',
					help='time steps')
parser.add_argument('--chkp_path', default='', type=str, metavar='PATH',
					help='path to save the training model (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
					help='path to save the training record (default: none)')

args = parser.parse_args()