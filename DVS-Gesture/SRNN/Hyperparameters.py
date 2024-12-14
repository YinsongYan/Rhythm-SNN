import argparse

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')

parser.add_argument('--epochs', default=30, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,metavar='N',
					help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--algo', default='SRNN', type=str, metavar='N',
					help='algorithmn for learning')
parser.add_argument('--thresh', default=0.3, type=float, metavar='N',
					help='threshold of the neuron model')
parser.add_argument('--lens', default=0.5, type=float, metavar='N',
					help='lens of surrogate function')
parser.add_argument('--decay', default=0.9, type=float, metavar='N',
					help='decay of the neuron model')
parser.add_argument('--seq_len', default=500, type=int, metavar='N',
					help='sequential length')
parser.add_argument('--in_size', default=1024, type=int, metavar='N',
					help='model input size')
parser.add_argument('--out_size', default=11, type=int, metavar='N',
					help='model output size')
parser.add_argument('--fc', nargs= '+', default=[512, 512, 512], type=int, metavar='N',
					help='model architecture')
parser.add_argument('--seed', default=1111, type=int, metavar='N',
					help='seed')
parser.add_argument('--chkp_path', default='', type=str, metavar='PATH',
					help='path to save the training model (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
					help='path to save the training record (default: none)')

args = parser.parse_args()