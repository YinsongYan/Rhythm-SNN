import argparse

parser = argparse.ArgumentParser(description='PyTorch psmnit Training')

parser.add_argument('--epochs', default=150, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N',
					help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--algo', default='STBP', type=str, metavar='N',
					help='algorithmn for learning')
parser.add_argument('--thresh', default=0.3, type=float, metavar='N',
					help='threshold of the neuron model')
parser.add_argument('--lens', default=0.2, type=float, metavar='N',
					help='lens of surrogate function')
parser.add_argument('--decay', default=0.9, type=float, metavar='N',
					help='decay of the neuron model')
parser.add_argument('--in_size', default=1, type=int, metavar='N',
					help='model input size')
parser.add_argument('--out_size', default=10, type=int, metavar='N',
					help='model output size')
parser.add_argument('--fc', nargs= '+', default=[64, 256, 256], type=int, metavar='N',
					help='model architecture')
parser.add_argument('--chkp_path', default='', type=str, metavar='PATH',
					help='path to save the training model (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
					help='path to save the training record (default: none)')

args = parser.parse_args()