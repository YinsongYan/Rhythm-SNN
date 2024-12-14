# coding: utf-8
import argparse
import time
import math
import os
import copy
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import random
import data

###############################################################################
# Define Activation Function for SNNs
###############################################################################
lens = 0.5
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float() / (2 * lens)

WinFunc = ActFun.apply

###############################################################################
# Define NN Model
###############################################################################

class NNModel(nn.Module):
    def __init__(self, 
                 ntoken, ninp, nn_shape, 
                 lstm_or_snn= "lstm", 
                 dropout=None, device="cuda"):
        
        super(NNModel, self).__init__()
        
        self.ntoken      = ntoken
        self.ninp        = ninp
        self.nn_shape    = nn_shape
        self.dropout     = nn.Dropout(dropout)
        self.device      = torch.device(device)
        self.lstm_or_snn = lstm_or_snn
        
        # constant parameters
        self.nlayers = 2
        self.decay   = 0.6
        self.thresh  = 0.6
        
        self.act_fun = WinFunc

        # spike resolution : population coding
        self.n_res = 1

        # encoder-decoder definition
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nn_shape[-1] // self.n_res, ntoken) # caution!
        
        # neuron weight definition
        if lstm_or_snn == "lstm":
            self.lstm_wi1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(4, ninp, nn_shape[0])))
            self.lstm_wi2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(4, nn_shape[0], nn_shape[1]))) # caution!
            self.lstm_wh1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(4, nn_shape[0], nn_shape[0])))
            self.lstm_wh2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(4, nn_shape[1], nn_shape[1])))
        elif lstm_or_snn == "snn":
            self.snn_fc1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(ninp, nn_shape[0])))
            self.tau_m1 = nn.Parameter(nn.init.uniform_(torch.Tensor(nn_shape[0]), -1, 1))
            self.snn_fc2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(nn_shape[0], nn_shape[1])))
            self.tau_m2 = nn.Parameter(nn.init.uniform_(torch.Tensor(nn_shape[1]), -1, 1))

        else:
            NotImplementedError("Unknown model type")

        if lstm_or_snn == "lstm":
            self.neuron_forward = self.lstm_forward
        elif lstm_or_snn == "snn":
            self.neuron_forward = self.snn_forward
        else:
            NotImplementedError("Unknown model type")
        
        self.init_weights()

    def init_weights(self):
        # initialize the weights
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        # initialize the hidden tensors to zero
        if self.lstm_or_snn == "lstm":
            hidden1 = torch.zeros([bsz, self.nn_shape[0]], dtype=torch.float32, device=self.device) 
            hidden2 = torch.zeros([bsz, self.nn_shape[1]], dtype=torch.float32, device=self.device)
            c1      = torch.zeros([bsz, self.nn_shape[0]], dtype=torch.float32, device=self.device) 
            c2      = torch.zeros([bsz, self.nn_shape[1]], dtype=torch.float32, device=self.device) 
            return (hidden1, hidden2, c1, c2)
        elif self.lstm_or_snn == "snn":
            hidden1 = torch.zeros([bsz, 0], dtype=torch.float32, device=self.device) 
            hidden2 = torch.zeros([bsz, 0], dtype=torch.float32, device=self.device)
            c1      = torch.zeros([bsz, 0], dtype=torch.float32, device=self.device) 
            c2      = torch.zeros([bsz, 0], dtype=torch.float32, device=self.device)
            return (hidden1, hidden2, c1, c2)

    def lstm_update(self, wi, wh, x, h, c):
        i = (x.mm(wi[0]) + h.mm(wh[0])).sigmoid()
        f = (x.mm(wi[1]) + h.mm(wh[1])).sigmoid()    
        g = (x.mm(wi[2]) + h.mm(wh[2])).tanh()
        o = (x.mm(wi[3]) + h.mm(wh[3])).sigmoid()
        
        _c = torch.mul(f,c) + torch.mul(i, g)
        _h = o * torch.tanh(_c)
        
        return (_h, _c)
    
    def snn_update(self, fc, inputs, mem, spike, tau_m):
        state = inputs.mm(fc)
        decay = torch.sigmoid(tau_m)
        mem = mem * (1 - spike) * decay + state
        now_spike = self.act_fun(mem - self.thresh)
        return mem, now_spike.float()    

    # def lstm_forward(self, input, hidden):
    #     n_win, batch_size, input_size = input.size()
    #     h1_y, h2_y, c1, c2 = hidden
    #     buf = []
    #     for t in range(n_win):
    #         h1_y, c1 = self.lstm_update(self.lstm_wi1, self.lstm_wh1, input[t], h1_y, c1)
    #         h2_y, c2 = self.lstm_update(self.lstm_wi2, self.lstm_wh2, h1_y,     h2_y, c2)
    #         buf.append(h2_y)
    #     stacked_output = torch.stack(buf, dim=0)
    #     return stacked_output, (h1_y, h2_y, c1, c2)

    def lstm_forward(self, input, hidden):
        n_win, batch_size, input_size = input.size()
        h1_y, h2_y, c1, c2 = hidden
        buf = []
        for t in range(n_win):
            h1_y, c1 = self.lstm_update(self.lstm_wi1, self.lstm_wh1, input[t], h1_y, c1)
            h2_y, c2 = self.lstm_update(self.lstm_wi2, self.lstm_wh2, h1_y,     h2_y, c2)
            buf.append(h2_y)
        reshaped_tensor = torch.stack(buf, dim=0).view(n_win, batch_size, -1, self.n_res)
        stacked_output = reshaped_tensor.mean(dim=-1)
        return stacked_output, (h1_y, h2_y, c1, c2)

    def snn_forward(self, input, hidden):
        n_win, batch_size, input_size = input.size()
        h1_mem = h1_spike = torch.zeros(batch_size, self.nn_shape[0], device = self.device)
        h2_mem = h2_spike = torch.zeros(batch_size, self.nn_shape[1], device = self.device)
        buf = []
        for t in range(n_win):
            h1_mem, h1_spike = self.snn_update(self.snn_fc1, input[t], h1_mem, h1_spike, self.tau_m1)
            h2_mem, h2_spike = self.snn_update(self.snn_fc2, h1_spike, h2_mem, h2_spike, self.tau_m2)
            buf.append(h2_spike)
        stacked_output = torch.stack(buf, dim=0)
        # reshaped_tensor = torch.stack(buf, dim=0).view(n_win, batch_size, -1, self.n_res)
        # stacked_output = reshaped_tensor.mean(dim=-1)
        return stacked_output, hidden
        
    def forward(self, raw_input, hidden):
        
        # input vector embedding
        emb   = self.encoder(raw_input)
        input = self.dropout(emb)
        
        # neuron forward through multiple timesteps
        stacked_output, _hidden = self.neuron_forward(input, hidden)

        # dropout and decoded
        output  = self.dropout(stacked_output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), _hidden  


###############################################################################
# Parse arguments
###############################################################################

parser = argparse.ArgumentParser(description='NN Model on Language Dataset')

parser.add_argument('--data', type=str,
                    default='./data/penn-treebank',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='snn', help='type of network (lstm, snn)')
parser.add_argument('--mode', type=str, default='train', help='type of operation (train, test)')
# Activate parameters
parser.add_argument('--batch_size', type=int, default=25, metavar='N', help='batch size')
parser.add_argument('--train_epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--lr', type=float, default=0.5, help='upper epoch limit')
parser.add_argument('--lr_decay', type=int, default=25, help='lr decay interval')

# Default parameters
parser.add_argument('--emsize', type=int, default=650, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650, help='number of hidden units per layer')  # 120
parser.add_argument('--bptt', type=int, default=200, help='sequence length')
parser.add_argument('--clip', type=float, default=10, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--gpu', type=str, default='1', help='gpu number')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.manual_seed(args.seed)
device = torch.device("cuda")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)


###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

    print('reset optimizer with the learning rate', optimizer.param_groups[0]['lr'])
    return optimizer

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


eval_batch_size = args.batch_size
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)

criterion = nn.CrossEntropyLoss()
l1_criterion = nn.L1Loss()


def train(arg_model, optimizers, epoch, logger):
    arg_model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = arg_model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)

        hidden = repackage_hidden(hidden)
        arg_model.zero_grad()
        output, hidden = arg_model.forward(data, hidden)

        target_loss = criterion(output.view(-1, ntokens), targets)

        both_loss = target_loss
        both_loss.backward()
        torch.nn.utils.clip_grad_norm_(arg_model.parameters(), args.clip)

        optimizers.step()
        optimizers.zero_grad()

        total_loss += target_loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed  = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(arg_model, data_source):
    # Turn on evaluation mode which disables dropout.
    arg_model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = arg_model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = arg_model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

def train_over_epoch(arg_model, epoch_num, optimizer, lr, decay_epoch, logger, arg_model_name):
    # record variables
    valid_loss_record = []
    best_val_loss = None
    best_model = None
    last_decay = 0
    # train epoch_num epochs
    for epoch in range(1, epoch_num + 1):
        epoch_start_time = time.time()
        train(arg_model, optimizer, epoch, logger)
        val_loss = evaluate(arg_model, val_data)
        valid_loss_record.append(copy.deepcopy(math.exp(val_loss)))
        logger.info('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        logger.info('-' * 89)

        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(arg_model)
            state = {
                'net' : arg_model.state_dict(),
                'seed': args.seed,
                'nn_shape': arg_model.nn_shape,
                'lstm_or_snn': arg_model.lstm_or_snn,
                'ntoken'   : arg_model.ntoken,
                'ninp'     : arg_model.ninp,
                'val_loss_record': valid_loss_record
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + arg_model_name + ".t7")
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.25
            last_decay = epoch
        if (epoch - last_decay) >= decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            last_decay = epoch
    test_trained_model(best_model, logger)
    return best_model

def test_trained_model(arg_model, logger):
    test_loss = evaluate(arg_model, test_data)
    logger.info('=' * 89)
    logger.info('| Performance on Test Set | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info('=' * 89)
      

def count_parameters(model):
    param_sum = 0
    for p in model.parameters():
        if p.requires_grad:
            print(p.size(), p.numel())
            param_sum = param_sum + p.numel()
    return param_sum

def train_origin_model(model_name, logger):
    if args.model in ["lstm", "snn"]:
        nn_shape = [args.nhid, args.nhid]
    else:
        NotImplementedError("Unknown model type")
    
    logger.info("model with nn_shape: [{:3d}, {:3d}]".format(nn_shape[0], nn_shape[1]))

    init_model = NNModel(ntokens, args.emsize, nn_shape=nn_shape, lstm_or_snn=args.model, dropout=args.dropout, device="cuda").to(device)
    n = count_parameters(init_model)
    print("Number of parameters: %s" % n)
    
    optimizer    = optim.SGD(init_model.parameters(), lr=args.lr, momentum=0.9)

    trained_model = train_over_epoch(init_model, args.train_epochs, optimizer, args.lr, args.lr_decay, logger, model_name)
    return trained_model


def logger_generation(file_name):
    if not os.path.isdir('log'):
        os.mkdir('log')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    fh = logging.FileHandler("./log/" + file_name + ".log")
    fh.setLevel(logging.DEBUG)
    
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def load_model(arg_model_name):
    ckpt_path = "./checkpoint/" + arg_model_name + ".t7"
    ckpts     = torch.load(ckpt_path, map_location="cpu")
    weights_dict   = ckpts["net"]
    tmp_nn_shape   = ckpts["nn_shape"]
    lstm_or_snn    = ckpts["lstm_or_snn"]
    tmp_ntoken     = ckpts["ntoken"]
    tmp_ninp       = ckpts["ninp"]
    tmp_model      = NNModel(ntoken=tmp_ntoken, 
                             ninp=tmp_ninp, 
                             nn_shape=tmp_nn_shape, 
                             lstm_or_snn=lstm_or_snn, 
                             dropout=args.dropout, 
                             device=device).to(device)
    tmp_model.load_state_dict(weights_dict)
    return tmp_model 


assert args.model in ["lstm", "snn"]

if args.data.find("wiki") >= 0:
    dataset_name = "wiki"
else:
    assert args.data.find("penn") >= 0 or args.data.find("ptb") >= 0
    dataset_name = "ptb"

if args.mode == "train":
    model_name = dataset_name + "_" + args.model + "_plif" + "_" + str(args.seed) + "_nhid" + str(args.nhid) + "_t" + str(args.bptt)
    logfile_name = "train" + "_" + model_name
    train1_logger = logger_generation(logfile_name)
    trained_model = train_origin_model(model_name, train1_logger)

elif args.mode == "test":
    model_name = dataset_name + "_" + args.model + "_" + str(args.seed)
    logfile_name = "test" + "_" + model_name
    test1_logger  = logger_generation(logfile_name)
    trained_model = load_model(model_name)
    test_trained_model(trained_model, test1_logger)
else:
    NotImplementedError("Unknown mode")
