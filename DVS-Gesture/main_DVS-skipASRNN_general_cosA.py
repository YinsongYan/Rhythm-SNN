from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os,time
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
import torch.nn as nn
import numpy as np
import scipy.io
from tools.lib import dump_json, set_seed
from torch.optim.lr_scheduler import StepLR
from skipASRNN.spiking_dvs_model import skipSRNN_ALIF_1Adapt_general
from skipASRNN.Hyperparameters import args
from tools.datasets import data_generator 

seed = args.seed
set_seed(seed)
print("seed:", seed)

# CUDA configuration
if torch.cuda.is_available():
    device = 'cuda'
    print('GPU is available')
else:
    device = 'cpu'
    print('GPU is not available')


snn_ckp_dir = './logs/skipASRNN/general/checkpoint/'
snn_rec_dir = './logs/skipASRNN/general/record/'
data_path = './data'


num_epochs = args.epochs
learning_rate = args.lr
seq_len = args.seq_len
batch_size = args.batch_size
arch_name = '-'.join([str(s) for s in args.fc])
max_phase_name = '-'.join([str(s) for s in args.phase_max])
min_cycle_name = '-'.join([str(s) for s in args.cycle_min])
max_cycle_name = '-'.join([str(s) for s in args.cycle_max])
min_dc_name = '-'.join([str(s) for s in args.duty_cycle_min])
max_dc_name = '-'.join([str(s) for s in args.duty_cycle_max])

print('algo is %s, lens = %.2f, in_size = %d, lr = %.5f, T = %d' %
      (args.algo, args.lens, args.in_size, args.lr, seq_len))
print('General mask. Arch: {0}, max phase: {1}, min cycle: {2}, max cycle: {3}, min duty cycle: {4}, max duty cycle: {5}'.format(arch_name, max_phase_name, min_cycle_name, max_cycle_name, min_dc_name, max_dc_name))

#train_record_path = args.save_path
#train_chkpnt_path = args.chkp_path
train_record_path = 'SkipASRNN_general_T{0}_decay{1}_thr{2}_lens{3}_arch{4}_pMax{5}_cMin{6}_cMax{7}_dcMin{8}_dcMax{9}_lr{10}_s{11}'\
    .format(args.seq_len, args.decay, args.thresh, args.lens, arch_name, max_phase_name, min_cycle_name, max_cycle_name, min_dc_name, max_dc_name, learning_rate, seed) #args.save_path
train_chk_pnt_path = 'SkipASRNN_general_T{0}_decay{1}_thr{2}_lens{3}_arch{4}_pMax{5}_cMin{6}_cMax{7}_dcMin{8}_dcMax{9}_lr{10}_s{11}.pt'\
    .format(args.seq_len, args.decay, args.thresh, args.lens, arch_name, max_phase_name, min_cycle_name, max_cycle_name, min_dc_name, max_dc_name, learning_rate, seed) #args.chkp_path


########################################################################################################################
###### Load Dataset  #############
train_loader, test_loader, seq_length, input_channels, n_classes = data_generator('DVS-Gesture',
                                                                    batch_size=batch_size,
                                                                    dataroot='./data/',
                                                                    T=seq_len)
########################################################################################################################


net = skipSRNN_ALIF_1Adapt_general()
net = net.to(device)
print(net)

#criterion = nn.MSELoss()  # Mean square error loss
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
train_best_acc = 0
test_best_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_acc_record = list([])
test_acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # define weight update method
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)#num_epochs) 


# Training
def train(epoch):
    print('\nEpoch: %d' % (epoch + 1))
    global train_best_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # starts = time.time()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(-1, seq_length, 2,128,128)
        
        optimizer.zero_grad()
        outputs = net(inputs) #[N, 6, T]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()
        #scheduler.step()

        if (batch_idx + 1) % (len(train_loader)//5) == 0 :
            elapsed = time.time() - starts

            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f | Acc: %.5f%% (%d/%d)'
                  % (epoch + 1, num_epochs, batch_idx + 1, len(train_loader), train_loss/ (batch_idx + 1), 100. * correct / total, correct, total))

    print('Train time past: ', elapsed, 's', 'Iter number:', epoch+1)
    train_acc = 100. * correct / total
    loss_train_record.append(train_loss/(batch_idx+1))
    train_acc_record.append(train_acc)
    if train_best_acc < train_acc:
        train_best_acc = train_acc

def test(epoch):
    global test_best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, seq_length, 2,128,128)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % (len(test_loader)//1) == 0:
                print(batch_idx + 1, '/', len(test_loader), 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    test_acc_record.append(acc)
    loss_test_record.append(test_loss/(batch_idx+1))
    if test_best_acc < acc:
        test_best_acc = acc

        # Save Model
        print("Saving the model.")
        if not os.path.isdir(snn_ckp_dir):
            os.makedirs(snn_ckp_dir)
        state = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'acc': test_best_acc,
        }
        #torch.save(state, snn_ckp_dir + train_chk_pnt_path)


if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch + num_epochs):
        starts = time.time()
        train(epoch)
        test(epoch)
        elapsed = time.time() - starts
        scheduler.step()
        print('Time past: ', elapsed, 's', 'Iter number:', epoch+1)
    print(" Best Train Acc: ", train_best_acc)
    print(" Best Test Acc: ", test_best_acc)
    print('=====================End of trail=============================\n\n')

    if not os.path.isdir(snn_ckp_dir):
        os.makedirs(snn_ckp_dir)

    training_record = {
        'learning_rate': args.lr,
        'algo': args.algo,
        'thresh': args.thresh,
        'lens': args.lens,
        'decay': args.decay,
        'architecture': args.fc,
        'loss_test_record': loss_test_record,
        'loss_train_record': loss_train_record,
        'test_acc_record': test_acc_record,
        'train_acc_record': train_acc_record,
        'train_best_acc': train_best_acc,
        'test_best_acc': test_best_acc,
    }
    dump_json(training_record, snn_rec_dir, train_record_path)