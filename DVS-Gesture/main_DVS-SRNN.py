from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os,time
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.nn as nn
import numpy as np
import scipy.io
from tools.lib import dump_json, set_seed
from torch.optim.lr_scheduler import StepLR
from SRNN.spiking_dvs_model import SRNN
from SRNN.Hyperparameters import args
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



snn_ckp_dir = './logs/SRNN/checkpoint/'
snn_rec_dir = './logs/SRNN/record/'
data_path = './data'




num_epochs = args.epochs
learning_rate = args.lr
seq_len = args.seq_len
batch_size = args.batch_size
arch_name = '-'.join([str(s) for s in args.fc])

print('algo is %s, lens = %.2f, in_size = %d, lr = %.5f, T = %d' %
      (args.algo, args.lens, args.in_size, args.lr, seq_len))
print('Arch: {0}'.format(arch_name))

#train_record_path = args.save_path
#train_chkpnt_path = args.chkp_path
train_record_path = 'SRNN_T{0}_decay{1}_thr{2}_lens{3}_arch{4}_lr{5}_s{6}'\
    .format((seq_len), args.decay, args.thresh, args.lens,arch_name, learning_rate, seed)
train_chk_pnt_path = 'SRNN_T{0}_decay{1}_thr{2}_lens{3}_arch{4}_lr{5}_s{6}.pt'\
    .format((seq_len), args.decay, args.thresh, args.lens,arch_name, learning_rate, seed)


########################################################################################################################
###### Load Dataset  #############
train_loader, test_loader, seq_length, input_channels, n_classes = data_generator('DVS-Gesture',
                                                                    batch_size=batch_size,
                                                                    dataroot='./data/',
                                                                    T=seq_len)
########################################################################################################################


net = SRNN()
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
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
#scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs) 

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
        torch.save(state, snn_ckp_dir + train_chk_pnt_path)


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