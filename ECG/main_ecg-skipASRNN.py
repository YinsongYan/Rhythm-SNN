from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os,time
#os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import torch.nn as nn
import numpy as np
import scipy.io
from tools.lib import dump_json, set_seed, load_max_i, convert_dataset_wtime
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from skip_SRNN.spiking_ECG_model import SRNN_ALIF_mix_v3
from skip_SRNN.Hyperparameters import args

set_seed(1111)

# CUDA configuration
if torch.cuda.is_available():
    device = 'cuda'
    print('GPU is available')
else:
    device = 'cpu'
    print('GPU is not available')


# Path to save checkpoints and log files
snn_ckp_dir = './exp/skip_ASRNN/checkpoint/'
snn_rec_dir = './exp/skip_ASRNN/record/'


num_epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size
arch_name = '-'.join([str(s) for s in args.fc])
skip_name = '-'.join([str(s) for s in args.skip_length])

print('algo is %s, lens = %.2f, in_size = %d, lr = %.5f' %
      (args.algo, args.lens, args.in_size, args.lr))
print('Arch: {0}, skip length: {1}'.format(arch_name, skip_name))

#train_record_path = args.save_path
#train_chkpnt_path = args.chkp_path
train_record_path = 'skip_SRNN_ALIF_step-wise_mix_in4_T{0}_arch{1}_skip{2}_lr{3}'\
    .format((1301), arch_name, skip_name, learning_rate)
train_chk_pnt_path = 'skip_SRNN_ALIF_step-wise_mix_in4_T{0}_arch{1}_skip{2}_lr{3}.pt'\
    .format((1301), arch_name, skip_name, learning_rate)


############################################################
###### Load Dataset  #####
train_mat = scipy.io.loadmat('./data/QTDB_train.mat')
test_mat = scipy.io.loadmat('./data/QTDB_test.mat')

train_dt, train_x, train_y = convert_dataset_wtime(train_mat)
train_max_i = load_max_i(train_mat)
test_dt, test_x, test_y = convert_dataset_wtime(test_mat)
test_max_i = load_max_i(test_mat)

nb_of_sample, seq_dim, input_dim = np.shape(train_x)
print('sequence length: {} , input dimension: {}'.format(seq_dim, input_dim))
print('training dataset distribution: ',train_y.shape)
print('test dataset distribution: ',test_y.shape)

train_data = TensorDataset(torch.from_numpy(train_x*1.),torch.from_numpy(train_y))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=2)
test_data = TensorDataset(torch.from_numpy(test_x*1.),torch.from_numpy(test_y))
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=2)
############################################################


net = SRNN_ALIF_mix_v3(in_size = args.in_size, hidden_size=args.fc[0], output_size=args.out_size)
net = net.to(device)
print(net)

#criterion = nn.MSELoss()  # Mean square error loss
#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
train_best_acc = 0
test_best_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_acc_record = list([])
test_acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
#optimizer = assign_optimizer(net, lrs=learning_rate)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # define weight update method
base_params = [net.i2h.weight, net.i2h.bias, net.h2h.weight,
                   net.h2h.bias, net.h2o.weight, net.h2o.bias]
optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': net.tau_m_h, 'lr': learning_rate * 3},
        {'params': net.tau_m_o, 'lr': learning_rate * 2},
        {'params': net.tau_adp_h, 'lr': learning_rate * 3},
        {'params': net.tau_adp_o, 'lr': learning_rate * 2}, ],
        lr=learning_rate)

#scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
#scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
scheduler = StepLR(optimizer, step_size=100, gamma=0.75)

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
        inputs = inputs.view(-1, seq_dim, input_dim).requires_grad_().to(device)
        labels = labels.view((-1,seq_dim)).long().to(device) # [N , T]
        
        optimizer.zero_grad()
        outputs = net(inputs) #[N, 6, T]

        #loss = criterion(outputs, labels)
        loss = 0
        for i in range(outputs.size(2)):
            loss += criterion(outputs[:,:,i], labels[:, i])

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0) * labels.size(1)

        correct += predicted.eq(labels).sum().item()
        #scheduler.step()

        if (batch_idx + 1) % (len(train_loader)//1) == 0 :
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
            inputs = inputs.view(-1, seq_dim, input_dim).requires_grad_().to(device)
            labels = labels.view((-1,seq_dim)).long().to(device) # [N , T]

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0) * labels.size(1)

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