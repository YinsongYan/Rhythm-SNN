from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os,time
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
from tools.lib import dump_json, set_seed
from torch.optim.lr_scheduler import StepLR
from SRNN.spiking_psmnist_model import SRNN_ALIF_2RNN
from SRNN.Hyperparameters import args

set_seed(1111)

# CUDA configuration
if torch.cuda.is_available():
    device = 'cuda'
    print('GPU is available')
else:
    device = 'cpu'
    print('GPU is not available')



# Path to save checkpoints and log files
snn_ckp_dir = './exp/ASRNN/checkpoint/'
snn_rec_dir = './exp/ASRNN/record/'
data_path = './data'


num_epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size
arch_name = '-'.join([str(s) for s in args.fc])

print('algo is %s, lens = %.2f, in_size = %d, lr = %.5f' %
      (args.algo, args.lens, args.in_size, args.lr))
print('Arch: {0}'.format(arch_name))

#train_record_path = args.save_path
#train_chkpnt_path = args.chkp_path
train_record_path = 'SRNN_ALIF_2RNN_in4_T{0}_lens{1}_arch{2}_lr{3}_tau5'\
    .format((784/args.in_size), args.lens, arch_name, learning_rate)
train_chk_pnt_path = 'SRNN_ALIF_2RNN_in4_T{0}_lens{1}_arch{2}_lr{3}_tau5.pt'\
    .format((784/args.in_size), args.lens, arch_name, learning_rate)


############################################################
class psMNIST(torch.utils.data.Dataset):
    """ Dataset that defines the psMNIST dataset, given the MNIST data and a fixed permutation """

    def __init__(self, mnist, perm):
        self.mnist = mnist # also a torch.data.Dataset object
        self.perm  = perm

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        unrolled = img.reshape(-1)
        permuted = unrolled[self.perm]
        permuted = permuted.reshape(-1, 1)
        return permuted, label

transform = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(root= data_path, train = True, download = True, transform = transform)
mnist_val   = torchvision.datasets.MNIST(root= data_path, train = False, download = True, transform = transform)

perm = torch.load("./ps_data/permutation.pt").long() # created using torch.randperm(784)
ds_train = psMNIST(mnist_train, perm)
ds_val   = psMNIST(mnist_val, perm)

trainloader = torch.utils.data.DataLoader(ds_train, batch_size = batch_size, shuffle = True, num_workers = 8)
testloader  = torch.utils.data.DataLoader(ds_val, batch_size = batch_size, shuffle = False, num_workers = 8)
############################################################


net = SRNN_ALIF_2RNN(in_size = 4)
net = net.to(device)

print(net)

#criterion = nn.MSELoss()  # Mean square error loss
criterion = nn.CrossEntropyLoss()
train_best_acc = 0
test_best_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_acc_record = list([])
test_acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
#optimizer = assign_optimizer(net, lrs=learning_rate)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # define weight update method
scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

# Training
def train(epoch):
    print('\nEpoch: %d' % (epoch + 1))
    global train_best_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # starts = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.view(-1, 784, 1).requires_grad_().to(device)
        #inputs = inputs.to(device)  # [72, 500, 2, 32, 32]
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs.cpu(), targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.cpu().max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()
        #_, targets_ = labels_.cpu().max(1)
        #correct += float(predicted.eq(targets_).sum().item())

        if (batch_idx + 1) % (len(trainloader)//10) == 0 :
            elapsed = time.time() - starts

            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f | Acc: %.5f%% (%d/%d)'
                  % (epoch + 1, num_epochs, batch_idx + 1, len(trainloader), train_loss/ (batch_idx + 1), 100. * correct / total, correct, total))

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
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.view(-1, 784, 1).requires_grad_().to(device)
            #inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs.cpu(), targets)
            test_loss += loss.item()
            _, predicted = outputs.cpu().max(1)
            total += targets.size(0)
            #_, targets_ = labels_.cpu().max(1)
            #correct += float(predicted.eq(targets_).sum().item())

            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % (len(testloader)//1) == 0:
                print(batch_idx + 1, '/', len(testloader), 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
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
        #optimizer = lr_scheduler(optimizer, epoch, init_lr=learning_rate)
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