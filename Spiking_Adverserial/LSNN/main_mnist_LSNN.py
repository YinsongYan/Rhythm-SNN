from __future__ import print_function
import torchvision.datasets as dsets
import torchvision
import torchvision.transforms as transforms
import os,time

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from spiking_mnist_model_lif_LSNN_v2 import *
#from ntidigits.ntidigits_dataloaders import *

from tensorboardX import SummaryWriter

names = 'spiking_mnist_lsnn_v2_model'
#data_path = './'  # input your path

# CUDA configuration
if torch.cuda.is_available():
    device = 'cuda'
    print('GPU is available')
else:
    device = 'cpu'
    print('GPU is not available')

# Data preprocessing
print('==> Preparing data for %s (thresh = %.5f, lens = %.5f, decay = %.5f)...' % (algo, thresh, lens, decay))


#trainset, testset = create_datasets()
#trainloader,  testloader= create_dataloader(batch_size=batch_size)  # default batch size 72

'''
STEP 1: LOADING DATASET
'''
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

'''
STEP 2: MAKING DATASET ITERABLE
'''
trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


'''
'''

net = LSNN_v2()
net = net.to(device)

#criterion = nn.MSELoss()  # Mean square error loss
criterion = nn.CrossEntropyLoss()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
#optimizer = assign_optimizer(net, lrs=learning_rate)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # define weight update method

#writer = SummaryWriter()
#try_input = torch.rand([20,1000,64,1,1], device=device)
#try_output = net(try_input)

# Training
def train(epoch):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # starts = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.view(-1, seq_dim, input_size).requires_grad_().to(device)  # inputs shape [200,784,1]
        #inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = net(epoch, batch_idx, inputs)
        #labels_ = targets[:, 0, :]  # lables_ [72, 11], targets [72, 1000, 11]
        labels_ = targets  #[200]
        #labels_ = torch.zeros(inputs.size(0), 10).scatter_(1, targets.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.cpu().max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()
        #_, targets_ = labels_.cpu().max(1)
        #correct += float(predicted.eq(targets_).sum().item())

        #writer.add_scalar('Train/loss', loss, epoch)
        #writer.add_scalar('Train/acc', 100. * correct / total, epoch)

        if (batch_idx + 1) % (len(trainloader)//1 ) == 0 :
            elapsed = time.time() - starts
            #print(batch_idx, 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
            #      % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f | Acc: %.5f%% (%d/%d)'
                  % (epoch + 1, num_epochs, batch_idx + 1, len(trainloader), train_loss/ (batch_idx + 1), 100. * correct / total, correct, total))

    print('Time past: ', elapsed, 's', 'Iter number:', epoch+1)
    loss_train_record.append(train_loss)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #inputs = inputs.to(device)
            inputs = inputs.view(-1, seq_dim, input_size).requires_grad_().to(device)  # images shape [200,784,1]
            optimizer.zero_grad()
            outputs = net(epoch, batch_idx, inputs)
            #labels_ = targets[:, 0, :]
            #labels_ = torch.zeros(inputs.size(0), 10).scatter_(1, targets.view(-1, 1), 1)
            labels_ = targets
            loss = criterion(outputs.cpu(), labels_)
            test_loss += loss.item()
            _, predicted = outputs.cpu().max(1)
            total += targets.size(0)
            #_, targets_ = labels_.cpu().max(1)
            correct += float(predicted.eq(targets).sum().item())

            #correct += predicted.eq(targets).sum().item()
            #writer.add_scalar('Test/loss', loss, epoch)
            #writer.add_scalar('Test/acc', 100. * correct / total, epoch)

            if (batch_idx + 1) % (len(testloader)//1 ) == 0:
                print(batch_idx + 1, '/', len(testloader), 'Loss: %.5f | Acc: %.5f%% (%d/%d)'
                    % (test_loss / (batch_idx ), 100. * correct / total, correct, total))
        loss_test_record.append(test_loss)

    # Save checkpoint.
    acc = 100. * correct / total
    acc_record.append(acc)

    if best_acc < acc:
        best_acc = acc
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
            'loss_train_record': loss_train_record,
            'loss_test_record': loss_test_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + names + '.t7')


if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch + num_epochs):
        starts = time.time()
        train(epoch)
        test(epoch)
        elapsed = time.time() - starts
        optimizer = lr_scheduler(optimizer, epoch, init_lr=learning_rate, lr_decay_epoch=50)
        print(" \n\n\n")
        print('Time past: ', elapsed, 's', 'Iter number:', epoch+1)
