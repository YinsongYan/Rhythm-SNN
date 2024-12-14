from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os,time
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.lib import dump_json, set_seed, perturbation_distance
import attack
from ASRNN.spiking_psmnist_model import SRNN_ALIF_2RNN
from skip_ASRNN.spiking_psmnist_model_ALIF import ASRNN_general
from ASRNN.Hyperparameters_psmnist import args
import matplotlib.pyplot as plt
import numpy as np

set_seed(1111)

# CUDA configuration
if torch.cuda.is_available():
    device = 'cuda'
    print('GPU is available')
else:
    device = 'cpu'
    print('GPU is not available')


# Set the font to Arial
plt.rcParams['font.sans-serif'] = ['Arial']  # ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  #
# plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42


snn_ckp_dir = './exp/ASRNN/checkpoint/'
snn_rec_dir = './exp/ASRNN/record/'
data_path = './data'
bb_snn_ckp_dir = os.path.join('./exp/skip_ASRNN/checkpoint/')
bb_snn_rec_dir = os.path.join('./exp/skip_ASRNN/record/')
pretrained_model = 'SRNN_ALIF__in4_T784.0_lens0.2_arch64-256-256_lr0.01_tau5.pt'
# bb_model = 'Skip_FFSNN_general_in1_T784_decay0.9_thr0.3_lens0.2_arch64-256-256_pMax0.5-0.5-0.5_cMin10-10-10_cMax50-50-50_dcMin0.02-0.02-0.02_dcMax0.1-0.1-0.1_lr0.005.pt'
bb_model = 'Rhy_ASRNN_dl0.01_dh0.1_in1_T784_decay0.9_thr0.3_lens0.2_arch64-256-256_pMax0.5-0.5-0.5_cMin10-10-10_cMax100-100-100_dcMin0.01-0.01-0.01_dcMax0.1-0.1-0.1_lr0.01.pt'


num_epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size
arch_name = '-'.join([str(s) for s in args.fc])
eps_factor = 8

print('algo is %s, thresh = %.2f, lens = %.2f, decay = %.2f, in_size = %d, lr = %.5f' %
      (args.algo, args.thresh, args.lens, args.decay, args.in_size, args.lr))
print('Arch: {0}'.format(arch_name))
print('eps_factor:', eps_factor)

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

testloader  = torch.utils.data.DataLoader(ds_val, batch_size = batch_size, shuffle = False, num_workers = 8)

############################################################


net = SRNN_ALIF_2RNN(in_size=1)
net = net.to(device)
print(net)

state = torch.load(snn_ckp_dir + pretrained_model)
missing_keys, unexpected_keys = net.load_state_dict(state['model_state_dict'])#, strict=False)

# Config bb model
bb_net = ASRNN_general(in_size=1)
bb_net = bb_net.to(device)
bb_state = torch.load(bb_snn_ckp_dir + bb_model)
bb_net.load_state_dict(bb_state['model_state_dict'])#, strict=False)


#criterion = nn.MSELoss()  # Mean square error loss
criterion = nn.CrossEntropyLoss()
train_best_acc = 0
test_best_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_acc_record = [] #list([])
test_acc_record = [] #list([])
loss_train_record = [] #list([])
loss_test_record = [] #list([])
fire_rate_record = []

def val(model, test_loader, device, atk=None):
    correct = 0
    total = 0
    firing_rate = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
        if atk is not None:
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets.to(device))
        with torch.no_grad():
            outputs, _, fr = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        firing_rate += fr
    final_acc = 100 * correct / total
    print('layer wise firing rate:', firing_rate / len(test_loader))
    return final_acc

def val_analysis(model, test_loader, device, atk=None):
    correct = 0
    correct_atk = 0
    total = 0
    firing_rate = 0
    firing_rate_atk = 0
    perturb_dist_input = 0
    perturb_dist_h1 = 0
    perturb_dist_h2 = 0
    perturb_dist_h3 = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
        if atk is not None:
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs_atk = atk(inputs, targets.to(device))
        else:
            inputs_atk = inputs
        with torch.no_grad():
            outputs, hidden, fr = model(inputs)
            outputs_atk, hidden_atk, fr_atk = model(inputs_atk)
        _, predicted = outputs.cpu().max(1)
        _, predicted_atk = outputs_atk.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        correct_atk += float(predicted_atk.eq(targets).sum().item())

        firing_rate += fr
        firing_rate_atk += fr_atk
        dist_norm_input = perturbation_distance(inputs, inputs_atk)
        dist_norm_h1 = perturbation_distance(hidden[0], hidden_atk[0])
        dist_norm_h2 = perturbation_distance(hidden[1], hidden_atk[1])
        dist_norm_h3 = perturbation_distance(hidden[2], hidden_atk[2])
        perturb_dist_input += dist_norm_input
        perturb_dist_h1 += dist_norm_h1
        perturb_dist_h2 += dist_norm_h2
        perturb_dist_h3 += dist_norm_h3
    final_acc = 100 * correct / total
    final_acc_atk = 100 * correct_atk / total
    print('layer wise Original firing rate:', firing_rate / len(test_loader))
    print('layer wise Attacked firing rate:', firing_rate_atk / len(test_loader))
    print('input perturb_dist:', perturb_dist_input / len(test_loader))
    print('h1 perturb_dist:', perturb_dist_h1 / len(test_loader))
    print('h2 perturb_dist:', perturb_dist_h2 / len(test_loader))
    print('h3 perturb_dist:', perturb_dist_h3 / len(test_loader))
    print('Original acc:', final_acc)
    print('Attacked acc:', final_acc_atk)
    print('Delta accuracy:', final_acc - final_acc_atk)
    return final_acc_atk

def val_fr_analysis(model, test_loader, device, atk=None):
    correct = 0
    correct_atk = 0
    total = 0
    h1_firing_rate = 0
    h1_firing_rate_atk = 0
    h2_firing_rate = 0
    h2_firing_rate_atk = 0
    h3_firing_rate = 0
    h3_firing_rate_atk = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)
        if atk is not None:
            atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs_atk = atk(inputs, targets.to(device))
        else:
            inputs_atk = inputs
        with torch.no_grad():
            outputs, hidden, fr = model(inputs)
            outputs_atk, hidden_atk, fr_atk = model(inputs_atk)
        _, predicted = outputs.cpu().max(1)
        _, predicted_atk = outputs_atk.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        correct_atk += float(predicted_atk.eq(targets).sum().item())

        h1_firing_rate += fr[0]
        h1_firing_rate_atk += fr_atk[0]
        h2_firing_rate += fr[1]
        h2_firing_rate_atk += fr_atk[1]
        h3_firing_rate += fr[2]
        h3_firing_rate_atk += fr_atk[2]

    final_acc = 100 * correct / total
    final_acc_atk = 100 * correct_atk / total

    h1_firing_rate = h1_firing_rate / len(test_loader)
    h1_firing_rate_atk = h1_firing_rate_atk / len(test_loader)
    h2_firing_rate = h2_firing_rate / len(test_loader)
    h2_firing_rate_atk = h2_firing_rate_atk / len(test_loader)
    h3_firing_rate = h3_firing_rate / len(test_loader)
    h3_firing_rate_atk = h3_firing_rate_atk / len(test_loader)

    print('h1_firing_rate:', h1_firing_rate[-1])
    print('h1_firing_rate_atk:', h1_firing_rate_atk[-1])
    print('h2_firing_rate:', h2_firing_rate[-1])
    print('h2_firing_rate_atk:', h2_firing_rate_atk[-1])
    print('h3_firing_rate:', h3_firing_rate[-1])
    print('h3_firing_rate_atk:', h3_firing_rate_atk[-1])

    print('Original acc:', final_acc)
    print('Attacked acc:', final_acc_atk)
    print('Delta accuracy:', final_acc-final_acc_atk)
    return final_acc_atk

atk_fgsm = attack.FGSM(bb_net, eps=eps_factor / 255)

atk_pgd = attack.PGD(bb_net, eps=eps_factor / 255, alpha=0.01, steps=7)

atk_gn = attack.GN(bb_net, eps=eps_factor / 255)

def plot_single_sample(model, test_loader, device, atk=None, noise_type=""):
    model.eval()
    sample_inputs, sample_targets = next(iter(test_loader))
    sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)
    
    if atk is not None:
        atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
        sample_inputs_atk = atk(sample_inputs, sample_targets)
    else:
        sample_inputs_atk = sample_inputs
    
    with torch.no_grad():
        sample_outputs, sample_hidden, _ = model(sample_inputs)
        sample_outputs_atk, sample_hidden_atk, _ = model(sample_inputs_atk)
    
    plot_diff_images(
        sample_inputs[0].cpu().numpy(), 
        sample_inputs_atk[0].cpu().numpy(), 
        [h[0] for h in sample_hidden],
        [h[0] for h in sample_hidden_atk],
        noise_type
    )


def plot_images(original, perturbed, original_hidden_layers, perturbed_hidden_layers, noise_type):
    num_layers = len(original_hidden_layers)
    fig, axes = plt.subplots(2, num_layers + 1, figsize=(15, 10))
    
    # Plot original image
    axes[0, 0].imshow(original.reshape(28, 28), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot original hidden layer representations
    for i, hidden in enumerate(original_hidden_layers):
        size = int(np.sqrt(hidden.numel()))
        hidden_reshaped = hidden.cpu().numpy().reshape(size, size)  # Reshape to 2D square array
        axes[0, i + 1].imshow(hidden_reshaped, cmap='viridis')
        axes[0, i + 1].set_title(f'Hidden Layer {i+1}')
        axes[0, i + 1].axis('off')
    
    # Plot perturbed image
    axes[1, 0].imshow(perturbed.reshape(28, 28), cmap='gray')
    axes[1, 0].set_title('Perturbed Image')
    axes[1, 0].axis('off')
    
    # Plot perturbed hidden layer representations
    for i, hidden in enumerate(perturbed_hidden_layers):
        size = int(np.sqrt(hidden.numel()))
        hidden_reshaped = hidden.cpu().numpy().reshape(size, size)  # Reshape to 2D square array
        axes[1, i + 1].imshow(hidden_reshaped, cmap='viridis')
        axes[1, i + 1].set_title(f'Hidden Layer {i+1}')
        axes[1, i + 1].axis('off')
    
    plt.show()
    plt.savefig('./plot/figures_attack/{}_ASRNN_bb_hidden_representaion.pdf'.format(noise_type),   bbox_inches='tight')

def plot_diff_images(original, perturbed, original_hidden_layers, perturbed_hidden_layers, noise_type):
    print("original min", np.min(original))
    print("original max", np.max(original))
    print("perturbed min", np.min(perturbed))
    print("perturbed max", np.max(perturbed))
    
    orig_hidden_values = []
    for i, orig_hidden in enumerate(original_hidden_layers):
        orig_hidden_values.extend(orig_hidden.cpu().numpy().flatten())
    
    print("orig_hidden min", np.min(orig_hidden_values))
    print("orig_hidden max", np.max(orig_hidden_values))

    all_values = []
    for i, (orig_hidden, pert_hidden) in enumerate(zip(original_hidden_layers, perturbed_hidden_layers)):
        diff_hidden = pert_hidden - orig_hidden
        all_values.extend(diff_hidden.cpu().numpy().flatten())
    
    print("diff_hidden min", np.min(all_values))
    print("diff_hidden max", np.max(all_values))
    
    num_layers = len(original_hidden_layers)
    fig, axes = plt.subplots(2, num_layers + 1, figsize=(15, 10))
    
    # Plot original image
    axes[0, 0].imshow(original.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot original hidden layer representations
    for i, hidden in enumerate(original_hidden_layers):
        size = int(np.sqrt(hidden.numel()))
        hidden_reshaped = hidden.cpu().numpy().reshape(size, size)  # Reshape to 2D square array
        im_hidden = axes[0, i + 1].imshow(hidden_reshaped, cmap='viridis', vmin=0, vmax=1)
        axes[0, i + 1].set_title(f'Hidden Layer {i+1}')
        axes[0, i + 1].axis('off')
    
    # Add a single color bar for all hidden layer representations
    fig.colorbar(im_hidden, ax=axes[0, 1:num_layers + 1], orientation='vertical', fraction=0.013, pad=0.04)
    
    # Plot perturbed image
    axes[1, 0].imshow((perturbed ).reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Perturbed Image')
    axes[1, 0].axis('off')
    
    # Plot the difference between original and perturbed hidden layer representations
    for i, (orig_hidden, pert_hidden) in enumerate(zip(original_hidden_layers, perturbed_hidden_layers)):
        size = int(np.sqrt(orig_hidden.numel()))
        orig_hidden_reshaped = orig_hidden.cpu().numpy().reshape(size, size)
        pert_hidden_reshaped = pert_hidden.cpu().numpy().reshape(size, size)
        diff_hidden_reshaped = orig_hidden_reshaped - pert_hidden_reshaped  # Calculate the difference
        im_diff = axes[1, i + 1].imshow(diff_hidden_reshaped, cmap='viridis', vmin=-0.55, vmax=0.5)
        axes[1, i + 1].set_title(f'Difference Layer {i+1}')
        axes[1, i + 1].axis('off')
    
    # Add a single color bar for all difference representations
    fig.colorbar(im_diff, ax=axes[1, 1:num_layers + 1], orientation='vertical', fraction=0.013, pad=0.04)
    
    plt.savefig('./plot/figures_attack/{}_ASRNN_bb_hid_feats_diff.pdf'.format(noise_type),   bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch + num_epochs):
        starts = time.time()
        #test(epoch)
        # acc_test = val(net, testloader, device, None)
        # print('acc_test:', acc_test)

        print('Analysing FGSM.....')
        acc_fgsm = val_analysis(net, testloader, device, atk_fgsm)
        print('Analysing PGD.....')
        acc_pgd = val_analysis(net, testloader, device, atk_pgd)
        print('Analysing GN.....')
        acc_gn = val_analysis(net, testloader, device, atk_gn)


        # print('Analysing FGSM.....')
        # plot_single_sample(net, testloader, device, atk_fgsm, "FGSM")
        # #acc_fgsm = val_analysis(net, testloader, device, atk_fgsm)
        # print('Analysing PGD.....')
        # plot_single_sample(net, testloader, device, atk_fgsm, "PGD")
        # #acc_pgd = val_analysis(net, testloader, device, atk_pgd)
        # print('Analysing GN.....')
        # plot_single_sample(net, testloader, device, atk_fgsm, "GN")


        #acc_gn = val_analysis(net, testloader, device, atk_gn)
        #print('acc_test:', acc_test)
        # print('acc_fgsm:', acc_fgsm)
        # print('acc_pgd:', acc_pgd)
        # print('acc_gn:', acc_gn)

        elapsed = time.time() - starts
        print('Time past: ', elapsed, 's', 'Iter number:', epoch+1)
    print('=====================End of trail=============================\n\n')