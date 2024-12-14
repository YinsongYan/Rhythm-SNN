from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os,time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.lib import dump_json, set_seed, perturbation_distance
import attack
from skip_ASRNN.spiking_psmnist_model_ALIF import ASRNN_general, Rhy_ASRNN_thermal, Rhy_ASRNN_silence, W_Quantize
from skip_ASRNN.Hyperparameters_psmnist import args
import copy
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

# Data preprocessing
snn_ckp_dir = './exp/skip_ASRNN/checkpoint/'
snn_rec_dir = './exp/skip_ASRNN/record/'
data_path = './data'


# Set the font to Arial
plt.rcParams['font.sans-serif'] = ['Arial']  # ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 
# plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42



pretrained_model = 'Rhy_ASRNN_dl0.01_dh0.1_in1_T784_decay0.9_thr0.3_lens0.2_arch64-256-256_pMax0.5-0.5-0.5_cMin10-10-10_cMax100-100-100_dcMin0.01-0.01-0.01_dcMax0.1-0.1-0.1_lr0.01.pt'
# pretrained_model = 'Rhy_ASRNN_dl0.1_dh0.1_in1_T784_decay0.9_thr0.3_lens0.2_arch64-256-256_pMax0.5-0.5-0.5_cMin10-10-10_cMax100-100-100_dcMin0.1-0.1-0.1_dcMax0.1-0.1-0.1_lr0.01.pt'
# pretrained_model = 'Rhy_ASRNN_dl0.1_dh0.2_in1_T784_decay0.9_thr0.3_lens0.2_arch64-256-256_pMax0.5-0.5-0.5_cMin10-10-10_cMax100-100-100_dcMin0.1-0.1-0.1_dcMax0.2-0.2-0.2_lr0.01.pt'





num_epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size
arch_name = '-'.join([str(s) for s in args.fc])
max_phase_name = '-'.join([str(s) for s in args.phase_max])
min_cycle_name = '-'.join([str(s) for s in args.cycle_min])
max_cycle_name = '-'.join([str(s) for s in args.cycle_max])
min_dc_name = '-'.join([str(s) for s in args.duty_cycle_min])
max_dc_name = '-'.join([str(s) for s in args.duty_cycle_max])

#noise = "noise_mm"
noise = "noise_thermal"
# noise = "noise_silence"
# noise = "noise_quant"
level_factor = 0.20   # 0.30  # 0.20

print('algo is %s, thresh = %.2f, lens = %.2f, decay = %.2f, in_size = %d, lr = %.5f' %
      (args.algo, args.thresh, args.lens, args.decay, args.in_size, args.lr))
print('General mask. Arch: {0}, max phase: {1}, min cycle: {2}, max cycle: {3}, min duty cycle: {4}, max duty cycle: {5}'.format(arch_name, max_phase_name, min_cycle_name, max_cycle_name, min_dc_name, max_dc_name))
print('Noise:', noise)
print('level_factor:', level_factor)

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
# Original model
net = ASRNN_general(in_size=1)
net = net.to(device)
print (net)

# Noise model
# net_mm = FFSNN_general_mismatch(std_p=level_factor)
net_thermal = Rhy_ASRNN_thermal(noise_std=level_factor)
net_silence = Rhy_ASRNN_silence(silence_p=level_factor)

# Load pre-trained models
state = torch.load(snn_ckp_dir + pretrained_model)
missing_keys, unexpected_keys = net.load_state_dict(state['model_state_dict'])#, strict=False)
# missing_keys, unexpected_keys = net_mm.load_state_dict(state['model_state_dict'])#, strict=False)
missing_keys, unexpected_keys = net_thermal.load_state_dict(state['model_state_dict'])#, strict=False)
missing_keys, unexpected_keys = net_silence.load_state_dict(state['model_state_dict'])#, strict=False)

# if noise == "noise_mm":
#     noise_model = net_mm.to(device)
#     noise_model._apply_mismatch()
if noise == "noise_thermal":
    noise_model = net_thermal.to(device)
elif noise == "noise_silence":
    noise_model = net_silence.to(device)
elif noise == "noise_quant":
    numBits = 4 # 6
    print("numBits:", numBits)
    noise_model = W_Quantize(copy.deepcopy(net), numBits=numBits, device=device) # apply weight quantization
    noise_model = noise_model.to(device)
    net = net.to(device)
else:
    noise_model = net.to(device)
print (noise_model)

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

def val_analysis(model, test_loader, device, noise_model):
    correct = 0
    correct_noise = 0
    total = 0
    firing_rate = 0
    firing_rate_noise = 0
    perturb_dist_input = 0
    perturb_dist_h1 = 0
    perturb_dist_h2 = 0
    perturb_dist_h3 = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate((test_loader)):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs, hidden, fr = model(inputs)
            outputs_noise, hidden_noise, fr_noise = noise_model(inputs)

        _, predicted = outputs.cpu().max(1)
        _, predicted_noise = outputs_noise.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        correct_noise += float(predicted_noise.eq(targets).sum().item())

        firing_rate += fr
        firing_rate_noise += fr_noise
        dist_norm_input = perturbation_distance(inputs, inputs)
        dist_norm_h1 = perturbation_distance(hidden[0], hidden_noise[0])
        dist_norm_h2 = perturbation_distance(hidden[1], hidden_noise[1])
        dist_norm_h3 = perturbation_distance(hidden[2], hidden_noise[2])
        perturb_dist_input += dist_norm_input
        perturb_dist_h1 += dist_norm_h1
        perturb_dist_h2 += dist_norm_h2
        perturb_dist_h3 += dist_norm_h3
    final_acc = 100 * correct / total
    final_acc_noise = 100 * correct_noise / total
    print('layer wise Original firing rate:', firing_rate / len(test_loader))
    print('layer wise Noise firing rate:', firing_rate_noise / len(test_loader))
    print('input perturb_dist:', perturb_dist_input / len(test_loader))
    print('h1 perturb_dist:', perturb_dist_h1 / len(test_loader))
    print('h2 perturb_dist:', perturb_dist_h2 / len(test_loader))
    print('h3 perturb_dist:', perturb_dist_h3 / len(test_loader))
    print('Original acc:', final_acc)
    print('Noise acc:', final_acc_noise)
    print('Delta accuracy:', final_acc - final_acc_noise)
    return final_acc_noise

def plot_single_sample(model, test_loader, device, noise_type=""):
    model.eval()
    sample_inputs, sample_targets = next(iter(test_loader))
    sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)

    with torch.no_grad():
        sample_outputs, sample_hidden, _ = model(sample_inputs)
        sample_outputs_noise, sample_hidden_noise, _ = noise_model(sample_inputs)
    
    plot_diff_images(
        sample_inputs[0].cpu().numpy(), 
        sample_inputs[0].cpu().numpy(), 
        [h[0] for h in sample_hidden],
        [h[0] for h in sample_hidden_noise],
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
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    # Plot perturbed hidden layer representations
    for i, hidden in enumerate(perturbed_hidden_layers):
        size = int(np.sqrt(hidden.numel()))
        hidden_reshaped = hidden.cpu().numpy().reshape(size, size)  # Reshape to 2D square array
        axes[1, i + 1].imshow(hidden_reshaped, cmap='viridis')
        axes[1, i + 1].set_title(f'Hidden Layer {i+1}')
        axes[1, i + 1].axis('off')
    
    plt.show()
    plt.savefig('./plot/figures_intrinsic/{}_skipASRNN_hidden_representaion.pdf'.format(noise_type),   bbox_inches='tight')

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
        im_diff = axes[1, i + 1].imshow(diff_hidden_reshaped, cmap='viridis', vmin=-0.47, vmax=0.10)
        axes[1, i + 1].set_title(f'Difference Layer {i+1}')
        axes[1, i + 1].axis('off')
    
    # Add a single color bar for all difference representations
    fig.colorbar(im_diff, ax=axes[1, 1:num_layers + 1], orientation='vertical', fraction=0.013, pad=0.04)
    
    plt.savefig('./plot/figures_intrinsic/{}_skipASRNN_hid_feats_diff.pdf'.format(noise_type),   bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch + num_epochs):
        starts = time.time()
        
        print('Analysing noise.....')
        # plot_single_sample(net, testloader, device, noise)
        acc_noise = val_analysis(net, testloader, device, noise_model)

        elapsed = time.time() - starts
        print('Time past: ', elapsed, 's', 'Iter number:', epoch+1)
    print('=====================End of trail=============================\n\n')