"""
This code is modified from the work: Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. Nature communications, 11(1), 3625.

"""

import datetime
from datetime import datetime
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import socket
from time import time
import matplotlib.pyplot as plt
from matplotlib import collections as mc, patches
from matplotlib.ticker import MultipleLocator
import numpy as np
import numpy.random as rd
import random
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys

# sys.path.append("..")
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from lib import dump_json
from lsnn.spiking_seqmnist_model import SRNN_ALIF_fix, Gen_skip_SRNN_ALIF_fix, Gen_skip_SRNN_DEXAT_mix, SRNN_DEXAT

from lsnn.toolbox.file_saver_dumper_no_h5py import save_file
from lsnn.toolbox.matplotlib_extension import strip_right_top_axis, raster_plot
from lsnn.toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik
from lsnn.toolbox.tensorflow_utils import torch_downsample
# from lsnn.spiking_models import tf_cell_to_savable_dict, placeholder_container_for_rnn_state,\
#     feed_dict_with_placeholder_container, exp_convolve, ALIF
from tutorial_storerecall_utils import generate_storerecall_data, error_rate, gen_custom_delay_batch

gc.collect()
torch.cuda.empty_cache()

# # CUDA configuration
# if torch.cuda.is_available():
#     device = 'cuda'
#     print('GPU is available')
# else:
#     device = 'cpu'
#     print('GPU is not available')
#
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

device = 'cpu'

script_name = os.path.basename(__file__)[:-3]
result_folder = 'results/' + script_name + '/'
# FLAGS = tf.app.flags.FLAGS

parser = argparse.ArgumentParser(description='STORE-RECALL Task')

##
parser.add_argument('--comment', default='', type=str, metavar='N',
                    help='comment to retrieve the stored results')
##
parser.add_argument('--batch_train', default=64, type=int, metavar='N',
                    help='batch size fo the training set')
parser.add_argument('--batch_val', default=64, type=int, metavar='N',
                    help='batch size of the validation set')
parser.add_argument('--batch_test', default=64, type=int, metavar='N',
                    help='batch size of the testing set')
parser.add_argument('--n_in', default=100, type=int, metavar='N',
                    help='number of input units')
parser.add_argument('--f0', default=50, type=int, metavar='N',
                    help='input firing rate')
parser.add_argument('--reg_rate', default=10, type=int, metavar='N',
                    help='target rate for regularization')
parser.add_argument('--n_iter', default=200, type=int, metavar='N',
                    help='number of training iterations')
parser.add_argument('--n_delay', default=10, type=int, metavar='N',
                    help='maximum synaptic delay')
parser.add_argument('--n_ref', default=3, type=int, metavar='N',
                    help='Number of refractory steps')
parser.add_argument('--seq_len', default=12, type=int, metavar='N',
                    help='Number of character steps')
parser.add_argument('--seq_delay', default=6, type=int, metavar='N',
                    help='Expected delay in character steps. Must be <= seq_len - 2')
parser.add_argument('--tau_char', default=100, type=int, metavar='N',
                    help='Duration of symbols (frequency of input value changes)')
parser.add_argument('--seed', default=42, type=int, metavar='N',
                    help='Random seed')
parser.add_argument('--lr_decay_every', default=80, type=int, metavar='N',
                    help='Decay every')
parser.add_argument('--print_every', default=20, type=int, metavar='N',
                    help='Validation frequency')
##
parser.add_argument('--stop_crit', default=0.05, type=float, metavar='N',
                    help='Stopping criterion. Stops training if error goes below this value')
parser.add_argument('--thr', default=0.01, type=float, metavar='N',
                    help='Baseline threshold at which the LSNN neurons spike')
parser.add_argument('--beta', default=1.7, type=float, metavar='N',
                    help='adaptive threshold beta scaling parameter')
parser.add_argument('--tau_a', default=1200, type=float, metavar='N',
                    help='adaptation time constant (threshold decay time constant)')
parser.add_argument('--tau_out', default=20, type=float, metavar='N',
                    help='tau for PSP decay in LSNN and output neurons')
parser.add_argument('--learning_rate', default=0.05, type=float, metavar='N',
                    help='Base learning rate')
parser.add_argument('--lr_decay', default=0.3, type=float, metavar='N',
                    help='Learning rate decaying factor')
parser.add_argument('--reg', default=1e-2, type=float, metavar='N',
                    help='Firing rate regularization coefficient (scaling regularization loss)')
parser.add_argument('--dampening_factor', default=0.3, type=float, metavar='N',
                    help='Parameter necessary to approximate the spike derivative')
##
parser.add_argument('--save_data', default=True, type=bool, metavar='N',
                    help='Save the data (training, test, network, trajectory for plotting)')
parser.add_argument('--do_plot', default=True, type=bool, metavar='N',
                    help='Perform plots')
parser.add_argument('--monitor_plot', default=True, type=bool, metavar='N',
                    help='Perform plots during training')
parser.add_argument('--interactive_plot', default=True, type=bool, metavar='N',
                    help='Perform plots')
parser.add_argument('--device_placement', default=False, type=bool, metavar='N',
                    help='')
parser.add_argument('--verbose', default=True, type=bool, metavar='N',
                    help='')

FLAGS = parser.parse_args()

# Run asserts to check seq_delay and seq_len relation is ok
_ = gen_custom_delay_batch(FLAGS.seq_len, FLAGS.seq_delay, 1)

# Fix the random seed if given as an argument
if FLAGS.seed >= 0:
    seed = FLAGS.seed
else:
    seed = rd.randint(10 ** 6)
rd.seed(seed)
# tf.set_random_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Experiment parameters
dt = 1.
repeat_batch_test = 10
print_every = FLAGS.print_every

# Frequencies
input_f0 = FLAGS.f0 / 1000  # in kHz in coherence with the usgae of ms for time
regularization_f0 = FLAGS.reg_rate / 1000

# Network parameters
tau_v = FLAGS.tau_out
thr = FLAGS.thr

decay = np.exp(-dt / FLAGS.tau_out)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
# Symbol number
n_charac = 2  # Number of digit symbols
n_input_symbols = n_charac + 2  # Total number of symbols including recall and store
n_output_symbols = n_charac  # Number of output symbols
recall_symbol = n_input_symbols - 1  # ID of the recall symbol
store_symbol = n_input_symbols - 2  # ID of the store symbol

# Neuron population sizes
input_neuron_split = np.array_split(np.arange(FLAGS.n_in), n_input_symbols)



# Define the number of iterations and other constants
learning_rate = FLAGS.learning_rate


model = Gen_skip_SRNN_ALIF_fix(in_size=FLAGS.n_in, bias=True, n_out=n_charac, tau_a=600)
# model = SRNN_ALIF_fix(in_size=FLAGS.n_in, bias=True, n_out=n_charac, tau_a=600)
# model = Gen_skip_SRNN_DEXAT_mix(in_size=FLAGS.n_in, bias=True, n_out=n_charac, tau_a1=30, tau_a2=600)
# model = SRNN_DEXAT(in_size=FLAGS.n_in, bias=True, n_out=n_charac, tau_a1=30, tau_a2=600)


model_name = 'Gen_skip_SRNN_ALIF_fix(tau_a=600)'
# model_name = 'SRNN_ALIF_fix(tau_a=600)'
# model_name = 'Gen_skip_SRNN_DEXAT(tau_a1=30, tau_a2=600)'
# model_name = 'SRNN_DEXAT(tau_a1=30, tau_a2=600)'


model = model.to(device)
print(model)
# Define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



print('\n -------------- \n' + model_name + '\n -------------- \n')
time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

file_reference = '{}_{}_seqlen{}_seqdelay{}_in{}_lr{}_tauchar{}_comment{}'.format(
    time_stamp, model_name, FLAGS.seq_len, FLAGS.seq_delay, FLAGS.n_in, FLAGS.learning_rate,
    FLAGS.tau_char, FLAGS.comment)

print('FILE REFERENCE: ' + file_reference)


# Generate input data streams
def get_data_dict(batch_size, seq_len=FLAGS.seq_len, batch=None, override_input=None):
    p_sr = 1 / (1 + FLAGS.seq_delay)
    spk_data, is_recall_data, target_seq_data, memory_seq_data, in_data, target_data = generate_storerecall_data(
        batch_size=batch_size,
        f0=input_f0,
        sentence_length=seq_len,
        n_character=2,
        n_charac_duration=FLAGS.tau_char,
        n_neuron=FLAGS.n_in,
        prob_signals=p_sr,
        with_prob=True,
        override_input=override_input,
    )
    data_dict = {'input_spikes': spk_data, 'input_nums': in_data, 'target_nums': target_data,
                 'recall_mask': is_recall_data,
                 'target_sequence': target_seq_data, 'batch_size_holder': batch_size}

    return data_dict


def convert_data_dict(data_dict):
    # Convert your data_dict tensors to PyTorch tensors
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            data_dict[key] = torch.tensor(value, dtype=torch.float32)
        if key == 'input_spikes':
            data_dict[key] = torch.tensor(value, dtype=torch.float32, device=device)
    return data_dict


# Define the loss function. In this work, we don't add the Firing rate regularization term to our loss function
class CustomLoss(nn.Module):
    def __init__(self, decay, n_output_symbols):
        super(CustomLoss, self).__init__()
        self.decay = decay
        self.n_output_symbols = n_output_symbols

    def forward(self, z, target_nums, input_nums, fire_rate):
        n_neurons = z.size(2)

        n_recall_symbol = n_charac + 1
        recall_charac_mask = torch.eq(input_nums, n_recall_symbol)

        # Calculate Y and Y_predict
        target_nums_at_recall = target_nums[recall_charac_mask].to(dtype=torch.int64)

        # out = z.detach().clone().cpu()
        out = z.cpu()
        out_char_step = torch_downsample(out, new_size=FLAGS.seq_len, axis=1)
        Y_predict = out_char_step[recall_charac_mask]

        # Calculate recall loss
        loss_recall = F.cross_entropy(Y_predict, target_nums_at_recall)  # .item()


        non_recall_charac_mask = ~recall_charac_mask
        out_char_step_non_recall = out_char_step[non_recall_charac_mask]
        loss_other = F.mse_loss(out_char_step_non_recall, torch.full_like(out_char_step_non_recall, 0.5))


        # Calculate regularization loss
        # Firing rate regularization
        # av = torch.mean(z, dim=(0, 1)) / dt
        # regularization_coeff = torch.tensor(np.ones(n_neurons) * FLAGS.reg, dtype=torch.float32, device=z.device)

        av = fire_rate
        regularization_coeff = torch.tensor(FLAGS.reg, dtype=torch.float32, device=z.device)

        loss_reg = torch.sum(torch.square(av - regularization_f0) * regularization_coeff)  # .item()

        # Total loss
        # loss = loss_reg + loss_recall
        loss = loss_recall + 0.01 * loss_other

        results_tensors = {
            'loss': loss,
            'loss_reg': loss_reg,
            'loss_recall': loss_recall,
            'regularization_coeff': regularization_coeff,
        }

        return results_tensors


# Create the custom loss function
criterion = CustomLoss(decay=decay, n_output_symbols=n_output_symbols)




# Open an interactive matplotlib window to plot in real time
if FLAGS.do_plot and FLAGS.interactive_plot:
    plt.ion()
if FLAGS.do_plot:
    fig, ax_list = plt.subplots(5, figsize=(5.9, 7))
    # fig, ax_list = plt.subplots(3, figsize=(5.9, 6))
    # re-name the window with the name of the cluster to track relate to the terminal window
    fig.canvas.manager.set_window_title(socket.gethostname() + ' - ' + FLAGS.comment)


def update_plot(plot_result_values, batch=0, n_max_neuron_per_raster=20):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    # Plot the data, from top to bottom each axe represents: inputs, recurrent and controller
    for k_data, data, d_name in zip(range(2),
                                    [plot_result_values['input_spikes'], plot_result_values['z']],
                                    ['Input', 'Hidden']):

        ax = ax_list[k_data]
        ax.grid(color='black', alpha=0.15, linewidth=0.4)
        # print(np.size(data))
        if np.size(data) > 0:
            data = data[batch]
            n_max = min(data.shape[1], n_max_neuron_per_raster)
            cell_select = np.linspace(start=0, stop=data.shape[1] - 1, num=n_max, dtype=int)
            data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
            raster_plot(ax, data, linewidth=0.3)
            if d_name == 'Hidden':
                ax.set_ylabel('Spikes', fontsize=8.5)
                ax.tick_params(axis='y', labelsize=7)
            ax.set_xticklabels([])

            if d_name == 'Input':
                ax.set_yticklabels([])
                n_channel = data.shape[1] // n_input_symbols
                ax.add_patch(  # Value 0 row
                    patches.Rectangle((0, 0), data.shape[0], n_channel, facecolor="red", alpha=0.15))
                ax.add_patch(  # Value 1 row
                    patches.Rectangle((0, n_channel), data.shape[0], n_channel, facecolor="blue", alpha=0.15))
                ax.add_patch(  # Store row
                    patches.Rectangle((0, 2 * n_channel), data.shape[0], n_channel, facecolor="yellow", alpha=0.15))
                ax.add_patch(  # Recall row
                    patches.Rectangle((0, 3 * n_channel), data.shape[0], n_channel, facecolor="green", alpha=0.15))

                top_margin = 0.10
                left_margin = -0.12
                ax.text(left_margin, 1. - top_margin, 'RECALL', transform=ax.transAxes,
                        fontsize=8, verticalalignment='top')
                ax.text(left_margin, 0.75 - top_margin, 'STORE', transform=ax.transAxes,
                        fontsize=8, verticalalignment='top')
                ax.text(left_margin, 0.5 - top_margin, 'Value 1', transform=ax.transAxes,
                        fontsize=8, verticalalignment='top')
                ax.text(left_margin, 0.25 - top_margin, 'Value 0', transform=ax.transAxes,
                        fontsize=8, verticalalignment='top')

    # plot membrane potential
    ax = ax_list[2]
    ax.set_xticklabels([])
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('Membrane potential', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    sub_data = plot_result_values['mem potential'][batch, :, 0]
    presentation_steps = np.arange(sub_data.shape[0])
    ax.plot(presentation_steps, sub_data, color='blue', label='Voltage', alpha=0.7, linewidth=1.2)
    # ax.axis([0, presentation_steps[-1] + 1, np.min(sub_data, axis=-1),
    #          np.max(sub_data, axis=-1)])  # [xmin, xmax, ymin, ymax]
    ax.axis([0, presentation_steps[-1], ax.get_ylim()[0], ax.get_ylim()[1]])

    # plot adaptive threshold
    ax = ax_list[3]
    ax.set_xticklabels([])
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('Threshold')
    ax.tick_params(axis='y', labelsize=7)
    sub_data = plot_result_values['threshold'][batch, :, 0]
    presentation_steps = np.arange(sub_data.shape[0])
    ax.plot(presentation_steps, sub_data, color='red', label='Threshold', alpha=0.7, linewidth=1.2)
    # ax.axis([0, presentation_steps[-1] + 1, np.min(sub_data, axis=-1),
    #          np.max(sub_data, axis=-1)])  # [xmin, xmax, ymin, ymax]
    ax.axis([0, presentation_steps[-1], ax.get_ylim()[0], ax.get_ylim()[1]])

    # plot targets
    ax = ax_list[-1]
    mask = plot_result_values['recall_charac_mask'][batch]
    data = plot_result_values['target_nums'][batch]
    data[np.invert(mask)] = -1
    lines = []
    ind_nt = np.argwhere(data != -1)
    for idx in ind_nt.tolist():
        i = idx[0]
        lines.append([(i * FLAGS.tau_char, data[i]), ((i + 1) * FLAGS.tau_char, data[i])])
    lc_t = mc.LineCollection(lines, colors='green', linewidths=2, label='Target')
    ax.add_collection(lc_t)  # plot target segments

    # plot output per tau_char
    data = plot_result_values['out_plot_char_step'][batch]
    data = np.array([(d[1] - d[0] + 1) / 2 for d in data])
    data[np.invert(mask)] = -1
    lines = []
    ind_nt = np.argwhere(data != -1)
    for idx in ind_nt.tolist():
        i = idx[0]
        lines.append([(i * FLAGS.tau_char, data[i]), ((i + 1) * FLAGS.tau_char, data[i])])
    lc_o = mc.LineCollection(lines, colors='blue', linewidths=2, label='Output')
    ax.add_collection(lc_o)  # plot target segments

    # plot softmax of psp-s per dt for more intuitive monitoring
    # ploting only for second class since this is more intuitive to follow (first class is just a mirror)
    output2 = plot_result_values['out_plot'][batch, :, 1]
    presentation_steps = np.arange(output2.shape[0])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=7)
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('Output')
    line_output2, = ax.plot(presentation_steps, output2, color='purple', label='softmax', alpha=0.7)
    ax.axis([0, presentation_steps[-1], -0.3, 1.1])
    ax.legend(handles=[lc_t, lc_o, line_output2], loc='lower center', fontsize=7,
              bbox_to_anchor=(0.5, -0.05), ncol=3)
    my_ticks = list(range(0, 1400, 200))
    ax.set_xticks(my_ticks)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    # ax.set_xlabel([0, 400, 800, 1200])
    # ax.set_xticklabels([])

    # # debug plot for psp-s or biases
    # ax.set_xticklabels([])
    # ax = ax_list[-1]
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    # ax.set_ylabel('Threshold')
    # sub_data = plot_result_values['thr'][batch]
    # vars = np.var(sub_data, axis=0)
    # # cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses * 3:3]
    # cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses]
    # presentation_steps = np.arange(sub_data.shape[0])
    # ax.plot(sub_data[:, cell_with_max_var], color='r', label='Output', alpha=0.4, linewidth=1)
    # ax.axis([0, presentation_steps[-1], np.min(sub_data[:, cell_with_max_var]),
    #          np.max(sub_data[:, cell_with_max_var])])  # [xmin, xmax, ymin, ymax]

    ax.set_xlabel('Time in ms')
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.do_plot:
        plt.draw()
        plt.pause(1)


test_loss_list = []
test_loss_with_reg_list = []
validation_error_list = []
tau_delay_list = []
training_time_list = []
time_to_ref_list = []



def placeholder_for_init_state(batch_size):
    hidden_state = model.init_state(batch_size)  # hidden, cell
    return hidden_state



init_state_val = placeholder_for_init_state(FLAGS.batch_val)
init_state_train = placeholder_for_init_state(FLAGS.batch_train)
init_state_test = placeholder_for_init_state(FLAGS.batch_test)

plot_results_values = {}
global save_iter

date = datetime.now()
# Training loop
t_train = 0
start_time = time()
for k_iter in range(FLAGS.n_iter):
    # global optimizer
    # if k_iter == 0:
    #     init_state_val = placeholder_for_init_state(FLAGS.batch_val)
    #     init_state_train = placeholder_for_init_state(FLAGS.batch_train)

    if k_iter > 0 and np.mod(k_iter, FLAGS.lr_decay_every) == 0:
        old_lr = learning_rate
        learning_rate *= 0.8  # Adjust the decay factor as needed
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate)  # Update the optimizer with new learning rate
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, learning_rate))

    # Monitor the training with a validation set
    t_ref = time()

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_dict = get_data_dict(FLAGS.batch_val)
        val_dict = convert_data_dict(val_dict)
        output, final_state, fire_rate, hidden_spike, hidden_mem, hidden_theta = model(val_dict['input_spikes'],
                                                                                       init_state_val)
        # init_state_val = final_state
        results_values = criterion(output, val_dict['target_nums'], val_dict['input_nums'], fire_rate)

        out = output.detach().clone().cpu()
        out_char_step = torch_downsample(out, new_size=FLAGS.seq_len, axis=1)
        out_plot = F.softmax(out, dim=2)
        out_plot_char_step = torch_downsample(out_plot, new_size=FLAGS.seq_len, axis=1)
        _, recall_errors, false_sentence_id_list = error_rate(out_char_step, val_dict['target_nums'],
                                                              val_dict['input_nums'], n_charac)

        n_recall_symbol = n_charac + 1
        recall_charac_mask = torch.eq(val_dict['input_nums'], n_recall_symbol)

        plot_results_values['out_plot'] = out_plot.numpy()
        plot_results_values['out_plot_char_step'] = out_plot_char_step.numpy()
        plot_results_values['input_spikes'] = val_dict['input_spikes'].detach().clone().cpu().numpy()
        plot_results_values['z'] = hidden_spike.detach().clone().cpu().numpy()
        plot_results_values['target_nums'] = val_dict['target_nums'].numpy()
        plot_results_values['input_nums'] = val_dict['input_nums'].numpy()
        plot_results_values['recall_charac_mask'] = recall_charac_mask.numpy()
        plot_results_values['mem potential'] = hidden_mem.numpy()
        plot_results_values['threshold'] = hidden_theta.numpy()

    t_run = time() - t_ref

    # Store results
    test_loss_with_reg_list.append(results_values['loss_reg'].item())
    test_loss_list.append(results_values['loss'].item())
    validation_error_list.append(recall_errors.numpy())
    training_time_list.append(t_train)
    time_to_ref_list.append(time() - t_ref)

    if np.mod(k_iter, print_every) == 0:
        print('''Iteration {}, statistics on the validation set average error {:.2g} +- {:.2g} (trial averaged)'''
              .format(k_iter, np.mean(validation_error_list[-print_every:]),
                      np.std(validation_error_list[-print_every:])))

        if FLAGS.do_plot and FLAGS.monitor_plot:
            update_plot(plot_results_values)
            dt_str = date.strftime("%Y-%m-%d %H:%M:%S")
            datetime_obj = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            tmp_path = os.path.join(result_folder,
                                    'tmp/figure' + datetime_obj.strftime("%H%M") + '_' +
                                    str(k_iter) + '.pdf')
            if not os.path.exists(os.path.join(result_folder, 'tmp')):
                os.makedirs(os.path.join(result_folder, 'tmp'))
            fig.savefig(tmp_path, format='pdf')

        if np.mean(validation_error_list[-print_every:]) < FLAGS.stop_crit:
            print('LESS THAN ' + str(FLAGS.stop_crit) + ' ERROR ACHIEVED - STOPPING - SOLVED at epoch ' + str(k_iter))
            save_iter = k_iter
            break

    # Train
    model.train()  # Set model back to training mode
    train_dict = get_data_dict(FLAGS.batch_train)
    train_dict = convert_data_dict(train_dict)
    t_ref = time()
    optimizer.zero_grad()
    # Forward pass and backward pass
    output, final_state, fire_rate, hidden_spike, hidden_mem, hidden_theta = model(train_dict['input_spikes'],
                                                                                   init_state_train)
    # init_state_train = final_state
    results_values = criterion(output, train_dict['target_nums'], train_dict['input_nums'], fire_rate)
    loss = results_values['loss'].requires_grad_(True)
    loss.backward()  # retain_graph=True
    optimizer.step()
    t_train = time() - t_ref

print('FINISHED IN {:.2g} s'.format(time() - start_time))




# Save everything
if FLAGS.save_data:

    # Saving setup
    full_path = os.path.join(result_folder, file_reference)
    if not os.path.exists(full_path):
        os.makedirs(full_path)


    results = {
        'error': validation_error_list[-1],
        'loss': test_loss_list[-1],
        'loss_with_reg': test_loss_with_reg_list[-1],
        'loss_with_reg_list': test_loss_with_reg_list,
        'error_list': validation_error_list,
        'loss_list': test_loss_list,
        'time_to_ref': time_to_ref_list,
        'training_time': training_time_list,
        'tau_delay_list': tau_delay_list,
        'learning_rate': FLAGS.learning_rate,
        'batch_size': FLAGS.batch_train,
    }

    # save_file(flag_dict, full_path, 'flag', file_type='json')
    save_file(results, full_path, 'training_results', file_type='json')

    # Save sample trajectory (input, output, etc. for plotting)
    test_errors = []
    # init_state_test = init_state_val
    for i in range(16):
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            test_dict = get_data_dict(FLAGS.batch_test)
            test_dict = convert_data_dict(test_dict)
            output, final_state, fire_rate, hidden_spike, hidden_mem, hidden_theta = model(test_dict['input_spikes'],
                                                                                           init_state_test)
            results_values = criterion(output, test_dict['target_nums'], test_dict['input_nums'], fire_rate)
            # init_state_test = final_state

            out = output.detach().clone().cpu()
            out_char_step = torch_downsample(out, new_size=FLAGS.seq_len, axis=1)
            out_plot = F.softmax(out, dim=2)
            out_plot_char_step = torch_downsample(out_plot, new_size=FLAGS.seq_len, axis=1)
            _, recall_errors, false_sentence_id_list = error_rate(out_char_step, test_dict['target_nums'],
                                                                  test_dict['input_nums'], n_charac)

            n_recall_symbol = n_charac + 1
            recall_charac_mask = torch.eq(test_dict['input_nums'], n_recall_symbol)

            plot_results_values['out_plot'] = out_plot.numpy()
            plot_results_values['out_plot_char_step'] = out_plot_char_step.numpy()
            plot_results_values['input_spikes'] = test_dict['input_spikes'].detach().clone().cpu().numpy()
            plot_results_values['z'] = hidden_spike.detach().clone().cpu().numpy()
            plot_results_values['target_nums'] = test_dict['target_nums'].numpy()
            plot_results_values['input_nums'] = test_dict['input_nums'].numpy()
            plot_results_values['recall_charac_mask'] = recall_charac_mask.numpy()
            plot_results_values['mem potential'] = hidden_mem.numpy()
            plot_results_values['threshold'] = hidden_theta.numpy()



        test_errors.append(recall_errors.numpy())

    print('''Statistics on the test set average error {:.2g} +- {:.2g} (averaged over 16 test batches of size {})'''
          .format(np.mean(test_errors), np.std(test_errors), FLAGS.batch_test))
    save_file(plot_results_values, full_path, 'plot_trajectory_data', 'pickle')

    # Save test results
    results = {
        'test_errors': test_errors,
        'test_errors_mean': np.mean(test_errors),
        'test_errors_std': np.std(test_errors),
    }
    save_file(results, full_path, 'test_results', file_type='json')
    print("saved test_results.json")
    # Save network variables (weights, delays, etc.)

    state = {
        # 'iter': save_iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_errors': test_errors,
    }
    torch.save(state, full_path + '/model_state_dict.pt')
    print("saved model_state_dict.pt")

    # network_data = tf_cell_to_savable_dict(cell, sess)
    # network_data['w_out'] = results_values['w_out']
    # save_file(network_data, full_path, 'tf_cell_net_data', file_type='pickle')


