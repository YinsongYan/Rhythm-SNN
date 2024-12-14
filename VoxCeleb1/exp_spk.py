#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is to define the experiment class used to perform training and testing
of ANNs and SNNs on all speech command recognition datasets.
"""
import errno
import logging
import os
import time
from datetime import timedelta
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import transformers
from sparch.models.neurons.utils import *

logger = logging.getLogger(__name__)
from sparch.dataloaders.vox1_dataset import RawWaveformDataset as SpectrogramDataset
from sparch.dataloaders.utils_leaf.utils import _collate_fn_raw_multiclass, setup_dataloaders, setup_testloader
from sparch.dataloaders.utils_leaf.raw_transform import leaf_supervised_transforms, leaf_supervised_transforms_test
from sparch.models.anns import ANN
# from sparch.models.snns import SNN
# from sparch.models.snns_baseline import SNN #copy from codebase
from sparch.models.snns_Rhy import SNN #Rhy
from sparch.parsers.model_config import print_model_options
from sparch.parsers.training_config import print_training_options
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
# torch.autograd.set_detect_anomaly(True) 

class Experiment:
    """
    Class for training and testing models (ANNs and SNNs) on all four
    datasets for speech command recognition (shd, ssc, hd and sc).
    """

    def __init__(self, args):
        self.args=args
        # New model config
        self.model_type = args.model_type
        self.nb_layers = args.nb_layers
        self.nb_hiddens = args.nb_hiddens
        self.pdrop = args.pdrop
        self.normalization = args.normalization
        self.use_bias = args.use_bias
        self.bidirectional = args.bidirectional

        # Training config
        self.use_pretrained_model = args.use_pretrained_model
        self.only_do_testing = args.only_do_testing
        self.load_exp_folder = args.load_exp_folder
        self.new_exp_folder = args.new_exp_folder
        self.dataset_name = args.dataset_name
        self.data_folder = args.data_folder
        self.log_tofile = args.log_tofile
        self.save_best = args.save_best
        self.batch_size = args.batch_size
        self.nb_epochs = args.nb_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_factor = args.scheduler_factor
        self.use_regularizers = args.use_regularizers
        self.reg_factor = args.reg_factor
        self.reg_fmin = args.reg_fmin
        self.reg_fmax = args.reg_fmax
        self.use_augm = args.use_augm
        self.exp_name = args.exp_name
        self.frontend = args.frontend
        self.stu_enc = args.stu_enc
        self.scheduler_type = args.scheduler_type
        self.nb_inputs = args.nb_inputs
        #args for noised evaluation
        self.noise = args.noise
        self.noise_condition = args.noise_condition
        self.SNR = args.SNR
        #args for spike rate regulation
        self.reg_frontend = args.reg_frontend
        self.fmax_frontend = args.fmax_frontend

        # Initialize logging and output folders
        self.init_exp_folders()
        self.init_logging()
        print_model_options(args)
        print_training_options(args)
        
        self.writer = SummaryWriter(comment="spkid_" + self.frontend + "_" + self.exp_name)
        self.writer.add_text('Command-line Arguments', str(args))
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"\nDevice is set to {self.device}\n")

        # Initialize dataloaders and model
        self.init_dataset()
        self.init_model(args=args)

        # Define optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), self.lr)

        # Define learning rate scheduler
        if args.scheduler_type=='ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                optimizer=self.opt,
                mode="max",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=1e-6,
            )
        elif args.scheduler_type=='StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10, gamma=self.scheduler_factor)
        elif args.scheduler_type=='warmup_cosine':
            num_tr_steps_per_epoch = len(self.train_loader)
            total_tr_steps = num_tr_steps_per_epoch * self.nb_epochs
            warmup_steps = num_tr_steps_per_epoch * 10
            self.scheduler = transformers.get_cosine_schedule_with_warmup(self.opt, warmup_steps, total_tr_steps)
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self):
        """
        This function performs model training with the configuration
        specified by the class initialization.
        """
        if not self.only_do_testing:

            # Initialize best accuracy
            if self.use_pretrained_model:
                logging.info("\n------ Using pretrained model ------\n")
                best_epoch, best_acc = self.valid_one_epoch(self.start_epoch, 0, 0)
            else:
                best_epoch, best_acc = 0, 0

            # Loop over epochs (training + validation)
            logging.info("\n------ Begin training ------\n")

            for e in range(best_epoch + 1, best_epoch + self.nb_epochs + 1):
                self.train_one_epoch(e, self.frontend)
                best_epoch, best_acc = self.valid_one_epoch(e, best_epoch, best_acc, self.frontend)

            logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
            logging.info("\n------ Training finished ------\n")

            # Loading best model
            if self.save_best:
                # self.net = torch.load(
                #     f"{self.checkpoint_dir}/best_model.pth", map_location=self.device
                # )['net']
                self.net.load_state_dict(torch.load(f"{self.checkpoint_dir}/best_model.pth"))
                logging.info(
                    f"Loading best model, epoch={best_epoch}, valid acc={best_acc}"
                )
            else:
                logging.info(
                    "Cannot load best model because save_best option is "
                    "disabled. Model from last epoch is used for testing."
                )

        # Test trained model
        self.test_one_epoch(self.test_loader, self.frontend)
        print(f"Loading best model, epoch={best_epoch}, valid acc={best_acc}")
    def init_exp_folders(self):
        """
        This function defines the output folders for the experiment.
        """
        # Check if path exists for loading pretrained model
        if self.use_pretrained_model:
            exp_folder = self.load_exp_folder
            self.load_path = exp_folder + "/checkpoints/best_model.pth"
            if not os.path.exists(self.load_path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), self.load_path
                )

        # Use given path for new model folder
        elif self.new_exp_folder is not None:
            exp_folder = self.new_exp_folder

        # Generate a path for new model from chosen config
        else:
            outname = self.dataset_name + "_" + self.model_type + "_"
            outname += str(self.nb_layers) + "lay" + str(self.nb_hiddens)
            outname += "_drop" + str(self.pdrop) + "_" + str(self.normalization)
            outname += "_bias" if self.use_bias else "_nobias"
            outname += "_bdir" if self.bidirectional else "_udir"
            outname += "_reg" if self.use_regularizers else "_noreg"
            outname += "_lr" + str(self.lr)
            outname += "_" + self.frontend
            outname += "_" + self.exp_name
            if self.noise:
                exp_folder = "exp/noise_exps/" + outname.replace(".", "_")
            else:
                exp_folder = "exp/spk_id_Rhy_exps/" + outname.replace(".", "_")

        # For a new model check that out path does not exist
        if not self.use_pretrained_model and os.path.exists(exp_folder):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)

        # Create folders to store experiment
        self.log_dir = exp_folder + "/log/"
        self.checkpoint_dir = exp_folder + "/checkpoints/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.exp_folder = exp_folder

    def init_logging(self):
        """
        This function sets the experimental log to be written either to
        a dedicated log file, or to the terminal.
        """
        if self.log_tofile:
            logging.FileHandler(
                filename=self.log_dir + "exp.log",
                mode="a",
                encoding=None,
                delay=False,
            )
            logging.basicConfig(
                filename=self.log_dir + "exp.log",
                level=logging.INFO,
                format="%(message)s",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
            )

    def init_dataset(self):
        self.nb_outputs = 1251
        audio_config = {'feature': 'raw', 'normalize': False, 'sample_rate': 16000, 'min_duration': 1, 'random_clip_size': 1, 'val_clip_size': 1, 'mixup': False}
        labels_delimiter = ','
        mode = 'multiclass'
        sample_rate = 16000
        random_clip_size = 16000
        val_clip_size = 16000
        meta_dir = '/home/zeyang/Project/leaf-pytorch/voxceleb1_meta'
        tr_tfs = leaf_supervised_transforms(True, random_clip_size,
                                              sample_rate=sample_rate)
        val_tfs = leaf_supervised_transforms(False, val_clip_size,
                                               sample_rate=sample_rate)
        test_tfs = leaf_supervised_transforms_test(sample_rate=sample_rate)

        train_set = SpectrogramDataset(os.path.join(meta_dir, "train.csv"),
                                       os.path.join(meta_dir, "lbl_map.json"),
                                       audio_config,
                                       mode=mode, augment=True,
                                       mixer=None, delimiter=labels_delimiter,
                                       transform=tr_tfs, is_val=False, cropped_read=False)

        val_set = SpectrogramDataset(os.path.join(meta_dir, "val.csv"),
                                    os.path.join(meta_dir, "lbl_map.json"),
                                     audio_config,
                                     mode=mode, augment=False,
                                     mixer=None, delimiter=labels_delimiter,
                                     transform=val_tfs, is_val=True)
        test_set = SpectrogramDataset(os.path.join(meta_dir, "test.csv"),
                                 os.path.join(meta_dir, "lbl_map.json"),
                                 audio_config,
                                mode=mode,
                                 transform=test_tfs, is_val=True, delimiter=labels_delimiter
                                 )
        collate_fn = _collate_fn_raw_multiclass

        train_loader, val_loader = setup_dataloaders(train_set, val_set,
                                                 batch_size=self.batch_size, collate_fn=collate_fn,
                                                 num_workers=4)
        test_loader = setup_testloader(test_set, collate_fn)
        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.test_loader = test_loader


    def init_model(self, args):
        """
        This function either loads pretrained model or builds a
        new model (ANN or SNN) depending on chosen config.
        """
        input_shape = (self.batch_size, None, self.nb_inputs)
        layer_sizes = [self.nb_hiddens] * (self.nb_layers - 1) + [self.nb_outputs]

        if self.use_pretrained_model:
            self.net = SNN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                neuron_type=self.model_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
                frontend=self.frontend,
                args=args
            ).to(self.device)
            self.net.load_state_dict(torch.load(self.load_path, map_location=self.device))
            logging.info(f"\nLoaded model at: {self.load_path}\n {self.net}\n")

        elif self.model_type in ["LIF", "PLIF", "adLIF", "RhyLIF", "RhyPLIF", "RhyadLIF"]:

            self.net = SNN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                neuron_type=self.model_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
                frontend=self.frontend,
                args=args
            ).to(self.device)

            logging.info(f"\nCreated new spiking model:\n {self.net}\n")

        elif self.model_type in ["MLP", "RNN", "LiGRU", "GRU"]:

            self.net = ANN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                ann_type=self.model_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
            ).to(self.device)

            logging.info(f"\nCreated new non-spiking model:\n {self.net}\n")

        else:
            raise ValueError(f"Invalid model type {self.model_type}")
        self.net_forward = torch.nn.DataParallel(self.net)
        self.net = self.net_forward.module
        self.nb_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        logging.info(f"Total number of trainable parameters is {self.nb_params}")

    def train_one_epoch(self, e, frontend):
        """
        This function trains the model with a single pass over the
        training split of the dataset.
        """
        start = time.time()
        self.net_forward.train()
        losses, accs = [], []
        epoch_frontend_spike_rate = 0
        epoch_spike_rate = 0

        # Loop over batches from train set
        for step, (raw_audio, feats, y) in enumerate(tqdm(self.train_loader)):

            # Dataloader uses cpu to allow pin memory
            raw_audio = raw_audio.to(self.device)
            feats = feats.to(self.device)
            y = y.to(self.device)

            reset_states(model=self.net)
            # Forward pass through network
            if frontend == "fbank" or frontend == "Spiking_fbank":
                output, firing_rates, enc_output = self.net_forward(feats)
            else:
                output, firing_rates, enc_output = self.net_forward(raw_audio.squeeze(1))


            # Compute loss
            # loss_val = self.loss_fn(output, y)
            # losses.append(loss_val.item())
            #additional frontedn spike rate loss
            loss_SR = F.relu(enc_output.mean() - self.fmax_frontend).sum()
            loss_val = self.loss_fn(output, y) + self.reg_frontend * loss_SR
            losses.append(loss_val.item())

            # Spike activity
            if self.net.is_snn:
                epoch_spike_rate += torch.mean(firing_rates)
                epoch_frontend_spike_rate += torch.mean(enc_output)

                # if self.use_regularizers:
                #     reg_quiet = F.relu(self.reg_fmin - firing_rates).sum()
                #     reg_burst = F.relu(firing_rates - self.reg_fmax).sum()
                #     loss_val += self.reg_factor * (reg_quiet + reg_burst)

            # Backpropagate
            self.opt.zero_grad()
            # print(f'e: {e}, step: {step}')s
            loss_val.backward()

            # for name, parameter in self.net.named_parameters():
            #     if parameter.grad is not None:
            #         self.writer.add_scalar(f'Gradients/{name}', parameter.grad.data.norm(), global_step=e*len(self.train_loader) + step)
            self.opt.step()


            # Compute accuracy with labels
            pred = torch.argmax(output, dim=1)
            acc = np.mean((y == pred).detach().cpu().numpy())
            accs.append(acc)
            if self.scheduler_type=='warmup_cosine':
                self.scheduler.step()
        

        # Learning rate of whole epoch
        current_lr = self.opt.param_groups[-1]["lr"]
        logging.info(f"Epoch {e}: lr={current_lr}")
        self.writer.add_scalar('lr', current_lr, e)

        # Train loss of whole epoch
        train_loss = np.mean(losses)
        logging.info(f"Epoch {e}: train loss={train_loss}")
        self.writer.add_scalar('Train/loss', train_loss, e)

        # Train accuracy of whole epoch
        train_acc = np.mean(accs)
        logging.info(f"Epoch {e}: train acc={train_acc}")
        self.writer.add_scalar('Train/acc', train_acc, e)

        # Train spike activity of whole epoch
        if self.net.is_snn:
            epoch_spike_rate /= step
            logging.info(f"Epoch {e}: train mean act rate={epoch_spike_rate}")
            self.writer.add_scalar('Train/Spike_rate', epoch_spike_rate, e)
            epoch_frontend_spike_rate /= step
            logging.info(f"Epoch {e}: train mean act rate={epoch_frontend_spike_rate}")
            self.writer.add_scalar('Train/Spike_rate_frontend', epoch_frontend_spike_rate, e)

        if self.scheduler_type=='StepLR':
            self.scheduler.step()
        # elif self.scheduler_type=='ReduceLROnPlateau':
        #     self.scheduler.step(train_acc)

        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        logging.info(f"Epoch {e}: train elapsed time={elapsed}")


    def valid_one_epoch(self, e, best_epoch, best_acc, frontend):
        """
        This function tests the model with a single pass over the
        validation split of the dataset.
        """
        with torch.no_grad():

            self.net.eval()
            losses, accs = [], []
            epoch_spike_rate = 0
            epoch_frontend_spike_rate = 0

            # Loop over batches from validation set
            for step, (raw_audio, feats, y) in enumerate(tqdm(self.valid_loader)):

                # Dataloader uses cpu to allow pin memory
                raw_audio = raw_audio.to(self.device)
                feats = feats.to(self.device)
                y = y.to(self.device)
                reset_states(model=self.net)
                # Forward pass through network
                if frontend == "fbank" or frontend == "Spiking_fbank":
                    output, firing_rates, enc_output = self.net_forward(feats)
                else:
                    output, firing_rates, enc_output = self.net_forward(raw_audio.squeeze(1))

                # Compute loss
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                pred = torch.argmax(output, dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
                accs.append(acc)

                # Spike activity
                if self.net.is_snn:
                    epoch_spike_rate += torch.mean(firing_rates)
                    epoch_frontend_spike_rate += torch.mean(enc_output)

            # Validation loss of whole epoch
            valid_loss = np.mean(losses)
            logging.info(f"Epoch {e}: valid loss={valid_loss}")
            self.writer.add_scalar('Valid/loss', valid_loss, e)

            # Validation accuracy of whole epoch
            valid_acc = np.mean(accs)
            logging.info(f"Epoch {e}: valid acc={valid_acc}")
            self.writer.add_scalar('Valid/acc', valid_acc, e)

            # Validation spike activity of whole epoch
            if self.net.is_snn:
                epoch_spike_rate /= step
                logging.info(f"Epoch {e}: valid mean act rate={epoch_spike_rate}")
                self.writer.add_scalar('Valid/Spike_rate', epoch_spike_rate, e)
                epoch_frontend_spike_rate /= step
                logging.info(f"Epoch {e}: train mean act rate={epoch_frontend_spike_rate}")
                self.writer.add_scalar('Valid/Spike_rate_frontend', epoch_frontend_spike_rate, e)

            # Update learning rate
            if self.scheduler_type=='ReduceLROnPlateau':
                self.scheduler.step(valid_acc)
                
            # torch.save(self.net.state_dict(), f"{self.checkpoint_dir}/model_epoch{e}.pth")

            # Update best epoch and accuracy
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = e
                # Save best model
                if self.save_best:
                    torch.save(self.net.state_dict(), f"{self.checkpoint_dir}/best_model.pth")
                    with open(f"{self.checkpoint_dir}/args.json", 'w') as f:
                        json.dump(vars(self.args), f)
                    logging.info(f"\nBest model saved with valid acc={valid_acc}")

            logging.info("\n-----------------------------\n")

            return best_epoch, best_acc

    def test_one_epoch(self, test_loader, frontend):
        """
        This function tests the model with a single pass over the
        testing split of the dataset.
        """
        with torch.no_grad():

            self.net_forward.eval()
            losses, accs = [], []
            epoch_spike_rate = []
            epoch_frontend_spike_rate = 0

            logging.info("\n------ Begin Testing ------\n")

            # Loop over batches from test set
            for step, (raw_audio, feats, y) in enumerate(tqdm(test_loader)):

                # Dataloader uses cpu to allow pin memory
                raw_audio = raw_audio.to(self.device)
                feats = feats.to(self.device)
                y = y.to(self.device)
                
                reset_states(model=self.net)
                # Forward pass through network
                if frontend == "fbank" or frontend == "Spiking_fbank":
                    output, firing_rates, enc_output = self.net_forward(feats)
                else:
                    output, firing_rates, enc_output = self.net_forward(raw_audio.squeeze(1))

                # Compute loss
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                pred = torch.argmax(output, dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
                accs.append(acc)

                # Spike activity
                if self.net.is_snn:
                    # epoch_spike_rate += torch.mean(firing_rates)
                    epoch_spike_rate.append(firing_rates.unsqueeze(0))
                    epoch_frontend_spike_rate += torch.mean(enc_output)

            # Test loss
            test_loss = np.mean(losses)
            logging.info(f"Test loss={test_loss}")

            # Test accuracy
            test_acc = np.mean(accs)
            print(f"Test acc={test_acc}")
            logging.info(f"Test acc={test_acc}")
            self.writer.add_scalar('Test/acc', test_acc)

            # Test spike activity
            if self.net.is_snn:
                # epoch_spike_rate /= step
                ave_FR = torch.cat(epoch_spike_rate, dim=0).mean(dim=0)
                ave_1 = ave_FR[:self.nb_hiddens].mean()
                ave_2 = ave_FR[self.nb_hiddens:].mean()
                logging.info(f"Test mean act rate layer 1={ave_1}")
                self.writer.add_scalar('Test mean act rate layer 1', ave_1)
                logging.info(f"Test mean act rate layer 2={ave_2}")
                self.writer.add_scalar('Test mean act rate layer 2', ave_2)
                # logging.info(f"train mean act rate={epoch_frontend_spike_rate/step}")
                # self.writer.add_scalar('Test/Spike_rate_frontend', epoch_frontend_spike_rate/step)

            logging.info("\n-----------------------------\n")
