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
This is the script used to run experiments.
"""
import argparse
import logging
import os
#os.environ['CUDA_VISIBLE_DEVICES']='3'
from exp_spk import Experiment
from sparch.parsers.model_config import add_model_options
from sparch.parsers.training_config import add_training_options

logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Model training on spiking speech commands datasets."
    )
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()
    #args = parser.parse_args(args=['--dataset_name', 'vox1', '--data_folder', '/home/zeyang/data/VoxCeleb1', '--nb_epochs', '200', '--nb_hiddens', '512', '--nb_layers', '3', '--batch_size', '256', '--frontend', 'fbank', '--model_type', 'PLIF', '--exp_name', 'debug', '--log_tofile', 'True', '--use_augm', 'False', '--scheduler_type', 'ReduceLROnPlateau','--nb_inputs', '40', '--lr', '1e-3'])


    # args = parser.parse_args(args=['--dataset_name', 'vox1', '--data_folder', '/home/zeyang/data/VoxCeleb1', '--nb_epochs', '200', '--nb_hiddens', '512', '--batch_size', '2048', '--frontend', 'fbank', '--model_type', 'RhyPLIF', '--log_tofile', 'False', '--use_pretrained_model', 'True', "--only_do_testing", 'True','--load_exp_folder', '/home/zeyang/Project/Spiking-LEAF/exp/spk_id_Rhy_exps/vox1_LIF_3lay512_drop0_1_batchnorm_nobias_udir_noreg_lr0_001_fbank_LIF_Rhy_DCmin0_55'])

    # args = parser.parse_args(args=['--dataset_name', 'sc', '--data_folder', '/home/zeyang/data/GSCv2_official', '--nb_epochs', '200', '--nb_hiddens', '512', '--nb_layers', '3', '--batch_size', '128', '--frontend', 'Spiking_fbank', '--model_type', 'adLIF', '--exp_name', 'debug', '--log_tofile', 'True', '--use_augm', 'False', '--scheduler_type', 'ReduceLROnPlateau','--nb_inputs', '40', '--lr', '1e-3', '--normalization', 'batchnorm'])
    return args


def main():
    """
    Runs model training/testing using the configuration specified
    by the parser arguments. Run `python run_exp.py -h` for details.
    """

    # Get experiment configuration from parser
    args = parse_args()

    # Instantiate class for the desired experiment
    experiment = Experiment(args)

    # Run experiment
    experiment.forward()


if __name__ == "__main__":
    main()
