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
This is where the parser for the model configuration is defined.
"""
import logging
from distutils.util import strtobool

logger = logging.getLogger(__name__)


def add_model_options(parser):
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["LIF", "PLIF", "adLIF", "RhyLIF", "RhyPLIF", "RhyadLIF", "MLP", "RNN", "LiGRU", "GRU"],
        default="adLIF",
        help="Type of ANN or SNN model.",
    )
    parser.add_argument(
        "--nb_layers",
        type=int,
        default=3,
        help="Number of layers (including readout layer).",
    )
    parser.add_argument(
        "--nb_hiddens",
        type=int,
        default=128,
        help="Number of neurons in all hidden layers.",
    )
    parser.add_argument(
        "--nb_inputs",
        type=int,
        default=40,
        help="Number of neurons in all hidden layers.",
    )
    parser.add_argument(
        "--pdrop",
        type=float,
        default=0.1,
        help="Dropout rate, must be between 0 and 1.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="batchnorm",
        help="Type of normalization, Every string different from batchnorm "
        "and layernorm will result in no normalization.",
    )
    parser.add_argument(
        "--use_bias",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--bidirectional",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="If True, a bidirectional model that scans the sequence in both "
        "directions is used, which doubles the size of feedforward matrices. ",
    )
    parser.add_argument(
        "--frontend",
        type=str,
        choices=["fbank", "CoNNear_ANN", "CoNNear_SNN", "nnAudio_MFCC", "nnAudio_fbank", "SincConv", "Gabor_spiking", "Gabor", "Spiking_fbank"],
        help="CoNNear_ANN is the real value feature, CoNNear_SNN is the spiking latent feature",
    )
    parser.add_argument(
        "--stu_enc",
        type=list,
        default=[40, 128, 40],
        help="Hidden size for encoder MLP, and the last one is the input size of clasifier model",
    )
    parser.add_argument(
        "--teacher_path",
        type=str,
        help="Path for the teacher model",
    )
    #Configurations for Gated LIF
    parser.add_argument('--gate', type=float, default=[0.6, 0.8, 0.6], nargs='+', help='initial gate')
    parser.add_argument('--static-gate', default=False, action='store_true', help='use static_gate')
    parser.add_argument('--static-param', default=False, action='store_true', help='use static_LIF_param')
    parser.add_argument('--channel-wise', default=False, action='store_true', help='use channel-wise')
    parser.add_argument('--softsimple', default=False, action='store_true', help='experiments on coarsely fused LIF')
    parser.add_argument('--soft-mode', default=False, action='store_true', help='use soft_gate')
    parser.add_argument('--randomgate', default=False, action='store_true', help='activate uniform-randomly intialized gates')

    return parser


def print_model_options(args):
    logging.info(
        """
        Model Config
        ------------
        Model Type: {model_type}
        Number of layers: {nb_layers}
        Number of hidden neurons: {nb_hiddens}
        Dropout rate: {pdrop}
        Normalization: {normalization}
        Use bias: {use_bias}
        Bidirectional: {bidirectional}
    """.format(
            **vars(args)
        )
    )
