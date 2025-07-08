"""
The Clear BSD License

Copyright (c) 2019 the LSNN team, institute for theoretical computer science, TU Graz
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of LSNN nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch

def einsum_bi_ijk_to_bjk(a, b):
    batch_size = a.shape[0]
    shp_a = a.shape
    shp_b = b.shape

    b_ = b.view(shp_b[0], shp_b[1] * shp_b[2])
    ab_ = torch.matmul(a, b_)
    ab = ab_.view(batch_size, shp_b[1], shp_b[2])
    return ab

def einsum_bi_bij_to_bj(a, b):
    a_ = a.unsqueeze(1)
    a_b = torch.matmul(a_, b)
    ab = a_b[:, 0, :]
    return ab

def einsum_bi_bijk_to_bjk(a, b):
    a_ = a[:, :, None, None]
    a_b = a_ * b
    return torch.sum(a_b, dim=1)

def einsum_bij_jk_to_bik(a, b):
    n_b = a.shape[0]
    n_i = a.shape[1]
    n_j = a.shape[2]
    n_k = b.shape[1]

    a_ = a.view(n_b * n_i, n_j)
    a_b = torch.matmul(a_, b)
    ab = a_b.view(n_b, n_i, n_k)
    return ab

def einsum_bij_ki_to_bkj(a, b):
    a_ = a.unsqueeze(1)
    b_ = b.unsqueeze(0).unsqueeze(3)

    ab = torch.sum(a_ * b_, dim=2)
    return ab

