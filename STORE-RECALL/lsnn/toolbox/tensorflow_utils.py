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

# import tensorflow as tf
import torch
import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def reduce_variance(v, axis=None):
    m = torch.mean(v, dim=axis)
    if axis is not None:
        m = m.unsqueeze(dim=axis)

    return torch.mean((v - m)**2, dim=axis)


def boolean_count(var, axis=-1):
    v = var.to(torch.int32)
    return torch.sum(v, dim=axis)


# def variable_summaries(var,name=''):
#   """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#   with tf.name_scope(name + 'Summary'):
#     mean = tf.reduce_mean(var)
#     tf.summary.scalar('mean', mean)
#     with tf.name_scope('stddev'):
#       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#     tf.summary.scalar('stddev', stddev)
#     tf.summary.scalar('max', tf.reduce_max(var))
#     tf.summary.scalar('min', tf.reduce_min(var))
#     tf.summary.histogram('histogram', var)

def torch_repeat(tensor, num, axis):
    dims = tensor.dim()
    dtype = tensor.dtype
    assert dtype in [torch.float32, torch.float16, torch.float64, torch.int32, torch.int64, torch.bool], 'Data type not understood: ' + str(dtype)

    # Generate a new dimension with the specified size
    tensor = tensor.unsqueeze(dim=dims)
    exp = torch.ones(tuple([1] * dims) + (num,), dtype=dtype, device=tensor.device)
    tensor_exp = tensor * exp

    # Split and stack in the right dimension
    splitted = torch.chunk(tensor_exp, chunks=num, dim=axis)
    concatenated = torch.cat(splitted, dim=dims-1)

    # Permute to put back the axis where it should be
    axis_permutation = list(range(dims-1))
    axis_permutation.insert(axis, dims-1)

    transposed = concatenated.permute(*axis_permutation)

    return transposed

# def tf_repeat_test():
#
#   a = np.arange(12).reshape(3,4)
#   axis_a = 1
#   num_a = 5
#
#   b = rd.randn(3)
#   axis_b = 0
#   num_b = 4
#
#   c = rd.randn(4,5,7,3,2)
#   axis_c = 1
#   num_c = 11
#
#   sess = tf.Session()
#   for tensor, axis, num in zip([a,b,c], [axis_a,axis_b,axis_c], [num_a,num_b,num_c]):
#
#     res_np = np.repeat(tensor, repeats=num, axis=axis)
#     res_tf = sess.run(tf_repeat(tf.constant(value=tensor,dtype=tf.float32),axis=axis,num=num))
#     assert np.mean((res_np - res_tf)**2) < 1e-6, 'Repeat mismatched between np and tf: \n np: {} \n tf: {}'.format(res_np,res_tf)
#
#
#   print('tf_repeat_test -> success')


def torch_downsample(tensor,new_size,axis):
    dims = tensor.dim()

    splitted = torch.chunk(tensor, chunks=new_size, dim=axis)
    stacked = torch.stack(splitted, dim=dims)
    reduced = torch.mean(stacked, dim=axis)

    permutation = list(range(dims - 1))
    permutation.insert(axis, dims - 1)
    transposed = reduced.permute(*permutation)

    return transposed




# def tf_downsample_test():
#
#     a = np.array([1,2,1,2,4,6,4,6])
#     sol_a = np.array([1.5,5.])
#     axis_a = 0
#     num_a = 2
#
#     sol_c = rd.randn(4, 5, 7, 3, 2)
#     axis_c = 1
#     num_c = 5
#     c = np.repeat(sol_c,repeats=11,axis=axis_c)
#
#     sess = tf.Session()
#
#     for t_np,axis,num,sol in zip([a,c],[axis_a,axis_c],[num_a,num_c],[sol_a,sol_c]):
#         t = tf.constant(t_np,dtype=tf.float32)
#         t_ds = tf_downsample(t,new_size=num,axis=axis)
#
#         t_ds_np = sess.run(t_ds)
#         assert np.sum((t_ds_np - sol)**2) < 1e-6, 'Failed test: mistmatch between downsampled: \n arg: {} \n output: {} \n should be: {}'.format(t_np,t_ds_np,sol)
#
#     print('tf_downsample_test -> success')



def torch_roll(buffer, new_last_element=None, axis=0):
    shp = buffer.shape
    l_shp = len(shp)

    if shp[-1] == 0:
        return buffer

    # Permute the index to roll over the right index
    perm = [axis] + list(range(axis)) + list(range(axis+1, l_shp))
    buffer = buffer.permute(*perm)

    # Add an element at the end of the buffer if requested, otherwise, add zero
    if new_last_element is None:
        shp = buffer.shape
        new_last_element = torch.zeros(shp[1:], dtype=buffer.dtype, device=buffer.device)
    new_last_element = new_last_element.unsqueeze(0)
    new_buffer = torch.cat([buffer[1:], new_last_element], dim=0)

    # Revert the index permutation
    inv_perm = torch.argsort(torch.tensor(perm))
    new_buffer = new_buffer.permute(*inv_perm)

    return new_buffer


def torch_tuple_of_placeholder(shape_named_tuple, dtype, default_named_tuple=None, name='TupleOfPlaceholder'):
    placeholder_dict = OrderedDict()

    if default_named_tuple is not None:
        default_dict = default_named_tuple._asdict()
        for k, v in default_dict.items():
            placeholder_dict[k] = torch.tensor(v, dtype=dtype)
    else:
        shape_dict = shape_named_tuple._asdict()
        for k, v in shape_dict.items():
            placeholder_dict[k] = torch.empty(v, dtype=dtype)

    placeholder_tuple = shape_named_tuple.__class__(**placeholder_dict)
    return placeholder_tuple


def torch_feeding_dict_of_placeholder_tuple(tuple_of_placeholder, tuple_of_values):
    feed_dict = {}
    for k, v in tuple_of_placeholder._asdict().items():
        feed_dict[v] = tuple_of_values._asdict()[k]

    return feed_dict


def moving_sum(tensor, n_steps):
    n_batch = tensor.shape[0]
    n_time = tensor.shape[1]
    n_neuron = tensor.shape[2]

    assert tensor.dim() == 3, 'Shape tuple for time filtering should be of length 3, found {}'.format(tensor.shape)

    t0 = torch.tensor(0, dtype=torch.int32, device=tensor.device)
    out = []
    buffer = torch.zeros((n_batch, n_steps, n_neuron), dtype=tensor.dtype, device=tensor.device)

    while t0 < n_time:
        x = tensor[:, t0, :]

        buffer = torch_roll(buffer, new_last_element=x.unsqueeze(1), axis=1)
        new_y = buffer.sum(dim=1)
        out.append(new_y)

        t0 += 1

    out = torch.stack(out)
    out = out.permute(1, 0, 2)

    return out


def exp_convolve(tensor, decay):
    assert tensor.dtype in [torch.float16, torch.float32, torch.float64]

    tensor_time_major = tensor.permute(1, 0, 2)
    num_time_steps, num_batches, num_neurons = tensor_time_major.shape

    filtered_tensor = torch.empty_like(tensor_time_major)

    prev_state = torch.zeros(num_batches, num_neurons, dtype=tensor.dtype, device=tensor.device)
    for t in range(num_time_steps):
        prev_state = prev_state * decay + (1 - decay) * tensor_time_major[t]
        filtered_tensor[t] = prev_state

    filtered_tensor = filtered_tensor.permute(1, 0, 2)

    return filtered_tensor


def discounted_return(reward, discount, axis=-1, boundary_value=0):
    l_shp = len(reward.shape)
    assert l_shp >= 1, 'Tensor must be rank 1 or higher'

    axis = np.mod(axis, l_shp)
    perm = np.arange(l_shp)

    perm[0] = axis
    perm[axis] = 0

    t = reward.permute(*perm)
    t = torch.flip(t, [0])

    initializer = torch.ones_like(t[0]) * boundary_value

    output = []
    accumulated = initializer
    for x in t:
        accumulated = accumulated * discount + x
        output.append(accumulated)

    t = torch.stack(output)
    t = torch.flip(t, [0])

    t = t.permute(*perm)
    return t


# def tf_moving_sum_test():
#     sess = tf.Session()
#
#     def moving_sum_numpy(tensor,n_steps):
#         n_batch,n_time,n_neuron = tensor.shape
#
#         def zz(d):
#             z = np.zeros(shape=(n_batch,d,n_neuron),dtype=tensor.dtype)
#             return z
#
#         stacks = [np.concatenate([zz(d),tensor[:,:n_time-d,:]],axis=1) for d in range(n_steps)]
#         stacks = np.array(stacks)
#         return np.sum(np.array(stacks),axis=0)
#
#     def assert_quantitative_error(arr1,arr2):
#
#         err = np.mean((arr1 - arr2) ** 2)
#         if err > 1e-6:
#             plt.plot(arr1[0, :, :],color='blue')
#             plt.plot(arr2[0, :, :],color='green')
#             plt.show()
#             raise ValueError('Mistmatch of the smoothing with error {}'.format(err))
#
#     # quick test
#     a = np.array([0,1,2,4,1,2]).reshape((1,6,1))
#     n_a = 2
#     sol_a = np.array([0,1,3,6,5,3]).reshape((1,6,1))
#
#     # Check the numpy function
#     summed_np = moving_sum_numpy(a,n_a)
#     assert_quantitative_error(sol_a,summed_np)
#
#
#     # Check the tf function
#     summed_tf = sess.run(moving_sum(tf.constant(a),n_a))
#     assert_quantitative_error(sol_a,summed_tf)
#
#     T = 100
#     n_neuron = 10
#     n_batch=3
#     n_delay = 5
#
#     tensor = rd.randn(n_batch,T,n_neuron)
#
#     summed_np = moving_sum_numpy(tensor,n_delay)
#     summed_tf = sess.run(moving_sum(tf.constant(tensor,dtype=tf.float32),n_delay))
#     assert_quantitative_error(summed_np,summed_tf)
#
#     print('tf_moving_sum_test -> success')
#
# def tf_exp_convolve_test():
#     sess = tf.Session()
#
#     def exp_convolve_numpy(tensor,decay):
#         n_batch,n_time,n_neuron = tensor.shape
#
#         out = np.zeros_like(tensor,dtype=float)
#         running = np.zeros_like(tensor[:,0,:],dtype=float)
#         for t in range(n_time):
#             out[:,t,:] = decay * running + (1-decay) * tensor[:,t,:]
#             running = out[:,t,:]
#
#         return out
#
#     def assert_quantitative_error(arr_np, arr_tf):
#
#         err = np.mean((arr_np - arr_tf) ** 2)
#         if err > 1e-6:
#             plt.plot(arr_np[0, :, :], color='blue', label='np')
#             plt.plot(arr_tf[0, :, :], color='green', label='tf')
#             plt.legend()
#             plt.show()
#             raise ValueError('Mistmatch of the smoothing with error {}'.format(err))
#
#     # quick test
#     a = np.array([0,1,2,4,1,2]).reshape((1,6,1))
#     decay_a = 0.5
#
#     # Check the numpy function
#     summed_np = exp_convolve_numpy(a,decay_a)
#     summed_tf = sess.run(exp_convolve(tf.constant(a,dtype=tf.float32),decay_a))
#     assert_quantitative_error(summed_np,summed_tf)
#
#     T = 100
#     n_neuron = 10
#     n_batch= 3
#     decay = .5
#
#     tensor = rd.randn(n_batch,T,n_neuron)
#
#     summed_np = exp_convolve_numpy(tensor,decay)
#     summed_tf = sess.run(exp_convolve(tf.constant(tensor,dtype=tf.float32),decay))
#     assert_quantitative_error(summed_np,summed_tf)
#
#     print('tf_exp_convolve_test -> success')
#
# def tf_discounted_reward_test():
#
#     g = .8
#
#     a = [.1, 0, 1.]
#     a_return = [.1 + g**2,g,1.]
#
#     b = np.array([[1,2,3], a])
#     b_return = np.array([[1 + 2*g + 3*g**2, 2 + 3*g,3], a_return])
#
#     c = rd.rand(3,4,2)
#     c_return = np.zeros_like(c)
#     n_b,T,n = c.shape
#     for i in range(n_b):
#         tmp = np.zeros(n)
#         for t in range(T):
#             tmp =  g * tmp + c[i,T-1-t]
#             c_return[i,T-1-t] = tmp
#
#     sess = tf.Session()
#
#     for t,t_return,axis in zip([a,b,c],[a_return,b_return,c_return],[-1,-1,1]):
#         tf_return = discounted_return(tf.constant(t,dtype=tf.float32),g,axis=axis)
#         np_return = sess.run(tf_return)
#         assert np.sum((np_return - np.array(t_return))**2) < 1e-6, 'Mismatch: \n tensor {} \n solution {} \n found {}'.format(t,t_return,np_return)
#
# if __name__ == '__main__':
#   tf_repeat_test()
#   tf_downsample_test()
#   tf_moving_sum_test()
#   tf_exp_convolve_test()
#   tf_discounted_reward_test()