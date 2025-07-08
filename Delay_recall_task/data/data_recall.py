import torch
import numpy as np
from torch.autograd import Variable


# def data_generator(T, mem_length, b_size, encode=False):
#     """
#     Generate data for the delayed recall task
#     :param T: The delay length
#     :param mem_length: The length of the input sequences to be recalled
#     :param b_size: The batch size
#     :return: Input and target data tensor
#     """
#     seq = torch.from_numpy(np.random.randint(1, 9, size=(b_size, mem_length))).float()
#     zeros = torch.zeros((b_size, T))
#     marker = 9 * torch.ones((b_size, mem_length + 1))
#     placeholders = torch.zeros((b_size, mem_length))
#
#     x = torch.cat((seq, zeros[:, :-1], marker), 1)
#     y = torch.cat((placeholders, zeros, seq), 1).long()
#     if not encode:
#         x_out, y = Variable(x), Variable(y)
#     else:
#         one_hot=torch.eye(10)[x.long(),]
#         x_out,y =Variable(one_hot[:,:,1:].transpose(1,2)), Variable(y)
#     return x_out, y



def data_generator(T, seq_length, b_size, encode=False):
    """
    Generate data for the delayed recall task
    :param T: The delay length
    :param seq_length: The length of the input sequences to be recalled
    :param b_size: The batch size
    :return: Input and target data tensor
    """

    # Generate the main sequence (seq) with shape (b_size, 10, seq_length)
    seq = torch.randint(0, 2, size=(b_size, 4, seq_length)).float()  # First 9 rows: random 0s or 1s
    last_row = torch.zeros((b_size, 1, seq_length))  # Last row: all 0s
    seq = torch.cat((seq, last_row), dim=1)  # Combine to form (b_size, 10, seq_length)

    # Create zeros tensor with the same shape as seq
    zeros = torch.zeros((b_size, 5, T))

    # Create marker tensor where only the last row is 1s
    marker = torch.zeros((b_size, 5, seq_length))
    marker[:, -1, :] = 1  # Set the last row to 1s

    placeholders = torch.zeros((b_size, 5, seq_length))

    # Concatenate tensors to form x and y as described
    x = torch.cat((seq, zeros, marker), dim=-1)
    y = torch.cat((placeholders, zeros, seq), dim=-1).float()

    return Variable(x), Variable(y)




