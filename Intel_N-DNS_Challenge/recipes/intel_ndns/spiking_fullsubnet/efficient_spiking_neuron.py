import math
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import Parameter


MemoryState = namedtuple("MemoryState", ["hx", "cx"])


def efficient_spiking_neuron(
    input_size,
    hidden_size,
    num_layers,
    shared_weights=False,
    bn=False,
    batch_first=False,
):
    """
    Instantiate efficient spiking networks where each spiking neuron uses the gating mechanism to control the decay of membrane potential.
    :param input_size:
    :param hidden_size:
    :param num_layers:
    :param shared_weights: whether weights of the gate are shared with the ones of the cell or not.
    :param bn: whether batchnorm is used or not.
    :param batch_first: Not used.
    :return:
    """
    #     # The following are not implemented.
    assert not batch_first
    # assert shared_weights
    # assert bn

    return StackedGSU(
        num_layers,
        GSULayer,
        # first_layer_args=[GSUCell, input_size, hidden_size, shared_weights, bn],
        # other_layer_args=[GSUCell, hidden_size, hidden_size, shared_weights, bn],
        first_layer_args=[Rhy_GSUCell, input_size, hidden_size, shared_weights, bn],
        other_layer_args=[Rhy_GSUCell, hidden_size, hidden_size, shared_weights, bn],

    )


class StackedGSU(nn.Module):
    # __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedGSU, self).__init__()
        self.layers = init_stacked_gsu(num_layers, layer, first_layer_args, other_layer_args)

    def forward(self, input, states):
        output_states = []
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        all_layer_output = [input]
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            all_layer_output += [output]
            i += 1
        return output, output_states, all_layer_output


def init_stacked_gsu(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class GSULayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(GSULayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, state):
        inputs = input.unbind(0)
        outputs = []
        # logging.info(f"len(inputs): {len(inputs)}")
        # logging.info('len(inputs)', len(inputs))
        # print('len(inputs)', len(inputs))
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state, i)
            outputs += [out]
        return torch.stack(outputs), state


class Triangle(torch.autograd.Function):
    """Spike firing activation function"""

    @staticmethod
    def forward(ctx, input, gamma=1.0):
        out = input.ge(0.0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class GSUCell(nn.Module):
    def __init__(self, input_size, hidden_size, shared_weights=False, bn=False):
        super(GSUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.shared_weights = shared_weights
        self.use_bn = bn
        if self.shared_weights:
            self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        else:
            self.weight_ih = Parameter(torch.empty(2 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(2 * hidden_size))
        # self.bias_hh = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()

        # self.scale_factor = Parameter(torch.ones(hidden_size))
        if self.use_bn:
            self.batchnorm = nn.BatchNorm1d(hidden_size)

        # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        hx, cx = state
        if self.shared_weights:
            weight_ih = self.weight_ih.repeat((2, 1))
            weight_hh = self.weight_hh.repeat((2, 1))
        else:
            weight_ih = self.weight_ih
            weight_hh = self.weight_hh
        gates = (
            torch.mm(input, weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, weight_hh.t())
            # + self.bias_hh
        )
        forgetgate, cellgate = gates.chunk(2, 1)
        forgetgate = torch.sigmoid(forgetgate)
        cy = forgetgate * cx + (1 - forgetgate) * cellgate
        if self.use_bn:
            cy = self.batchnorm(cy)
        hy = Triangle.apply(cy)  # replace the Tanh activation function with step function to ensure binary outputs.

        return hy, (hy, cy)


# class Rhy_GSUCell(nn.Module):
#     def __init__(self, input_size, hidden_size, shared_weights=False, bn=False):
#         super(Rhy_GSUCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.shared_weights = shared_weights
#         self.use_bn = bn
#         if self.shared_weights:
#             self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
#             self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
#         else:
#             self.weight_ih = Parameter(torch.empty(2 * hidden_size, input_size))
#             self.weight_hh = Parameter(torch.empty(2 * hidden_size, hidden_size))
#         self.bias_ih = Parameter(torch.zeros(2 * hidden_size))
#         # self.bias_hh = Parameter(torch.zeros(2 * hidden_size))
#         self.reset_parameters()
#
#         # self.scale_factor = Parameter(torch.ones(hidden_size))
#         if self.use_bn:
#             self.batchnorm = nn.BatchNorm1d(hidden_size)
#
#         # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))
#         skip_length_min = 3
#         skip_length = 10
#         self.rhy_mask = self.create_mix_mask(hidden_size, skip_length_min, skip_length)
#
#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
#         for weight in self.parameters():
#             torch.nn.init.uniform_(weight, -stdv, stdv)
#
#     def create_mix_mask(self, dim=128, min_cycle=0, max_cycle=0):
#         # T = 784 // self.stride
#         # T = 8 * self.emb_size // self.input_size  # 8*256/32 = 64   8*768 / 64 = 96
#         # T = 8 * 768 // 64   # 96
#         T = 32 * 256  # 9 * 256 // 32 = 72
#
#         mask_cyc = []
#         for cycle in range(min_cycle, max_cycle + 1):
#             mask_ = []
#             for t in range(T):
#                 if t % cycle == 0:
#                     mask_.append(1)
#                 else:
#                     mask_.append(0)
#             mask_ = torch.tensor(mask_)
#             # tmp1 = mask_.detach().numpy()
#             mask_cyc.append(mask_)
#         mask_cyc = torch.stack(mask_cyc)
#         # tmp2 = mask_cyc.detach().numpy()
#
#         mask = mask_cyc
#         for n in range(1, dim // (max_cycle - min_cycle + 1) + 1):
#             mask = torch.cat((mask, torch.roll(mask_cyc, n, 1)), 0)
#             # tmp3 = mask.detach().numpy()
#         return mask[: dim]  # .to(device)  # [H, T]
#
#     def forward(self, input, state, time):
#         hx, cx = state
#         rhy_mask = self.rhy_mask[time].expand(cx.size(0), -1).to(cx.device)
#         if self.shared_weights:
#             weight_ih = self.weight_ih.repeat((2, 1))
#             weight_hh = self.weight_hh.repeat((2, 1))
#         else:
#             weight_ih = self.weight_ih
#             weight_hh = self.weight_hh
#         gates = (
#             torch.mm(input, weight_ih.t())
#             + self.bias_ih
#             + torch.mm(hx, weight_hh.t())
#             # + self.bias_hh
#         )
#         forgetgate, cellgate = gates.chunk(2, 1)
#         forgetgate = torch.sigmoid(forgetgate)
#         pre_cy = cx
#         cy = forgetgate * cx + (1 - forgetgate) * cellgate
#         cy = torch.where(rhy_mask == 0, pre_cy, cy)
#         if self.use_bn:
#             cy = self.batchnorm(cy)
#         hy = Triangle.apply(cy) * rhy_mask # replace the Tanh activation function with step function to ensure binary outputs.
#
#         return hy, (hy, cy)



class Rhy_GSUCell(nn.Module):
    def __init__(self, input_size, hidden_size, shared_weights=False, bn=False):
        super(Rhy_GSUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.shared_weights = shared_weights
        self.use_bn = bn
        if self.shared_weights:
            self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))
        else:
            self.weight_ih = Parameter(torch.empty(2 * hidden_size, input_size))
            self.weight_hh = Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.zeros(2 * hidden_size))
        # self.bias_hh = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()

        # self.scale_factor = Parameter(torch.ones(hidden_size))
        if self.use_bn:
            self.batchnorm = nn.BatchNorm1d(hidden_size)

        # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))
        cycle_min = 10
        cycle_max = 50
        duty_cycle_min = 0.05
        duty_cycle_max = 0.10
        phase_max = 0.5
        self.rhy_mask = self.create_general_mask(hidden_size, cycle_min, cycle_max, duty_cycle_min, duty_cycle_max, phase_max)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def create_general_mask(self, dim=128, c_min=4, c_max=8, min_dc=0.1, max_dc=0.9, phase_shift_max=0.5, T=784*4):
        mask = []
        # Create ranges for cycles and duty cycles
        dc_steps = torch.linspace(min_dc, max_dc, steps=dim)
        cycles = torch.linspace(c_min, c_max, steps=dim)
        # Generate phase shifts within the specified maximum
        phase_shifts = torch.linspace(0, int(phase_shift_max * c_max), steps=dim)

        for cycle, dc, phase_shift in zip(cycles, dc_steps, phase_shifts):
            cycle = int(torch.ceil(cycle))
            on_length = int(torch.ceil(dc * cycle))
            off_length = cycle - on_length
            pattern = [1] * on_length + [0] * off_length

            phase_shift = int(torch.round(phase_shift))
            pattern = pattern[-phase_shift:] + pattern[:-phase_shift]  # Negative slicing for shifting right

            full_pattern = pattern * (T // cycle) + pattern[:T % cycle]  # Ensure the pattern fits exactly into T
            mask.append(full_pattern)

        mask = torch.tensor(mask, dtype=torch.float32)
        return mask  # .to(device)

    def forward(self, input, state, time):
        hx, cx = state
        rhy_mask = self.rhy_mask[:, time].expand(cx.size(0), -1).to(cx.device)
        if self.shared_weights:
            weight_ih = self.weight_ih.repeat((2, 1))
            weight_hh = self.weight_hh.repeat((2, 1))
        else:
            weight_ih = self.weight_ih
            weight_hh = self.weight_hh
        gates = (
            torch.mm(input, weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, weight_hh.t())
            # + self.bias_hh
        )
        forgetgate, cellgate = gates.chunk(2, 1)
        forgetgate = torch.sigmoid(forgetgate)
        pre_cy = cx
        cy = forgetgate * cx + (1 - forgetgate) * cellgate
        cy = torch.where(rhy_mask == 0, pre_cy, cy)
        if self.use_bn:
            cy = self.batchnorm(cy)
        hy = Triangle.apply(cy) * rhy_mask # replace the Tanh activation function with step function to ensure binary outputs.

        return hy, (hy, cy)




class LIFCell(nn.Module):
    def __init__(self, input_size, hidden_size, bn=False, decay_factor=0.5, threshold=1.0):
        super(LIFCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bn = bn
        self.decay_factor = decay_factor
        self.threshold = threshold
        self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.zeros(hidden_size))
        # self.bias_hh = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()
        # self.scale_factor = Parameter(torch.ones(hidden_size))
        # if self.use_bn:
        #     self.batchnorm = nn.BatchNorm1d(hidden_size)

        # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        hx, cx = state

        weight_ih = self.weight_ih
        weight_hh = self.weight_hh
        input_current = (
                torch.mm(input, weight_ih.t())
                + self.bias_ih
                + torch.mm(hx, weight_hh.t())
            # + self.bias_hh
        )
        cy = self.decay_factor * cx + (1 - self.decay_factor) * input_current
        # if self.use_bn:
        #     hy = Triangle.apply(self.batchnorm(cy - self.threshold))
        #     # cy = self.batchnorm(cy)
        # else:
        hy = Triangle.apply(
            cy - self.threshold)  # replace the Tanh activation function with step function to ensure binary outputs.
        cy = cy - hy * self.threshold
        return hy, (hy, cy)


class ALIFCell(nn.Module):
    def __init__(self, input_size, hidden_size, bn=False, threshold=1.0):
        super(ALIFCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bn = bn
        self.threshold = threshold
        self.weight_ih = Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.zeros(hidden_size))
        # self.bias_hh = Parameter(torch.zeros(2 * hidden_size))
        self.reset_parameters()
        self.tau_m = Parameter(torch.zeros(hidden_size))
        # self.tau_m = Parameter(torch.ones(hidden_size) * (-1.3))
        # self.tau_adp = Parameter(torch.zeros(hidden_size))
        self.tau_adp = Parameter(torch.ones(hidden_size) * (-1.3))
        # tauM = 20
        # tauAdp_inital = 100
        # tauM_inital_std = 5
        # tauAdp_inital_std = 5
        # nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
        # nn.init.normal_(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
        # logging.info(f"tau_m: {self.tau_m}")
        # logging.info(f"tau_adp: {self.tau_adp}")
        # logging.info(f"decay_factor: {self.decay_factor}")
        # self.scale_factor = Parameter(torch.ones(hidden_size))
        # if self.use_bn:
        #     self.batchnorm = nn.BatchNorm1d(hidden_size)

        # self.scale = torch.tensor((1 - (-1)) / (127 - (-128)))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        if len(state) == 2:
            hx, cx = state
            cb = torch.ones_like(hx, device=hx.device)
        elif len(state) == 3:
            hx, cx, cb = state
        weight_ih = self.weight_ih
        weight_hh = self.weight_hh
        input_current = (
                torch.mm(input, weight_ih.t())
                + self.bias_ih
                + torch.mm(hx, weight_hh.t())
            # + self.bias_hh
        )

        tau_m = self.tau_m.sigmoid()
        tau_adp = self.tau_adp.sigmoid()
        cy = tau_m * cx + (1 - tau_m) * input_current
        hy = Triangle.apply(
            cy - self.threshold)  # replace the Tanh activation function with step function to ensure binary outputs.

        cb = tau_adp * cb + (1 - tau_adp) * hy
        # CB = 0.1 + 1.84 * cb
        # if self.use_bn:
        #     hy = Triangle.apply(self.batchnorm(cy - self.threshold))
        #     # cy = self.batchnorm(cy)
        # else:
        cy = cy - hy * cb
        return hy, (hy, cy, cb)




class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,thresh_t):  # input = membrane potential- threshold
        ctx.save_for_backward(input,thresh_t)
        return input.gt(thresh_t).float()  # is firing
    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input,thresh_t = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input-thresh_t) < 0.2
        return grad_input * temp.float(), -grad_input * temp.float()

act_fun_adp = ActFun_adp.apply






if __name__ == "__main__":
    input_size = 256
    hidden_size = 320
    num_layers = 2
    shared_weights = True
    bn = True
    batch_size = 128
    T = 100
    x = torch.rand((batch_size, input_size, T))  # [B, F, T]
    sequence_model = efficient_spiking_neuron(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        shared_weights=shared_weights,
        bn=bn,
    )

    states = [
        MemoryState(
            torch.zeros(batch_size, hidden_size, device=x.device),
            torch.zeros(batch_size, hidden_size, device=x.device),
        )
        for _ in range(num_layers)
    ]
    x = x.permute(2, 0, 1).contiguous()  # [B, F, T] => [T, B, F]
    o, _ = sequence_model(x, states)  # [T, B, F] => [T, B, F]
