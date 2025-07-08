import sys
# sys.path.append('bin/spiking_neuron')
from abc import abstractmethod
from typing import Callable
import torch
# from spiking_neuron import base
from . import surrogate, base
# import base

class BaseNode(base.MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 surrogate_function=None,
                 hard_reset: bool = False,
                 detach_reset: bool = False):

        assert isinstance(v_threshold, float)
        assert isinstance(hard_reset, bool)
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.register_memory('v', 0.)

        self.v_threshold = v_threshold

        self.hard_reset = hard_reset
        self.detach_reset = detach_reset

        self.surrogate_function = surrogate_function

    def forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.hard_reset:
            self.v = self.v * (1. - spike_d)
        else:
            self.v = self.v - spike_d * self.v_threshold

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, detach_reset={self.detach_reset}, hard_reset={self.hard_reset}'

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class LIFNode(BaseNode):
    def __init__(self,
                 decay_factor: float = 1.,
                 v_threshold: float = 1.,
                 surrogate_function: Callable = None,
                 hard_reset: bool = False,
                 detach_reset: bool = False):
        # print(f"tau: {tau} type: {type(tau)}")
        self.decay_factor = torch.tensor(decay_factor).float()
        # assert isinstance(tau, float) and tau >= 1.

        super().__init__(v_threshold, surrogate_function, hard_reset, detach_reset)

    def extra_repr(self):
        return super().extra_repr() + f', decay_factor={self.decay_factor}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v * self.decay_factor + x



class Rhythm_LIFNode(BaseNode):
    def __init__(self,
                 decay_factor: float = 1.,
                 v_threshold: float = 1.,
                 surrogate_function: Callable = surrogate.sigmoid(),
                 hard_reset: bool = False,
                 detach_reset: bool = False,
                 hidden_dim = 10,
                 period=3):
        # print(f"tau: {tau} type: {type(tau)}")
        self.decay_factor = torch.tensor(decay_factor).float()
        # assert isinstance(tau, float) and tau >= 1.

        super().__init__(v_threshold, surrogate_function, hard_reset, detach_reset)
        self.v_threshold = 1.
        self.mask = self.create_mix_mask(hidden_dim, period, period)


    def create_mix_mask(self, dim=10, min_cycle=0, max_cycle=0):
        # T = 8*256 // self.stride
        T = 32*200
        mask_cyc = []
        for cycle in range(min_cycle, max_cycle + 1):
            mask_ = []
            for t in range(T):
                if t % cycle == 0:
                    mask_.append(1)
                else:
                    mask_.append(0)
            mask_ = torch.tensor(mask_)
            # tmp1 = mask_.detach().numpy()
            mask_cyc.append(mask_)
        mask_cyc = torch.stack(mask_cyc)
        # tmp2 = mask_cyc.detach().numpy()

        mask = mask_cyc
        for n in range(1, dim // (max_cycle - min_cycle + 1) + 1):
            # mask = torch.cat((mask, torch.roll(mask_cyc, n, 1)), 0)
            mask = torch.cat((mask, torch.roll(mask_cyc, 0, 1)), 0)
            # tmp3 = mask.detach().numpy()
        return mask[: dim]  #.to(device)  # [H, T]

    def extra_repr(self):
        return super().extra_repr() + f', decay_factor={self.decay_factor}'

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)

    def neuronal_charge(self, x, mask):
        pre_v = self.v
        self.v = self.v * self.decay_factor + x * (1 - self.decay_factor)
        self.v = torch.where(mask == 0, pre_v, self.v)


    def neuronal_fire(self, mask):
        return self.surrogate_function(self.v - self.v_threshold) # * mask
        # return self.surrogate_function.apply(self.v - self.v_threshold) * mask

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.hard_reset:
            self.v = self.v * (1. - spike_d)
        else:
            self.v = self.v - spike_d * self.v_threshold

    def forward(self, x, t):
        # print('self.mask[:, t]', self.mask[:, t])
        # print('x.size(0)', x.size(0))
        mask = self.mask[:, t]  # .expand(x.size(0), -1)
        self.v_float_to_tensor(x)
        self.neuronal_charge(x, mask)
        spike = self.neuronal_fire(mask)
        self.neuronal_reset(spike)
        return spike




# def mem_update_skip_woDecay(ops, x, mem, spike, mask):
#     mask = mask.expand(mem.size(0), -1)
#     pre_mem = mem
#     mem = mem * decay * (1. - spike) + ops(x)
#     mem = torch.where(mask == 0, pre_mem, mem)
#     # tmp2 = (mem).detach().cpu().numpy()
#     spike = act_fun(mem) * mask
#     return mem, spike


