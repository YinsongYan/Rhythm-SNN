import torch
import random
import soundfile as sf
import numpy as np
import io
import torch
from torch.utils.data import DataLoader
import transformers
import torchaudio

def _collate_fn_raw_multiclass(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    channel_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, channel_size, max_seqlength)
    feats = []
    targets = torch.LongTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        feats.append(sample[1])
        target = sample[2]
        seq_length = real_tensor.size(1)
        # inputs[x] = real_tensor
        inputs[x].narrow(1, 0, seq_length).copy_(real_tensor)
        targets[x] = target
    input_feature = torch.cat(feats, dim=0)
    return inputs, input_feature, targets


def load_audio(f, sr, min_duration: float = 5.,
               read_cropped=False, frames_to_read=-1, audio_size=None):
    if min_duration is not None:
        min_samples = int(sr * min_duration)
    else:
        min_samples = None
    # x, clip_sr = torchaudio.load(f, channels_first=False)
    # x = x.squeeze().cpu().numpy()
    if read_cropped:
        assert audio_size
        assert frames_to_read != -1
        if frames_to_read >= audio_size:
            start_idx = 0
        else:
            start_idx = random.randint(0, audio_size - frames_to_read - 1)
        x, clip_sr = sf.read(f, frames=frames_to_read, start=start_idx)
        # print("start_idx: {} | clip size: {} | frames_to_read:{}".format(start_idx, len(x), frames_to_read))
        min_samples = frames_to_read
    else:
        x, clip_sr = sf.read(f)     # sound file is > 3x faster than torchaudio sox_io
    x = x.astype('float32')#.cpu().numpy()
    assert clip_sr == sr

    # min filtering and padding if needed
    if min_samples is not None:
        if len(x) < min_samples:
            tile_size = (min_samples // x.shape[0]) + 1
            x = np.tile(x, tile_size)[:min_samples]
    return x


def setup_dataloaders(train_set, val_set, batch_size,
                      device_world_size=1, local_rank=0,
                      collate_fn=None, num_workers=4,
                      multi_device_val=False, need_val=True):
    train_sampler = None
    val_sampler = None
    tr_shuffle = True
    if device_world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=device_world_size,
            rank=local_rank,
            shuffle=True)
        tr_shuffle = False
        if multi_device_val:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_set,
                num_replicas=device_world_size,
                rank=local_rank,
                shuffle=False
            )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=tr_shuffle,
                              sampler=train_sampler,
                              num_workers=num_workers, collate_fn=collate_fn)
    if need_val:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                sampler=val_sampler,
                                num_workers=num_workers, collate_fn=collate_fn)
    else:
        val_loader = None
    return train_loader, val_loader

def setup_testloader(test_set, collate_fn):
    return DataLoader(test_set, batch_size=1, num_workers=2, collate_fn=collate_fn)