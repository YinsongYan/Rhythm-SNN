"""Transforms on raw wav samples."""
import numpy
import random
import numpy as np
import librosa
from scipy.io import wavfile
import torch
from torch.utils.data import Dataset

def should_apply_transform(prob=0.5):
    """Transforms are only randomly applied with the given probability."""
    return random.random() < prob

class LoadAudio(object):
    """Loads an audio into a numpy array."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, data):
        path = data['path']
        if path:
            #samples, sample_rate = librosa.load(path=path, sr=self.sample_rate)
            try:
                sample_rate, samples = wavfile.read(path) # read as integer
            except ValueError as err:
                print('ValueError {} happen in {}'.format(err, path))
            samples = samples.astype(np.float32)
        else:
            # silence
            sample_rate = self.sample_rate
            samples = np.zeros(sample_rate, dtype=np.float32)
        data['samples'] = samples
        data['sample_rate'] = sample_rate
        return data

class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)
        if length < len(samples):
            data['samples'] = samples[:length]
        elif length > len(samples):
            data['samples'] = np.pad(samples, (0, length - len(samples)), "constant")
        return data

class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""

    def __init__(self, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range

    def __call__(self, data):
        if not should_apply_transform():
            return data

        data['samples'] = data['samples'] * random.uniform(*self.amplitude_range)
        return data

class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        sample_rate = data['sample_rate']
        scale = random.uniform(-self.max_scale, self.max_scale)
        speed_fac = 1.0  / (1 + scale)
        data['samples'] = np.interp(np.arange(0, len(samples), speed_fac), np.arange(0,len(samples)), samples).astype(np.float32)
        return data

class StretchAudio(object):
    """Stretches an audio randomly."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        scale = random.uniform(-self.max_scale, self.max_scale)
        data['samples'] = librosa.effects.time_stretch(data['samples'], 1+scale)
        return data

class TimeshiftAudio(object):
    """Shifts an audio randomly."""

    def __init__(self, max_shift_seconds=0.2):
        self.max_shift_seconds = max_shift_seconds

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        sample_rate = data['sample_rate']
        max_shift = (sample_rate * self.max_shift_seconds)
        shift = random.randint(-max_shift, max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        samples = np.pad(samples, (a, b), "constant")
        data['samples'] = samples[:len(samples) - a] if a else samples[b:]
        return data

class AddBackgroundNoise(Dataset):
    """Adds a random background noise."""

    def __init__(self, bg_dataset, max_percentage=0.5):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        noise = random.choice(self.bg_dataset)['samples']
        percentage = random.uniform(0.003, self.max_percentage)
        data['samples'] = samples * (1 - percentage) + noise * percentage
        return data


class AddNoiseSNR(Dataset):
    """Adds a random background noise.(Base on SNR(dB))"""

    def __init__(self, bg_dataset, min_SNR=0, max_SNR=50):
        self.bg_dataset = bg_dataset
        self.max_SNR = max_SNR
        self.min_SNR = min_SNR

    def __call__(self, data):
        #if not should_apply_transform():
            #return data

        samples = data['samples']
        noise = random.choice(self.bg_dataset)['samples']
        SNR = random.uniform(self.min_SNR, self.max_SNR)

        # Calculate signal power and Noise Power
        Spow = np.mean(np.power(samples, 2))
        Npow = np.mean(np.power(noise, 2))

        # Calculate SNR
        Ratio = np.sqrt(Spow/(np.power(10.0, SNR / 10.0) * Npow))

        # Add the noise
        data['samples'] = samples + Ratio * noise
        return data


class ExtractMFCC(object):
    """Extract MFCC feature"""
    def __init__(self, n_mfcc=64):
        self.n_mfcc = n_mfcc

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        data['samples'] = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=self.n_mfcc, hop_length=160, win_length=400)
        return data


class ToTensor(object):
    """Converts into a tensor."""

    def __init__(self, np_name, tensor_name, normalize=None):
        self.np_name = np_name
        self.tensor_name = tensor_name
        self.normalize = normalize

    def __call__(self, data):
        tensor = torch.FloatTensor(data[self.np_name])
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        data[self.tensor_name] = tensor
        return data
