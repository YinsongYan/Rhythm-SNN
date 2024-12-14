import os
import numpy as np
import torch
import librosa
import torchaudio
from torch.utils.data import Dataset

__all__ = [ 'CLASSES', 'SpeechCommandsDataset', 'BackgroundNoiseDataset' ]

CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', ')

all_CLASSES = ['three', 'five', 'zero', 'one', 'left', 'visual', 'six', 'nine', 'cat', 'up', 'dog', 'off',
               'learn', 'four', 'bed', 'go', 'backward', 'bird', 'happy', 'two', 'stop', 'follow', 'forward',
               'down', 'eight', 'wow', 'no', 'seven', 'right', 'house', 'on', 'marvin', 'tree', 'sheila', 'yes']

class NGSCDataset(Dataset):
    
    def __init__(self, dir, transform=None, classes=CLASSES, silence_percentage=0.1, frontend="fbank",):
        self.frontend = frontend
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_CLASSES:
            if c not in class_to_idx:
                class_to_idx[c] = 0

        data = []
        for folder in os.listdir(dir):
            for f in os.listdir(os.path.join(dir, folder)):
                if os.path.isfile(os.path.join(dir, folder, f)) and f.endswith('.wav'):
                    c = f.strip().split('_')[0]
                    target = class_to_idx[c]
                    path = os.path.join(dir, folder, f)
                    data.append((path, target))

        # add silence
        target = class_to_idx['silence']
        data += [('', target)] * int(len(data) * silence_percentage)

        self.classes = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        if self.frontend == "fbank" or self.frontend == "Spiking_fbank": 

            feat = torchaudio.compliance.kaldi.fbank(torch.Tensor(data['samples']).unsqueeze(0), num_mel_bins=40)
            audio = torch.Tensor(data['samples']).unsqueeze(0)
        else:
            feat =torch.Tensor(data['samples']).unsqueeze(0)
            audio = torch.Tensor(data['samples']).unsqueeze(0)

        return feat, torch.tensor(data['target']), audio

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight
    
    def generateBatch(self, batch):

        feats, ys, xs = zip(*batch)
        featlens = torch.tensor([feat.shape[0] for feat in feats])
        feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).squeeze(2)
        ys = torch.LongTensor(ys)
        
        return feats, featlens, ys, xs, xlens

class NGSCDataset_evaluation(Dataset):
    """Google speech commands dataset. Only 'yes', 'no', 'up', 'down', 'left',
    'right', 'on', 'off', 'stop' and 'go' are treated as known classes.
    All other classes are used as 'unknown' samples.
    See for more information: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
    """

    def __init__(self, dir, transform=None, classes=CLASSES, condition=None, SNR=None, silence_percentage=0.1, frontend='fbank'):
        self.frontend = frontend
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_CLASSES:
            if c not in class_to_idx:
                class_to_idx[c] = 0

        data = []
        for folder in os.listdir(dir):
            f_name = folder.strip().split('_')
            # Noise
            # '''
            if (f_name[0] in condition and f_name[1] == SNR):# or f_name[0].startswith(SNR):
                for f in os.listdir(os.path.join(dir, folder)):
                    if os.path.isfile(os.path.join(dir, folder, f)) and f.endswith('.wav'):
                        c = f.strip().split('_')[0]
                        target = class_to_idx[c]
                        path = os.path.join(dir, folder, f)
                        data.append((path, target))
            '''
            # Clean
            if f_name[0] in SNR:
                for f in os.listdir(os.path.join(dir, folder)):
                    if os.path.isfile(os.path.join(dir, folder, f)) and f.endswith('.wav'):
                        c = f.strip().split('_')[0]
                        target = class_to_idx[c]
                        path = os.path.join(dir, folder, f)
                        data.append((path, target))
            '''


        # add silence
        target = class_to_idx['silence']
        data += [('', target)] * int(len(data) * silence_percentage)

        self.classes = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        if self.frontend == "fbank" or self.frontend == "Spiking_fbank": 

            feat = torchaudio.compliance.kaldi.fbank(torch.Tensor(data['samples']).unsqueeze(0), num_mel_bins=40)
            audio = torch.Tensor(data['samples']).unsqueeze(0)
        else:
            feat =torch.Tensor(data['samples']).unsqueeze(0)
            audio = torch.Tensor(data['samples']).unsqueeze(0)

        return feat, torch.tensor(data['target']), audio
    
    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight
    
    def generateBatch(self, batch):

        feats, ys, xs = zip(*batch)
        featlens = torch.tensor([feat.shape[0] for feat in feats])
        feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).squeeze(2)
        ys = torch.LongTensor(ys)
        
        return feats, featlens, ys, xs, xlens