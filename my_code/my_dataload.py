import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import torch

dir_data = '../../input/train/image/brain95.h5'
f = h5py.File('../../input/train/image/brain95.h5', 'r')

print(list(f.keys()))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        #input, test, val로 데이터셋을 나눈 후 그 안에 input과 label 있음
        lst_data = os.listdir(self.data_dir) #data_dir에 있는 파일 리스트 불러옴

        lst_label = [f for f in lst_data if f.startswith('label')] #제일 앞에 label이라고 있는 것들 여기에 넣음
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])) #local location에 있을 때 (remote host에서 어떻게 불러오지)
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input' : input, 'label': label}

        if self.tranform:
            data = self.transform(data)

        return data

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label':torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input':input}

        return data

