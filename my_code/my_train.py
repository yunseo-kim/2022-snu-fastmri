from my_code.my_dataload import Dataset, ToTensor, Normalization
from torchvision import transforms, datasets
from my_code.my_unet import UNet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


transform = transforms.Compose([Normalization(means=0.5, std=0.5), ToTensor()])

lr = 1e-3 #learning rate
batch_size = 4
num_epoch = 100

data_dir = '' #data를불러올경로
ckpt_dir = ''#network저장될경로
log_dir = '' #tensorboard저장

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

dataset_train = Dataset(data_dir='', transform = transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val = Dataset(data_dir='', transform = transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

net = UNet().to(device)

#loss

#optimizer

