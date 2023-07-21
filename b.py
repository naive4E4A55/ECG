import os
import pathlib
import urllib.request
import tarfile
import torch
import torchaudio
import ml_collections
from typing import Tuple
#from .utils import normalise_data, split_data, load_data, save_data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import TensorDataset, DataLoader, random_split
data = torch.load("inputs_tensor1.pt")
labels_tensor = torch.load("labels_tensor1.pt")
dataset = TensorDataset(data, labels_tensor)
print(1)
print(data)
print(labels_tensor)
data = data.permute(0, 2, 1)
data = data.float()
train_size = int(0.8 * len(dataset))  # 使用80%的数据作为训练集
test_size = int(0.1 * len(dataset))  # 使用10%的数据作为测试集
val_size = len(dataset) - train_size - test_size  # 剩下的10%作为验证集

# 使用 random_split 函数来划分数据集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
# 保存划分后的数据集到文件
torch.save(train_dataset, 'train_dataset.pt')
torch.save(val_dataset, 'val_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')
