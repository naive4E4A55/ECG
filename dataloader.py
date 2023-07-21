"""
Adapted from https://github.com/patrick-kidger/NeuralCDE/blob/758d3a7134e3a691013e5cc6b7f68f277e9e6b69/experiments/datasets/speech_commands.py
"""
import os
import pathlib
import urllib.request
import tarfile
import torch
import torchaudio
import ml_collections
from typing import Tuple
from .utils import normalise_data, split_data, load_data, save_data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import TensorDataset, DataLoader, random_split


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # Convert features to float type on the fly
        features = self.data[0][index].float()
        label = self.data[1][index]  # Keep labels as they are
        return features, label

    def __len__(self):
        return len(self.data[0])

def addData(data,labels_tensor):
    # 找到所有标签为1的索引
    indices = (labels_tensor == 1)

    # 使用这些索引复制相应的数据
    data_copy = data[indices]
    labels_copy = torch.ones(data_copy.shape[0], dtype=torch.long) # 注意这里改为了long类型
    #from torch.utils.data import TensorDataset, DataLoader
    zeros_count = torch.sum(labels_tensor.eq(0)).item()
    ones_count = torch.sum(labels_tensor.eq(1)).item()
    for i in range(0,10):
        data=torch.cat((data,data_copy),dim=0)
        labels_tensor=torch.cat((labels_tensor, labels_copy), dim=0)
        print("Data shape: ", data.shape)
        print("Labels shape: ", labels_tensor.shape)
    return data,labels_tensor

class OverSamplingDataset(Dataset):
    def __init__(self, dataset, factor=1):
        self.dataset = dataset
        self.factor = factor

    def __getitem__(self, index):
        if index < len(self.dataset):
            # 原有样本，不做任何修改
            return self.dataset[index]
        else:
            # 额外的样本，通过复制实现
            index = index % len(self.dataset)
            data, label = self.dataset[index]
            if label == 1:  # 只复制标签为1的样本
                data = data.clone()  # 注意这里是深拷贝
            return data, label

    def __len__(self):
        # 在原有样本数量基础上增加
        return len(self.dataset) * self.factor
class BalancedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)
def f(train_dataset):
    from imblearn.over_sampling import SMOTE
    import numpy as np
    # 将数据和标签转换为numpy数组
    train_data_np = np.array([data.numpy() for data, _ in train_dataset])
    train_labels_np = np.array([label.numpy() for _, label in train_dataset])
    # 进行SMOTE过采样
    sm = SMOTE(random_state=42)
    train_data_res, train_labels_res = sm.fit_resample(train_data_np, train_labels_np)

    # 将过采样后的数据转换为TensorDataset
    train_data_res = [torch.from_numpy(data) for data in train_data_res]
    train_labels_res = [torch.from_numpy(label) for label in train_labels_res]
    train_dataset_resampled = TensorDataset(train_data_res, train_labels_res)

def getTrainTestData(config: ml_collections.ConfigDict):
    train_dataset = torch.load('train_dataset.pt')
    val_dataset = torch.load('val_dataset.pt')
    test_dataset = torch.load('test_dataset.pt')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    test_loader=DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    return train_loader,val_loader,test_loade
def augment(train_dataset):
    # 从 Subset 对象中获取所有的数据和标签
    all_data = [train_dataset[i][0] for i in range(len(train_dataset))]
    all_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

    # 将它们转化为张量
    data_tensor = torch.stack(all_data)
    all_labels_tensor = torch.tensor(all_labels)
    #print(data_tensor)
    j=len(data_tensor)
    d=[]
    l=[]
    for i in range(0,j):
        if(all_labels_tensor[i]==1):
            variance_per_channel = data_tensor[i].var(dim=1)
            copied_tensor=data_tensor[i].clone()
            noise_per_channel = torch.randn_like(data_tensor[i])*0*1.0/10000*variance_per_channel.view(-1, 1).sqrt()
            copied_tensor+=noise_per_channel
            d.append(copied_tensor)
            l.append(1)
            #print(copied_tensor)
            #print(data_tensor[i])
            #print(variance_per_channel)
            #print(1)
        #if(all_labels_tensor[i]==1):
            #variance_per_channel = data_tensor[i].var(dim=1)
            #copied_tensor=data_tensor[i].clone()
            #noise_per_channel = torch.randn_like(data_tensor[i])*1.0/1000*variance_per_channel.view(-1, 1).sqrt()
            #copied_tensor+=noise_per_channel
            #d.append(copied_tensor)
            #l.append(1)
            #print(copied_tensor)
            #print(data_tensor[i])
            #print(variance_per_channel)
            #print(1)
    data_tensor_new = torch.stack(d)
    labels_tensor_new = torch.tensor(l,dtype=torch.long)
    # 将新的数据张量和标签张量添加到原始的张量中
    data_tensor = torch.cat((data_tensor, data_tensor_new), dim=0)
    labels_tensor = torch.cat((all_labels_tensor,labels_tensor_new),dim=0)
    train_dataset = TensorDataset(data_tensor, labels_tensor)
    return train_dataset




def get_dataset(
    config: ml_collections.ConfigDict,
    num_workers: int = 4,
    data_root="./data",
) -> Tuple[dict, torch.utils.data.DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple ( dict(train_loader, val_loader) , test_loader)
    """
    data = torch.load("inputs_tensor1.pt")
    labels_tensor = torch.load("labels_tensor1.pt")
    #print(1)
    #print(data)
    #print(labels_tensor)
    data = data.permute(0, 2, 1)
    data = data.float()
    #data = data.double()
    # 加载数据集
    train_dataset = torch.load('train_dataset.pt')
    val_dataset = torch.load('val_dataset.pt')
    test_dataset = torch.load('test_dataset.pt')
    # 找到所有标签为1的索引

    #class_counts = torch.bincount(labels_tensor)
    #majority_class = torch.argmax(class_counts).item()
    #minority_class = 1 - majority_class  # 假设我们只有两个类别
    # 找出少数类别的样本索引
    #minority_indices = (labels_tensor == minority_class).nonzero(as_tuple=True)[0]
    #print("minority_indices",minority_indices)
    #print("majority_clas",majority_class)
    #import numpy as np
    # 找到样本数量最少的类别
    #minority_class = np.argmin(class_counts)

    # 找到所有少数类别的样本索引
    #minority_indices = np.where(np.array([label for _, label in dataset]) == minority_class)[0]

    # 找到所有多数类别的样本索引
    #majority_indices = np.where(np.array([label for _, label in dataset]) != minority_class)[0]

    # 复制少数类别的样本索引，使其数量与多数类别相等
    #oversampled_minority_indices = np.random.choice(minority_indices, size=len(majority_indices), replace=True)

    # 将多数类别的样本索引和复制的少数类别的样本索引合并
    #balanced_indices = np.concatenate([majority_indices, oversampled_minority_indices])

    # 创建一个平衡的数据集
    #balanced_dataset = BalancedDataset(dataset, balanced_indices)
    # 重复少数类别的样本索引，使得其数量与多数类别相同
    #oversampled_indices = minority_indices.repeat(class_counts[majority_class] // class_counts[minority_class])
    #remainder = class_counts[majority_class] % class_counts[minority_class]
    #oversampled_indices = torch.cat([oversampled_indices, minority_indices[:remainder]])
    # 创建平衡数据集
    #balanced_dataset = BalancedDataset(dataset, oversampled_indices)
    #dataset=balanced_dataset


    #data_loader = DataLoader(dataset, shuffle=True, num_workers=1)
    #test_loader=DataLoader(dataset, shuffle=True, num_workers=1)
    # 假设 data 是你的数据，labels_tensor 是你的标签
    # 定义训练集和验证集的大小
    dataset = TensorDataset(data, labels_tensor)
    train_size = int(0.9* len(dataset))  # 使用80%的数据作为训练集
    val_size = len(dataset) - train_size  # 剩下的20%作为验证集
    train_size = int(0.8 * len(dataset))  # 使用80%的数据作为训练集
    test_size = int(0.1 * len(dataset))  # 使用10%的数据作为测试集
    val_size = len(dataset) - train_size - test_size  # 剩下的10%作为验证集
    # 使用 random_split 函数来划分数据集
    #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    #train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    #torch.save(train_dataset, 'train_dataset.pt')
    #torch.save(val_dataset, 'val_dataset.pt')
    #torch.save(test_dataset, 'test_dataset.pt')
    #print('Total dataset size:', len(dataset))
    #print('Train size:', train_size)
    #print('Validation size:', val_size)
    train_dataset = torch.load('train_dataset.pt')
    val_dataset = torch.load('val_dataset.pt')
    test_dataset = torch.load('test_dataset.pt')
    #print('Test size:', test_size)
    # 找出需要复制的类的索引
    # 获取训练集的标签
    #train_features,labels_tensor=train_dataset.tensors
    #train_labels = labels_tensor[train_dataset.indices]

    # 找出需要复制的类的索引
    #indices = (train_labels == 1).nonzero(as_tuple=True)
    # 创建一个用于存储原始数据集标签的张量
    #original_labels = train_dataset.dataset.tensors[1]
    #train_labels = original_labels[train_dataset.indices]

    # 找出需要复制的类的索引
    #indices = (train_labels == 1).nonzero(as_tuple=True)[0]

    # 创建一个新的weights，对需要复制的类的样本赋予更大的权重
    #weights = torch.ones(len(train_dataset))
    #weights[indices]=2  # 可以根据需要调整这个值，这里我们设置为10

    #from torch.utils.data import WeightedRandomSampler
    # 使用这个weights创建一个新的OverSampler
    #oversampler = WeightedRandomSampler(weights, num_samples=len(weights))
    #indices = (train_dataset.dataset.tensors[1] == 1).nonzero(as_tuple=True)[0]
    #print("train_dataset.dataset.tensors[1]",indices)
    #oversampler = WeightedRandomSampler(weights, num_samples=len(weights))
    # 使用这个OverSampler创建一个新的数据加载器
    #train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=oversampler, num_workers=1)
    #for i,j in train_loader:
    #    print(j)
    #enhanced_train_dataset = OverSamplingDataset(train_dataset, factor=2)

    #train_loader = DataLoader(enhanced_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)

    #print(train_dataset)
    for i in range(0,2):
        train_dataset=augment(train_dataset)


    # 创建数据加载器 (data loaders)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)
    #val_loader=train_loader
    test_loader=DataLoader(test_dataset,batch_size=config.batch_size, shuffle=True, num_workers=1)
    #train_loader,val_loader,test_loader=getTrainTestData(config)
    # 保存到 dataloaders 字典中
    dataloaders = {
            "train": train_loader,
        "validation": val_loader
    }
    #dataloaders["train"]=data_loader
    #dataloaders["validation"]=data_loader

    return dataloaders, test_loader

    return dataloaders, test_loader

