import torch
from SpeechCommands.utils import count_parameters
import torch
import os
import numpy as np
import copy
#from src.runner.test import test
import datetime
import ml_collections
import yaml
from SpeechCommands.dataloader import get_dataset
from SpeechCommands.utils import model_path, EarlyStopping
import argparse


#model_path = 'models/SpeechCommands_model_DEV_param_SO_nhid1_10_nhid2_0_mfcc_True_gamma_1_lr_0.001_comment_None.pt' # Update with your correct path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
test_dataset = torch.load('test_dataset.pt')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# 创建一个新的模型实例
model1 = development_model(n_inputs=12, n_hidden1=10, n_outputs=2,param=so)

model_path = '/root/NewName/SpeechCommands/models/SpeechCommands_model_DEV_param_SO_nhid1_10_nhid2_0_mfcc_True_gamma_1_lr_0.001_comment_None.pt'

# 将存储的模型参数加载到新创建的模型中
model1.load_state_dict(torch.load(model_path))

model1.eval()  # 切换模型到评估模
# 2. 加载模型
model1.eval()
true_labels = []
preds = []

with torch.no_grad():
    for data, labels in data_loader:
        output = model1(data)
        preds.extend(output.detach().cpu().numpy())
        true_labels.extend(labels.detach().cpu().numpy())

# 5. 计算 AUC
auc_score = roc_auc_score(true_labels, preds)
print(f'The AUC score is {auc_score}')
