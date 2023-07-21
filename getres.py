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
import run
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
hd1=[6,9]
hd2=[6,9]
ep=150
if __name__ == '__main__':
    import torch
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.enabled=True
    with open('SpeechCommands/configs/train_lstm_dev.yaml') as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    config.model="cnn"
    config.kernel_size=90
    config.stride=90
    config.padding=20
    config.kernel_sizeP=50
    config.strideP=50
    config.weight_decay=5
    for i in range(hd1[0],hd1[1]):
        for j in range(hd2[0],hd2[1]):
            config.kernel_size=50+i*5
            config.stride=20+5*i
            config.n_hidde1=10+j
            config.n_hidden2=10+j
            run.lstmDev(config)
    run.main(config)
