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
hd1=[5,30]
hd2=[5,20]
ep=150
if __name__ == '__main__':
    import torch
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.enabled=True
    with open('SpeechCommands/configs/train_lstm_dev.yaml') as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    for i in range(hd1[0],hd1[1]):
        for j in range(hd2[0],hd2[1]):
            config
    run.main(config)
