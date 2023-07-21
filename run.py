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


def train_SC(
    model, dataloader, config, test_loader
):

    # Training parameters
    epochs = config.epochs
    device = config.device
    # clip = config.clip

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999
    # iterate over epochs
    #import pdb
    #pdb.set_trace()
    print(model)
    resauc=[]
    res=[]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01*config.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.gamma)
    criterion = torch.nn.CrossEntropyLoss()
    counter = 0
    maxAuc=0
    # wandb.watch(model, criterion, log="all", log_freq=1)
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
                print(type(dataloader[phase]))
            else:
                model.eval()
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import label_binarize
            probs = []
            real_labels = []


            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0
            # iterate over data
            for inputs, labels in dataloader[phase]:
                inputs = inputs.permute(0, 2, 1).to(device)
                inputs=inputs.float()
                #inputs = inputs.double()
                #print(inputs.dtype)
                labels = labels.to(device)
                #print(labels)
                #import pdb
                #pdb.set_trace()
                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)
                    #print(outputs)
                    loss = criterion(outputs, labels)
                    # Regularization:
                    l1_lambda = 0
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l1_lambda * l1_norm
                    _, preds = torch.max(outputs, 1)
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_
                        optimizer.step()
                        torch.cuda.empty_cache()
                    import torch.nn.functional as F
                    outputs = F.softmax(outputs, dim=1) 
                    probs.append(outputs.detach().cpu().numpy())
                    real_labels.append(labels.detach().cpu().numpy())


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)
                #print(probs)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))
            # 在每个epoch结束后，计算AUC
            probs = np.concatenate(probs)
            real_labels = np.concatenate(real_labels)
            # one hot encoding
            n_classes = np.unique(real_labels).shape[0]
            real_labels_one_hot = label_binarize(real_labels, classes=np.arange(n_classes))
            one_hot = np.eye(np.max(real_labels_one_hot+1))[real_labels_one_hot]
            real_labels_one_hot=np.squeeze(one_hot,axis=1)
            #print(one_hot)
            #print(real_labels_one_hot)
            #print(real_labels_one_hot.shape)
            # Compute AUC
            auc_score = roc_auc_score(real_labels_one_hot, probs, multi_class='ovr')
            print(phase,f'AUC: {auc_score}')
            #probs = np.fliplr(probs)
            auc_score = roc_auc_score(real_labels_one_hot, probs, multi_class='ovr')
            print(f'AUC: {auc_score}')
            #resauc.append(auc_score)

            print("{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}".format(
                phase, epoch_loss, epoch_acc, auc_score),flush=True)
            print(datetime.datetime.now())

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_acc >= 0:
                #resauc.append(auc_score)
                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss) and False:
                    pass
                else:
                    print(probs)
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()
                    # Perform test and log results
                    test_acc = test_SC(model, test_loader, config)
                    res.append(test_acc)
                    resauc.append(auc_score)
                    print(len(res))
                    print(len(resauc))
                    #if(auc_score>maxAuc):
                    #    res=test_acc
            if phase == "validation":
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'min').step(metrics=best_loss)
                EarlyStopping(patience=30)(val_acc=best_acc)
                if(auc_score>maxAuc):
                    maxAuc=auc_score
                    maxModel=copy.deepcopy(model.state_dict())
        if counter > config.patience:
            break
        else:
            lr_scheduler.step()
            print()

        lr_scheduler.step()
        print()
    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    res=np.array(res)
    resauc=np.array(resauc)
    indice=resauc.argsort()[-9:]
    res=[res[i] for i in indice]
    resauc=[resauc[i] for i in indice]
    print(len(res))
    print(len(resauc))
    print(resauc)
    print(maxAuc)
    print(res)
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), config.path)
    # Save the model in the exchangeable ONNX format
    torch.save(model.state_dict(), config.path)
    model.load_state_dict(maxModel)
    torch.save(model.state_dict(), config.path)
    # Save the model in the exchangeable ONNX format
    torch.save(model.state_dict(), config.path)

    # Return model and histories
    return model


def test_SC(model, test_loader, config):
    # send model to device
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0
    probs = []
    real_labels = []
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    with torch.no_grad():

        # Iterate through data
        for inputs, labels in test_loader:
            inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            probs.append(outputs.detach().cpu().numpy())
            real_labels.append(labels.detach().cpu().numpy())
            #outputs[outputs == 0] =2
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results
    test_acc = correct / total
    print(
        "Accuracy of the network on the {} test samples: {}".format(
            total, (100 * test_acc)
        )
    )
    probs = np.concatenate(probs)
    real_labels = np.concatenate(real_labels)
    # one hot encoding
    n_classes = np.unique(real_labels).shape[0]
    real_labels_one_hot = label_binarize(real_labels, classes=np.arange(n_classes))
    one_hot = np.eye(np.max(real_labels_one_hot+1))[real_labels_one_hot]
    real_labels_one_hot=np.squeeze(one_hot,axis=1)

    # Compute AUC
    auc_score = roc_auc_score(real_labels_one_hot, probs, multi_class='ovr')
    print(f'AUC: {auc_score}')
    auc_score = roc_auc_score(real_labels_one_hot, probs, multi_class='ovr')
    print(f'AUC: {auc_score}')
    #resauc.append(auc_score)
    return auc_score


def lstmDev(config):
    import sys
    outFile=open(getModelFileOutName(config),'w')
    sys.stdout=outFile
    main(config)



def getModelFileOutName(config):
    res=config.model+"-config.n_hidden1"+str(config.n_hidden1)+"-config.n_hidden2"+str(config.n_hidden2)+"-"+str(config.lr)+"-"+str(config.kernel_size)+"-"
    res+=str(config.stride)+"-"+str(config.padding)+"-"+str(config.kernel_sizeP)+"-"+str(config.strideP)+'output.txt'
    return res

def main(config):
    # print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print(type(config))  # <class 'dict'>
    # Set the seed
    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)

    print(config)
    #import pdb
    #pdb.set_trace()
    if (config.device ==
            "cuda" and torch.cuda.is_available()):
        config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)
    #config.update({"device": "cpu"}, allow_val_change=True)
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    from SpeechCommands.models import get_model
    model = get_model(config)

    #model = model.double()
    num_param = count_parameters(model)
    print('num_param;', num_param)
    #import pdb
    #pdb.set_trace()


    # Define transforms and create dataloaders
    dataloaders, test_loader = get_dataset(config, num_workers=1)

        #print(dataloaders[i].shape)
    #for dataloader in dataloaders:
        #first_batch = next(iter(dataloader))
        #print(dataloader)
        #for i, batch in enumerate(dataloaders[dataloader]):
            #data, labels = batch
            #print(f"Batch {i + 1}:")
            #print("Data:", data)
            #print("Labels:", labels
    torch.backends.cudnn.enabled = False
    # Create model directory and instantiate config.path
    model_path(config)

    if config.pretrained:
        # Load model state dict
        model.module.load_state_dict(torch.load(config.path), strict=False)

    # Train the model
    if config.train:
        # Print arguments (Sanity check)

        # Train the model
        import datetime

        print(datetime.datetime.now())
        train_SC(model, dataloaders, config, test_loader)
    # Select test function
    test_acc = test_SC(model, test_loader, config)
    return test_acc

if __name__ == '__main__':
    import torch
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.enabled=True
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM_DEV',
                        help='choose from LSTM, DEV, LSTM_DEV, signature')

    args = parser.parse_args()
    if args.model == 'LSTM_DEV':
        with open('SpeechCommands/configs/train_lstm_dev.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
        #import sys
        #file = open('output.txt', 'w')
        #sys.stdout = file
    elif args.model == 'LSTM':
        with open('SpeechCommands/configs/train_lstm.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    elif args.model == 'DEV':
        with open('SpeechCommands/configs/train_dev.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    elif args.model == 'signature':
        with open('SpeechCommands/configs/train_sig.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))

    main(config)
