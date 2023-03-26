import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
import json
import argparse

import utilities

args = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

args.add_argument('data_dir', nargs='?', action="store", default="./flowers/", help='dataset folder used for training, validation and testing')
args.add_argument('--gpu', dest="gpu", action="store_true", default="False", help='use gpu for training')
args.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth", help='checkpoint for saving the trained model')
args.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, help='learning rate for the training')
args.add_argument('--epochs', dest="epochs", action="store", type=int, default=10, help='number of epochs for the training')
args.add_argument('--arch', dest="arch", action="store", choices=["densenet121","densenet161"], default="densenet161", type = str, help='sets the architecture of the network')
args.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512, help='number of nodes in the hidden layer of the classifier')

args = args.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
lr = args.learning_rate
arch = args.arch
hidden_units= args.hidden_units
gpu = args.gpu
epochs = args.epochs

# print(data_dir)
# print(gpu)
# print(arch)
# print(epochs)
# print(lr)
# print(hidden_units)
# print(save_dir)


with open('cat_to_name.json', 'r') as f:
    flower_to_name = json.load(f)
flower_species=len(flower_to_name)

image_datasets, dataloaders = utilities.load_data(data_dir)
model=utilities.model_setup(arch,hidden_units,flower_species,gpu)
criterion, optimizer=utilities.optimizer_setup(model,lr)
model=utilities.train(epochs, dataloaders, model, optimizer, criterion, gpu, './model_transfer_densenet161.pt')
utilities.test(dataloaders, model, criterion, gpu)
utilities.save_checkpoint(path,arch,image_datasets,model,hidden_units,flower_species)