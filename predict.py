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
import pandas as pd

import utilities

args = argparse.ArgumentParser(
    description='predict.py')
args.add_argument('input_img', action="store", type = str, default='flowers/test/28/image_05230.jpg', help='image path for inference')
args.add_argument('checkpoint', nargs='*', action="store",type = str, default='./checkpoint.pth', help='checkpoint used for loading the trained model')
args.add_argument('--top_k', dest="top_k", action="store", type=int,  default=5, help='return top K most likely classes')
args.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help='sse a mapping of categories to real names')
args.add_argument('--gpu', action="store_true", dest="gpu", default="False", help='use gpu for inference')

args = args.parse_args()
image_path = args.input_img
top_k = args.top_k
gpu = args.gpu
checkpoint = args.checkpoint
category_names=args.category_names

print("Image file:",image_path)
# print(top_k)
# print(gpu)
# print(checkpoint)
x = image_path.split("/")
with open(category_names, 'r') as f:
    flower_to_name = json.load(f)
flower_species=len(flower_to_name)
target_class = flower_to_name[x[-2]]
print("Target class:",target_class)

#utilities.load_model(checkpoint)
# checkpoint = torch.load(checkpoint)
# print(checkpoint.keys())

#print(flower_species)

model=utilities.load_model(checkpoint,flower_species)
#print(model)
value,kclass = utilities.predict(image_path, model,gpu,top_k)
#print(kclass)


idx_to_class = {model.class_to_idx[i]:i for i in model.class_to_idx.keys()}
classes = [flower_to_name[idx_to_class[c]] for c in kclass]
#print(classes)
#print(value)
data={'Predicted Class':classes,'Probablity': value}
dframe=pd.DataFrame(data)
print(dframe.to_string(index=False))

