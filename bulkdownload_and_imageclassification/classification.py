#!/usr/bin/env python
# coding: utf-8

# In[2]:


from efficientnet_pytorch import EfficientNet
import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def processing(metabolite, device):
    model_name = 'efficientnet-b0'
    model = EfficientNet.from_pretrained(model_name,num_classes=2)
    model.load_state_dict(torch.load('./model/image_classification.pt'))
    model = model.to(device)
    dataset = datasets.ImageFolder(metabolite,
                                   transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ]))
    batch_size  = 4
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    dataloaders = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=4)

    allFiles, _ = map(list, zip(*dataloaders.dataset.samples))
    
    return model, allFiles, dataloader

def classification(metabolite, device):
    model, allFiles, dataloader= processing(metabolite, device)
    model.eval()
    total_result=dict()
    for i, (inputs, _) in enumerate(dataloader):
        outputs=model(inputs.to(device))
        _, preds = torch.max(outputs, 1)
        preds= list(preds.detach().cpu().numpy())
        for j in range(inputs.size()[0]):
            image_name= allFiles[i*batch_size +j]
            total_result[image_name]= preds[j]
    if not os.path.exists(os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'input'):
        os.mkdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'input')
        
    for name,label in total_result.items():
    revise_name= name.split('/')[-1]
    if label==0:
        os.remove(name)
    else:
        shutil.move(name, os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'input/'+revise_name)
        
    return 

