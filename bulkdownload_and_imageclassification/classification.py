#!/usr/bin/env python
# coding: utf-8

# In[2]:


from efficientnet_pytorch import EfficientNet
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
import os
import random
import shutil

def processing(metabolite, device):
    abs_path= os.path.dirname(__file__)
    model_name = 'efficientnet-b0'
    model = EfficientNet.from_pretrained(model_name,num_classes=2)
    model.load_state_dict(torch.load(abs_path+'/model/image_classification.pt',map_location='cpu'))
    model.to(device)
    dataset = datasets.ImageFolder(abs_path+'/output_file/'+ metabolite,
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
    
    return model, allFiles, dataloaders

def classification(metabolite, device):
    batch_size  = 4
    abs_path= os.path.dirname(__file__)
    device=  torch.device(device if torch.cuda.is_available() else "cpu")
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
    if not os.path.exists(os.path.abspath(os.path.join(abs_path, os.pardir))+'/input'):
        os.mkdir(os.path.abspath(os.path.join(abs_path, os.pardir))+'/input')
        
    for name,label in total_result.items():
        revise_name= name.split('/')[-1]
        if label==0:
            os.remove(name)
        else:
            shutil.move(name, os.path.abspath(os.path.join(abs_path, os.pardir))+'/input/'+revise_name)
        
    return 