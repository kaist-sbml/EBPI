#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import itertools
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T

parent_dir= os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def bounding_box_intersection(box1, box2): 

    a, b = box1, box2
    
    min_dist_x= min(a[2]-a[0],b[2]-b[0])
    min_dist_y= min(a[3]-a[1],b[3]-b[1])
    
    
    startX = max( min(a[0], a[2]), min(b[0], b[2]) )
    startY = max( min(a[1], a[3]), min(b[1], b[3]) )
    endX = min( max(a[0], a[2]), max(b[0], b[2]) )
    endY = min( max(a[1], a[3]), max(b[1], b[3]) )
    
    intersection_area= (endX-startX) * (endY-startY)
    box1_area= (a[2]-a[0]) * (a[3]-a[1])
    box2_area= (b[2]-b[0]) * (b[3]-b[1])
    min_area= min(box1_area, box2_area)
    
    if startX <= endX+1 and startY <= endY+1:
        if intersection_area/min_area>0.8:
            return True
        else:
            return False
    else:
        return False

def combineRect(rectA, rectB):
    a, b = rectA, rectB
    startX = min( a[0], b[0] )
    startY = min( a[1], b[1] )
    endX = max( a[2], b[2] )
    endY = max( a[3], b[3] )

    return (startX, startY, endX, endY)
    
    
def arrow_distinguish_test(threshold, test_image_path, device):
    checkpoint = parent_dir + '/arrow/model/checkpoint.pickle'
    cpu_device = torch.device("cpu")
    num_classes=2
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.to(device)
    model.eval()
    try:
        images = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
        resize_img = cv2.resize(images, (800, 800))
        image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        imgTensor = T.ToTensor()(image)
        transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        images= transform(imgTensor)
        images= torch.unsqueeze(images,0)
        images = images.to(device)
        cpu_device = torch.device("cpu")
        outputs = model(images)
        images= images.to("cpu")
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        mask = outputs[0]['scores']>threshold
        boxes = outputs[0]["boxes"][mask].detach().numpy().astype(np.int32)
        newboxes=[]
        for box in boxes:
            newboxes.append([box[0],box[1],box[2],box[3]]) 
        noIntersect = False
        while noIntersect == False: 
            newRectsArray = []
            for rectA, rectB in itertools.combinations(newboxes, 2):
                if bounding_box_intersection(rectA, rectB):
                    newRect = combineRect(rectA, rectB)
                    newRectsArray.append(newRect)
                    if rectA in newboxes:
                        newboxes.remove(rectA)
                    if rectB in newboxes:
                        newboxes.remove(rectB)
                    break
            if len(newRectsArray) == 0:
                noIntersect = True   
            else:
                newboxes+= newRectsArray
        return newboxes
    except:
        print('Cannot distinguish image')