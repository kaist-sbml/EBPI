
import argparse
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
from earlystopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df = pd.read_csv('/data/user_home/dlwnsrb/dataset/final_label.csv')

learning_rate= 1e-3
weight_decay_parameters= [0.001,0.0001]
max_norm_parameters=[1,5]

def get_args_parser():
    parser = argparse.ArgumentParser('fasterrcnn training', add_help=False)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--batch_size', default=8, type=int)
    return parser

parser = argparse.ArgumentParser('fasterrcnn training', parents=[get_args_parser()])
args = parser.parse_args()
device = torch.device(int(args.device) if torch.cuda.is_available() else "cpu")

class Dataset_make(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        super().__init__()
        
        self.image_ids = df["image"].unique() # all image filenames
        self.df = df
        self.image_dir = image_dir # dir to image files
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        records = self.df[self.df["image"] == image_id]
        image = cv2.imread(f"{self.image_dir}/{image_id}", cv2.IMREAD_COLOR)
        image = cv2.resize(image,(800,800))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        imgTensor = T.ToTensor()(image)
        transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image= transform(imgTensor)
        
        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        
        labels = records[["labels"]].values
        
        target = {}
        target["boxes"] = torch.tensor(boxes)
        target["labels"] = torch.tensor(labels).squeeze(1)
        target["image_id"] = torch.tensor([idx])
        target["area"] = area


        if self.transforms:
            sample = {"image": image, "boxes": target["boxes"], "labels": labels}
            sample = self.transforms(**sample)
            image = sample["image"]
            target["boxes"] = torch.stack(tuple(map(torch.tensor, zip(*sample["boxes"])))).permute(1, 0)

        return image, target, image_id

    def __len__(self):
        return self.image_ids.shape[0]

def collate_fn(batch):
    return tuple(zip(*batch))

dir_train= '/data/user_home/dlwnsrb/dataset/final_image'
dataset = Dataset_make(df, dir_train)
train_size= int(0.9*len(dataset))
validation_size= len(dataset)-train_size
train_dataset, validation_dataset= random_split(dataset, [train_size, validation_size])
train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_dl = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

for weight_decay_parameter in weight_decay_parameters:
    for max_norm_parameter in max_norm_parameters:
        dir_name= 'total_result/'+ str(learning_rate)+'_'+str(weight_decay_parameter)+'_'+str(max_norm_parameter)+'_'+str(args.batch_size)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        f=open(dir_name+'/result.txt','w')
        model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay= weight_decay_parameter)
        model.to(device)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        model.train()
        num_epochs = 50
        loss_train_iter=[]
        loss_train_epoch= []
        loss_val_epoch=[]

        es = EarlyStopping(patience=10, 
                           delta=0, 
                           mode='min', 
                           verbose=True
                          )
        for epoch in range(num_epochs):
            epoch_loss=0
            model.train()
            for i, (images, targets, image_ids) in enumerate(train_dl):
                optimizer.zero_grad()
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm = max_norm_parameter)
                optimizer.step()
                epoch_loss+= losses
                loss_train_iter.append(round(float(losses.detach().cpu().numpy()),4))
                if (i) % 200==0:
                    logging.info(f'Epoch {epoch+1} - Iteration: {i}/{len(train_dl)}, Total: {losses:.4f}, Regression: {loss_dict["loss_box_reg"]:.4f}, Classifier: {loss_dict["loss_classifier"]:.4f}')
            logging.info("Epoch_loss:",epoch_loss/len(train_dl))
            scheduler.step()  
            
            
            with torch.no_grad():
                val_loss=0
                for i, (images, targets, image_ids) in enumerate(val_dl):
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss+= losses
                logging.info('validation loss is: '+str(val_loss/len(val_dl)))

            pickle_name= str(learning_rate)+'_'+str(weight_decay_parameter)+'_'+str(max_norm_parameter)+'.pickle'

            epoch_training_loss= round(float(epoch_loss.detach().cpu().numpy()/len(train_dl)),4)
            epoch_val_loss= round(float(val_loss.detach().cpu().numpy()/len(val_dl)),4)
            es(epoch_val_loss)
            loss_train_epoch.append(epoch_training_loss)
            loss_val_epoch.append(epoch_val_loss)
            f.write('EPOCH: '+str(epoch)+'\t'+'training_loss: '+str(epoch_training_loss)+'\t'+ 'validation_loss: '+str(epoch_val_loss)+'\n')
            
            if es.store:
                torch.save(model.state_dict(), dir_name+'/'+pickle_name)
            elif es.early_stop:
                break
        
        f.close()
        
        plt.plot(loss_train_epoch)
        plt.plot(loss_val_epoch)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(dir_name+'/train_val.png')
        
        plt.cla()
        plt.clf()
        
        plt.plot(loss_train_iter)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig(dir_name+'/train_iter.png')
        
        plt.cla()
        plt.clf()
