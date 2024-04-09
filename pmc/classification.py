import numpy as np
import os
import random
import shutil
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms, datasets

abs_path= os.path.dirname(os.path.abspath(__file__))
Image.MAX_IMAGE_PIXELS = None

def processing(metabolite, device):
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


def classification(args, bulkdownload_result):
    if bulkdownload_result != []:
        batch_size = 4
        device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
        model, allFiles, dataloader= processing(args.metabolite, device)
        model.eval()
        total_result=dict()
        for i, (inputs, _) in enumerate(dataloader):
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            preds= list(preds.detach().cpu().numpy())
            for j in range(inputs.size()[0]):
                image_name = allFiles[i*batch_size +j]
                total_result[image_name] = preds[j]
        if not os.path.exists(os.path.abspath(os.path.join(abs_path, os.pardir))+ '/' + args.input):
            os.mkdir(os.path.abspath(os.path.join(abs_path, os.pardir))+ '/' + args.input)

        for name,label in total_result.items():
            revise_name = name.split('/')[-1]
            #if label == 0:
            #    os.remove(name)
            #else:
            if label == 1:
                print(
                #shutil.move(name, os.path.abspath(os.path.join(abs_path, os.pardir))+ '/' + args.input + '/' + revise_name)
                shutil.copy(name, os.path.abspath(os.path.join(abs_path, os.pardir))+ '/' + args.input + '/' + revise_name))
                
        return True
    else:
        return False
    #shutil.rmtree(abs_path+'/output_file/')