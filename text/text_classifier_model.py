import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import logging

parent_dir= os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

class Classification_Model(nn.Module):
    def __init__(self, bert_model, device):
        super(Classification_Model,self).__init__()
        self.bert_model = bert_model
        self.softmax= nn.Softmax()
        self.device= device
        self.linear= nn.Linear(768,3).to(self.device)
        
    def Embedding(self,input):
        out=self.bert_model(input_ids=input,output_hidden_states=True)
        out= out['pooler_output']
        out=self.linear(out)
        out= self.softmax(out)
        return out

def text_classifier(args, ocr_not_contained_name):
    checkpoint_bert = parent_dir + '/text/model/text_classifier_model.pickle'
    tokenizer= AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2', do_lower_case=False, truncation=True)
    encoded_dict = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs= ocr_not_contained_name,
            add_special_tokens = True,
            max_length = 20,
            pad_to_max_length = True,
            return_attention_mask = True
            )
    device=torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    bert_model= AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2').to(device)
    input_ids = encoded_dict['input_ids']
    model= Classification_Model(bert_model,device)
    model.load_state_dict(torch.load(checkpoint_bert, map_location=device), strict= False)
    logit= model.Embedding(torch.tensor(input_ids).to(device))

    return logit.cpu()