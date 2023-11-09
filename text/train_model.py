#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


# In[2]:


gene_data= pd.read_excel('./train_data/gene.xlsx')
protein_data= pd.read_excel('./train_data/protein.xlsx')
others_data= pd.read_excel('./train_data/others.xlsx')


# In[3]:


gene_list= list(gene_data.iloc[:,0])
protein_list = list(protein_data.iloc[:,0])
others_list = [str(value) for value in list(others_data.iloc[:,0])]

print(len(gene_list), len(protein_list), len(others_list))

gene_list = list(set(gene_list))
protein_list = list(set(protein_list))
others_list = list(set(others_list))

print(len(gene_list), len(protein_list), len(others_list))


# In[4]:


gene_label=[[1,0,0] for i in range(len(gene_list))]
protein_label=[[0,1,0] for i in range(len(protein_list))]
others_label=[[0,0,1] for i in range(len(others_list))]


# In[5]:


data_x= gene_list+protein_list+others_list
data_y= torch.tensor(gene_label+protein_label+others_label)


# In[6]:


class MyBaseDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index]
        
    def __len__(self): 
        return len(self.x_data)

dataset = MyBaseDataset(data_x, data_y)

dataset_size = len(dataset)

train_size = int(dataset_size * 0.8)
validation_size = int(dataset_size * 0.1)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
train_dataloader=  DataLoader(train_dataset, batch_size=32, shuffle=True)


# In[7]:


tokenizer= AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2', do_lower_case=False)

class Classification_Model(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device= device
        self.bert_model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2').to(self.device)
        self.linear= nn.Linear(768,3).to(self.device)
        self.softmax= nn.Softmax()
    
    def encode_id(self,input):
        encoded_dict = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs= input,
            add_special_tokens = True,
            max_length = 20,
            pad_to_max_length = True,
            return_attention_mask = True
            )
        input_id= encoded_dict['input_ids']
        return torch.tensor(input_id)
    
    def Embedding(self,input):
        out= self.encode_id(input)
        out=self.bert_model(input_ids=out.type(torch.LongTensor).to(self.device),output_hidden_states=True)
        out= out['pooler_output']
        out=self.linear(out)
        out= self.softmax(out)
        return out
    


# In[8]:


device=torch.device(1 if torch.cuda.is_available() else "cpu")
device_next= torch.device("cpu")
model= Classification_Model(device)


# In[9]:


batch_size=32
learning_rate=1e-5
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
epochs=20
criterion= torch.nn.BCELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(epochs):
    epoch_loss=0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data 
        logit= model.Embedding(inputs)
        labels = labels.type(torch.FloatTensor)
        logit= logit.to(device_next)
        loss= criterion(logit,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
        if i%200==0:
            print(f'Epoch: {epoch+1}/{epochs} | Batch: {i+1}/{len(train_dataloader)} | Cost: {loss.item()}')
    print(f'Epoch: {epoch+1}/{epochs} | Cost: {epoch_loss/len(train_dataloader)}')
    scheduler.step()


# In[10]:


validation_dataloader=  DataLoader(validation_dataset, batch_size=32, shuffle=False)
logit_tensor= torch.empty((0,3))
label_tensor= torch.empty((0,3))
with torch.no_grad():
    model.eval()
    for x, y in validation_dataloader:
        inputs, labels = data 
        logit= model.Embedding(inputs)
        labels = labels.type(torch.FloatTensor)
        logit= logit.to(device_next)
        logit_tensor= torch.cat([logit_tensor,logit], dim=0)
        label_tensor= torch.cat([label_tensor,labels], dim=0)
output = torch.argmax(logit_tensor,dim=1)
output= F.one_hot(output,num_classes=3)
accuracy= accuracy_score(label_tensor,output)


# In[12]:


print(accuracy)


# In[13]:


torch.save(model.state_dict(), './text_classifier_model.pickle')


# In[ ]:




