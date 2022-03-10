#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:17:54 2022

@author: hoon
"""
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset


## Functions ##

def set_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
    

tokenizer_bert = BertTokenizer.from_pretrained("klue/bert-base")
def custom_collate_fn(batch):
  global tokenizer_bert
  
  input_list, target_list = [], []
  for (text, label) in batch:
    target_list.append(label)
    input_list.append(text)
    
  tensorized_input = tokenizer_bert(input_list, padding='longest', truncation='longest_first', return_tensors='pt')
  tensorized_label = torch.tensor(target_list)
  
  return tensorized_input, tensorized_label

## Classes ##

class CustomDataset(Dataset):
  def __init__(self, input_data:list, target_data:list) -> None:
    self.x = input_data
    self.y = target_data

  def __len__(self):
    return len(self.y)

  def __getitem__(self, index):
    # encode
    return self.x[index], self.y[index]

class CustomClassifier(nn.Module):

  def __init__(self, hidden_size: int, n_label: int):
    super(CustomClassifier, self).__init__()
    self.bert = BertModel.from_pretrained("klue/bert-base")

    dropout_rate = 0.1
    linear_layer_hidden_size = 32

    self.classifier = nn.Sequential(
        nn.Linear(hidden_size, linear_layer_hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(linear_layer_hidden_size, 2),
    )
  
  def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    cls_token_last_hidden_states = outputs['pooler_output']
    logits = self.classifier(cls_token_last_hidden_states)
    return logits

'''
### train result ###
    Step : 10, Avg Loss : 0.6644
    Step : 20, Avg Loss : 0.6106
    Step : 30, Avg Loss : 0.4884
    Step : 40, Avg Loss : 0.4894
    Step : 50, Avg Loss : 0.4313
    Step : 60, Avg Loss : 0.3818
    Step : 70, Avg Loss : 0.3995
    Step : 80, Avg Loss : 0.3564
    Step : 90, Avg Loss : 0.3665
    Step : 100, Avg Loss : 0.3540
    Step : 110, Avg Loss : 0.4066
    Step : 120, Avg Loss : 0.4102
    Step : 130, Avg Loss : 0.4146
    Step : 140, Avg Loss : 0.3546
    Step : 150, Avg Loss : 0.3154
    Step : 160, Avg Loss : 0.3607
    Step : 170, Avg Loss : 0.3950
    Step : 180, Avg Loss : 0.3381
    Step : 190, Avg Loss : 0.3304
    Step : 200, Avg Loss : 0.3800
    Step : 210, Avg Loss : 0.3500
    Step : 220, Avg Loss : 0.3409
    Step : 230, Avg Loss : 0.3217
    Step : 240, Avg Loss : 0.3557
    Step : 250, Avg Loss : 0.3510
    Step : 260, Avg Loss : 0.3289
    Step : 270, Avg Loss : 0.3557
    Step : 280, Avg Loss : 0.3114
    Mean Loss : 0.3925
    Train Finished
'''