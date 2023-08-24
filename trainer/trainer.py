#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd


# In[3]:


cudnn.benchmark = True
cudnn.deterministic = False


# In[4]:


def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


# In[5]:


opt = get_config("config_files/train.yaml")


import torch

# Create a device object to represent the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if torch.cuda.is_available():
    
    print("CUDA is available")
else:
    print("CUDA is not available")


train(opt, amp=True).to(device)


print("finish")

