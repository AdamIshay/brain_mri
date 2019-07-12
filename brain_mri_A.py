# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 04:21:30 2019

@author: Adam
"""



import os
import numpy as np
import matplotlib.pyplot as plt
import sys 

from PIL import Image 
from torchvision import transforms
from sklearn.model_selection import ShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


from torchvision import transforms
import torchvision

from data_preprocessing_A import *


def train_DQN(model,train_loader, validation_loader, criterion, epochs,batch_size,opt):
    #breakpoint()
    opt.zero_grad()

    loss_hist=[]
    #mean_hist=[]
    sample_hist=[]

    for epoch in range(epochs):  
        #n_of_batches=agent_data.len//batch_size
        model.train()
        acc=0
        correct=0.
        total=0.
        for i_batch,batch in enumerate(train_loader):
            
            if i_batch==-1:
                breakpoint()
             
            opt.zero_grad()

            x=batch[0].float().cuda()
            labels=batch[1].unsqueeze(1).float().cuda()
            outputs=model.forward(x.permute(0,3,1,2))
            outputs=torch.nn.functional.sigmoid(outputs)
            
            correct+=(outputs.round()==labels).sum()
            total+=labels.shape[0]
            
            loss = criterion(outputs,labels)
            loss.backward()
            #breakpoint()
            loss_hist.append(loss.data.item())
            #print('{}/{}    {} %'.format(correct.item(),total,correct.item()/total))
            #print(loss_hist[-1])
        
            opt.step()
        print('Training Accuracy{}/{}    {} %'.format(correct.item(),total,100*correct.item()/total))
        
        model.eval()
        acc=0
        correct=0.
        total=0.
        for i_batch,batch in enumerate(validation_loader):
            
            if i_batch==-1:
                breakpoint()
             
            opt.zero_grad()

            x=batch[0].float().cuda()
            labels=batch[1].unsqueeze(1).float().cuda()
            outputs=model.forward(x.permute(0,3,1,2))
            outputs=torch.nn.functional.sigmoid(outputs)
            
            correct+=(outputs.round()==labels).sum()
            total+=labels.shape[0]
            
            loss = criterion(outputs,labels)
            #breakpoint()
            loss_hist.append(loss.data.item())
            #print('{}/{}    {} %'.format(correct.item(),total,correct.item()/total))
            #print(loss_hist[-1])
        
            opt.step()
        
        print('Validation Accuracy {}/{}    {} %'.format(correct.item(),total,100*correct.item()/total))
        
        
        
        
    return loss_hist,sample_hist



def evaluate(model,valid_loader,batch_size):
    #breakpoint()
    opt.zero_grad()

    loss_hist=[]
    #mean_hist=[]
    sample_hist=[]
        
    model.eval()
    acc=0
    correct=0.
    total=0.
        

            
    for i_batch,batch in enumerate(validation_loader):
    
        if i_batch==-1:
            breakpoint()
         

        x=batch[0].float().cuda()
        labels=batch[1].unsqueeze(1).float().cuda()
        outputs=model.forward(x.permute(0,3,1,2))
        outputs=torch.nn.functional.sigmoid(outputs)
        
        correct+=(outputs.round()==labels).sum()
        total+=labels.shape[0]
        
        loss = criterion(outputs,labels)
        #breakpoint()
        loss_hist.append(loss.data.item())
        #print('{}/{}    {} %'.format(correct.item(),total,correct.item()/total))
        #print(loss_hist[-1])
    

    acc=100*correct.item()/total
    #print('Validation Accuracy {}/{}    {} %'.format(correct.item(),total,acc))
        
        
        
        
    return acc






preprocessing = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip()#,
    #transforms.ToTensor()
])


    
    
path='C:/Users/Adam/Python_stuff/brain_mri/data_all/'
train_folder_name='224_by_224'
    
train_dataset=Data(path=path,folder_name=train_folder_name,seed=None, transform=preprocessing)
validation_dataset=Data(path=path,folder_name=train_folder_name,seed=None, transform=preprocessing)


learning_rate=0.0001
epochs=20
batch_size=20
criterion=torch.nn.BCELoss()



train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)




#model = torchvision.models.vgg19(pretrained=True)
model = torchvision.models.resnet34(pretrained=True)

model.fc = torch.nn.Linear(512, 1)

model.cuda();



opt=optim.Adam(model.parameters(),lr=learning_rate)
train_DQN(model,train_loader, validation_loader, criterion, epochs,batch_size,opt)



accuracy=[]

names=os.listdir('data_all')
n_of_folders=len(names)

path='C:/Users/Adam/Python_stuff/brain_mri/data_all/'

for i,name in enumerate(os.listdir('data_all')):
    #breakpoint()
    yes_path=path+names[i]+'/yes/'
    no_path=path+names[i]+'/no/'
    
    
    validation_dataset=Data(path=path,folder_name=name,seed=None, transform=preprocessing)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    
    accuracy.append(evaluate(model,validation_loader,batch_size))
    
    print('{} was achieved on the "{}" dataset'.format(accuracy[-1],name))
    
    





# =============================================================================
# rows = zip(all_acc[0],all_acc[1],all_acc[2],all_acc[3],all_acc[4],all_acc[5])
# 
# 
# import csv
# 
# with open('results_csv', "w") as f:
#     writer = csv.writer(f)
#     for row in rows:
#         writer.writerow(row)
# 
# 
# =============================================================================

