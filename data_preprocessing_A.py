# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:55:55 2019

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

from torch.utils.data import Dataset, DataLoader


from torchvision import transforms



class Data(Dataset):
    def __init__(self,path,folder_name,seed=None, transform=None):

        self.transform = transform
        self.seed=seed
        
        self.path=path
        
        self.yes_path=self.path+folder_name+'/yes/'
        self.no_path=self.path+folder_name+'/no/'
        
        
        self.yes,self.no=self.read_files(self.yes_path,self.no_path)
        #self.training,self.validation=self.create_sets()
        self.train_indexes,self.validation_indexes=self.create_sets()

    def __len__(self):
        return (len(self.yes)+len(self.no))
    
    def __getitem__(self, idx):
        #breakpoint()
        image=self.X_unshuffled[idx]
        label=self.Y_unshuffled[idx]
    
        if self.transform:
            image = self.transform(image)
            image = np.asarray(image)
        return [image, label]
    
    
    def create_sets(self):
        X=self.yes+self.no
        self.X_unshuffled=np.array(X)
        y_yes=np.ones(len(self.yes))
        y_no=np.zeros(len(self.no))
        
        self.Y_unshuffled=np.concatenate((y_yes,y_no))
        
        
        self.rs = ShuffleSplit(n_splits=1, test_size=.1, random_state=self.seed)
        self.rs.get_n_splits(self.X_unshuffled)
        
        
        train_ind=[];val_ind=[];
        for train_index,test_index in self.rs.split(self.X_unshuffled):
            #print("TRAIN:", train_index, "TEST:", test_index)
            train_ind.append(train_index);
            val_ind.append(test_index);
        
        
        
        train_unique=np.unique(self.Y_unshuffled[train_ind],return_counts=True)
        validation_unique= np.unique(self.Y_unshuffled[val_ind],return_counts=True)
        
        #breakpoint()
        #print("{} negative and {} positive in the training set".format(train_unique[1][0],train_unique[1][1]))
        #print("{} negative and {} positive in the validation set".format(validation_unique[1][0],validation_unique[1][1]))
        
        return train_ind,val_ind

    def read_files(self,yes_path,no_path):
    #returns the yes and no images, where they all have RGB channels
        
        yes_files=os.listdir(yes_path)
        no_files=os.listdir(no_path)
        
        yes_pil=[];
        no_pil=[];
        
        
        
        for file in yes_files:
            im=Image.open(yes_path+file).convert('RGB')
            np_im=np.array(im);
            if len(np_im.shape)<3:
                print('hiiiiiiiii')
                np_im=np_im[...,np.newaxis]
                np_im=np.concatenate((np_im,np_im,np_im),axis=2)
            yes_pil.append(np_im)
            assert yes_pil[-1].shape[2]==3
            
        
        for file in no_files:
            im=Image.open(no_path+file).convert('RGB')
            np_im=np.array(im);
            #print(np_im.mean());
            if len(np_im.shape)<3:
                print('hiiiiiiiii')
                np_im=np_im[...,np.newaxis]
                np_im=np.concatenate((np_im,np_im,np_im),axis=2)
            no_pil.append(np_im)
            if no_pil[-1].shape[2]!=3:
                breakpoint()
                pass;
        #breakpoint()
        return yes_pil,no_pil



