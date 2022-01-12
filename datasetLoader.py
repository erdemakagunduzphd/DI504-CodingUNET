from __future__ import print_function
import os
import torch 
import numpy as np
import scipy.io as sio

def get_images(trainSetSize, tsind, trind, vlind):
    input_images=[]
    target_masks=[]    
    gettingfiles=[]
    
    names=os.listdir('RGBs')
    for b in names[0:trainSetSize]:
        gettingfiles.append(b)
        a = sio.loadmat('RGBs/{}'.format(b))
        a = a['inputPatch']
        input_images.append(a)
        c=sio.loadmat('labels/{}'.format(b))
        c = c['inputPatch']
        target_masks.append(c)
  
    input_images = np.asarray(input_images, dtype=np.float32)
    target_masks = np.asarray(target_masks, dtype=np.float32)
    lim=224
    input_images = np.reshape(input_images[0:trainSetSize*lim*lim], (trainSetSize, lim, lim, 3)) 
    input_images = np.moveaxis(input_images,3,1)
    target_masks = np.reshape(target_masks[0:trainSetSize*lim*lim], (trainSetSize, 1, lim, lim)) 
    
    trMeanR = input_images[trind,0,:,:].mean()
    trMeanG = input_images[trind,1,:,:].mean()
    trMeanB = input_images[trind,2,:,:].mean()
    
    input_images[:,0,:,:] = input_images[:,0,:,:] - trMeanR
    input_images[:,1,:,:] = input_images[:,1,:,:] - trMeanG
    input_images[:,2,:,:] = input_images[:,2,:,:] - trMeanB
    
    input_images=torch.from_numpy(input_images)
    target_masks=torch.from_numpy(target_masks)
    
    return input_images, target_masks