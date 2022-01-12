from __future__ import print_function
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from crossValidation import crossVal
from datasetTorch import structureData
from trainModel import train_model
from testModel import test_model
from datasetLoader import get_images
from unetArchitecture import UNet

import warnings
warnings.filterwarnings("ignore")


##############################################################################   
if __name__ == '__main__':
    
    # first if I have GPU (I dont)
    if (torch.cuda.is_available()):
        print(torch.cuda.get_device_name(0))
    else:   
        print("no GPU found")        
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    # tell pyTorch to use the available device (which is CPU in my case)
    device = torch.device(dev)
        

    # some parameters for my experiments
    trainSetSize=16         #all set train+val+test = 16 - toy example
    fno = 1                 # I will be using N-fold cross validation
    fsiz = 4                # N = 4, and this will we the first fold.

    miniBatchSize = 16      # 16 batches per iter, 16/16=1 iter per epoch
    n_epochs = 10           # go over the entire train set 10 times (1*10 iter)
    learnRate = 0.001       # learning rate 10^-3
    step_size=5             # LR drop every 5 epochs by...
    gamma=0.9               # 0.9 --> start 10^-3 --> 0.9x10^-3 -->...
    imgSize=224             # sizes of image in my set (NxN square assumption)
        

    # aply cross validation and get file indices for train,test and val sets
    tsind,trind,vlind = crossVal(trainSetSize,fno,fsiz) 
    input_images, target_masks= get_images(trainSetSize, tsind, trind, vlind)
         
    # convert the images in the memory in to pyTorch DataLoader objects 
    # first you need the parameters object, bc it tells us the batch size
    # pyTorch object will arrange the data in batches
    params = {'batch_size': miniBatchSize, 'shuffle': False}   
    training_set = structureData(input_images[trind], target_masks[trind])
    trainingLoader = DataLoader(training_set, **params)    
    validation_set = structureData(input_images[vlind], target_masks[vlind])
    validationLoader = DataLoader(validation_set, **params)    
    test_set = structureData(input_images[tsind], target_masks[tsind])
    testLoader = DataLoader(test_set, **params)
        
    # create the model
    model = UNet(classes=1).to(device)
    
    # initizalize the conv layers by Xavier method ###########################        
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)   
    ##########################################################################
    model.apply(init_weights)                  
    #or maybe load an previous pre-trained model if you want to
    #model.load_state_dict(torch.load(os.path.join(data_folder, "finalModel.pt")))        

    # create the solver with the given parameters              
    optim = torch.optim.Adam(model.parameters(),learnRate)        
     
    # a pyTorch scheduler object for periodic LR drop
    scheduler = StepLR(optim, step_size, gamma)
    
    # I will saving everythin to this working folder
    pathm = os.getcwd()    
    
    # train the model for the given number of epochs
    train_model(n_epochs, model, scheduler, trainingLoader, optim, imgSize, validationLoader, pathm)        
    
    # test the final model with the test set
    test_model(testLoader, imgSize, pathm)  
    
    # delete the model and clear GPU memory (if there is one)
    del model
    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()

    
        

  
