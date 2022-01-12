from __future__ import print_function
import os
import torch 
import torch.nn as nn
import numpy as np
from jaccardIndex import Jaccard

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev) 

def train_model(n_epochs, model, scheduler, trainingLoader, optim, imgSize, validationLoader, pathm):
    training_losses = []
    for epoch in range(n_epochs):
        model.train()
        batch_losses = []
        jI = 0
        totalBatches = 0
        scheduler.step()
        #print('Epoch:', epoch,'LR:', scheduler.get_lr())

        mb=0
        for trainim, trainmas in trainingLoader:
            mb+=1
            optim.zero_grad()
            images=trainim.to(device)
            masks=trainmas.to(device)
            outputs=model(images)
            loss=nn.BCEWithLogitsLoss()
            output = loss(outputs, masks)            
            output.backward()
            optim.step()
            
            batch_losses.append(output.item())
            batchLoad = len(masks)*imgSize*imgSize
            totalBatches = totalBatches + batchLoad
            thisJac = Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
            jI = jI+thisJac.data[0]
                       
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)
        print("Training Jaccard:",(jI/totalBatches).item()," (epoch:",epoch,")")
        print("Training Loss:",training_losses[epoch]," (epoch:",epoch,")")

                
        validate(model, validationLoader, imgSize, pathm)
        
    torch.save(model.state_dict(), os.path.join(pathm, "finalModel.pt"))        
        
                
        
def validate(model, validationLoader, imgSize, pathm):
    jI = 0
    totalBatches = 0
    validation_losses = []
    
    model.eval()
    with torch.no_grad():
        val_losses = []
        for vaimgSize, valmas in validationLoader:
            #model.eval()
            images=vaimgSize.to(device)
            masks=valmas.to(device)
            outputs=model(images)
            loss=nn.BCEWithLogitsLoss()
            output = loss(outputs, masks)
            val_losses.append(output.item())
            batchLoad = len(masks)*imgSize*imgSize
            totalBatches = totalBatches + batchLoad        
            thisJac = Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
            jI = jI+thisJac.data[0] 
    dn=jI/totalBatches
    dni=dn.item()
    validation_loss = np.mean(val_losses)
    validation_losses.append(validation_loss)
    print("Validation Jaccard:",dni)
