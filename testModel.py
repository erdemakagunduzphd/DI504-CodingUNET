from __future__ import print_function
import torch 
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from jaccardIndex import Jaccard
from unetArchitecture import UNet


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(dev) 

def test_model(test_generator, imgSize, pathm):    
    net = UNet(classes=1).to(device) 
    net.load_state_dict(torch.load(os.path.join(pathm, "finalModel.pt")))

    jI = 0
    totalBatches = 0
    test_losses = []
    net.eval()
    with torch.no_grad():
        t_losses = []
        t=0
        for testim, testmas in test_generator:
            images=testim.to(device)
            masks=testmas.to(device)
            outputs = net(images)
            losst=nn.BCEWithLogitsLoss()
            output = losst(outputs, masks)
            t_losses.append(output.item())
            batchLoad = len(masks)*imgSize*imgSize
            totalBatches = totalBatches + batchLoad
            thisJac = Jaccard(torch.reshape(masks,(batchLoad,1)),torch.reshape(outputs,(batchLoad,1)))*batchLoad
            jI = jI+thisJac.data[0]
            t+=1
                 
    dn=jI/totalBatches
    dni=dn.item()
    test_loss = np.mean(t_losses)
    test_losses.append(test_loss)
    print("Test Jaccard:",dni)
