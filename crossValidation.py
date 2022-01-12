from __future__ import print_function
import numpy as np
import os


def crossVal(N,fno,fsiz):   
    
    fno = fno - 1
    ind=[]
    
    with open(os.path.join(os.getcwd(),"crossVal", "randInd{}.txt".format(N))) as f:
        lines = f.readlines()
        for l in lines:
            ind.append(int(l))
    ind = np.asarray(ind)
    
    tsind = 0
    tstsize = int(N/fsiz)
    if (fno+1)*tstsize>N:        
        tsind1 = range((fno*tstsize)%N,N)
        tsind2 = range(0,((fno+1)*tstsize)%N)
        tsind = np.concatenate(tsind1,tsind2,0) 
    else: 
        tsind = range(fno*tstsize,(fno+1)*tstsize)        
        
    trvlind = np.setdiff1d(ind, tsind)
    
    # I've set validation size to 1/8 of the training set.
    valRatio = 0.125;    
    valsize = int((N-tstsize)*valRatio)
    
    vlind = trvlind[0:valsize]
    trind = trvlind[valsize:]

    trind = ind[trind]
    tsind = ind[tsind]
    vlind = ind[vlind]

    return tsind,trind,vlind

