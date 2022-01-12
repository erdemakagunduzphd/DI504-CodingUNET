from __future__ import print_function
from torch.utils.data import Dataset 

##############################################################################
class structureData(Dataset):
    def __init__(self,images,masks):
        
        self.images = images
        self.masks = masks
        
    def __getitem__(self, index):
        
        im = self.images[index]
        ma = self.masks[index]

        return im, ma
        
    def __len__(self):

        return len(self.images)

##############################################################################