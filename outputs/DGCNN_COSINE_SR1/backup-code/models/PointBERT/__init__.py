import torch
import torch.nn as nn
from .ppat import Projected, PointPatchTransformer

def PointBERT(sd):
        model =  Projected(
            PointPatchTransformer(512,12,8,512*3,256,384,0.2,64,6),
            nn.Linear(512, 1280)) #1024 input of the classification head
        
        model.load_state_dict(sd)
        return model

