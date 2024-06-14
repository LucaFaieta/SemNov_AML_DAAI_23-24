import torch
import torch.nn as nn
from .ppat import Projected, PointPatchTransformer

def PointBERTG14():
        model =  Projected(
            PointPatchTransformer(512,12,8,512*3,256,384,0.2,64,6),
            nn.Linear(512, 1280)) #1024 input of the classification head
        return model

def PointBERTL14():
    model = Projected(
        PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6),
        nn.Linear(512, 768)
    )
    return model


def PointBERTB32():
    model = PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6)
    return model
