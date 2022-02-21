import torch
from torch import nn
import torch.nn.functional as F

def make_head():
    from torchvision.models import squeezenet1_1  
    base = squeezenet1_1(pretrained=True)
    features = base.features
    return features

def make_classifier(n_inputs, n_outputs):
    final_conv = nn.Conv2d(n_inputs, n_outputs*2, kernel_size=1)
    classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        final_conv,
        nn.LeakyReLU(0.1, inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1),
        nn.Linear(n_outputs*2, n_outputs),
    )
    return classifier
    

class PoseExtractor(nn.Module):
    
    def __init__(self, n_outputs):
        super().__init__()
        self.lowResHead = make_head()
        self.final = make_classifier(512, n_outputs)
        self.parameterNames = "Yaw Pitch Raw".split()
    
    def forward(self, x):
        x = F.interpolate(x, (256, 256), mode="bicubic", align_corners=False)
        low_res = self.lowResHead(x)
        res = self.final(low_res)
        return res
    
    def enableGradients(self, status):
        self.lowResHead.requires_grad_(status)
