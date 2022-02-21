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
    

class AttributesExtractor(nn.Module):
    
    def __init__(self, n_outputs):
        super().__init__()
        self.lowResHead = make_head()
        self.highResHead = make_head()
        self.final = make_classifier(512*2, n_outputs)
        self.attributeNames = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young".split()
    
    def forward(self, x):
        high_res = self.highResHead(x)
        x = F.interpolate(x, (256, 256), mode="bicubic", align_corners=False)
        low_res = self.lowResHead(x)
        low_res = F.interpolate(low_res, (high_res.shape[2], high_res.shape[3]), mode="bilinear", align_corners=False)
        res = self.final(torch.cat([high_res, low_res], 1))
        return res
    
    def enableGradients(self, status):
        self.lowResHead.requires_grad_(status)
        self.highResHead.requires_grad_(status)

 
