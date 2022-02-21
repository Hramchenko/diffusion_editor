import torch
from torchvision import transforms
import numpy as np
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
  ])

def load_face_attributes_extractor(device, model_path="./checkpoints/celeba_face_attributes/celeba.pt"):
    from .attributes_model import AttributesExtractor
    net = AttributesExtractor(40)
    ckpt = model_path
    ckpt = torch.load(ckpt, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    net = net.to(device)
    return net
  
def apply_face_attributes_extractor(face_attributes_extractor, device, img):
    if (img.shape[0] != 1024) or (img.shape[1] != 1024):
      raise Exception()
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
      res = face_attributes_extractor(img_tensor)
    return res[0].cpu().numpy()
  
def load_face_pose_estimator(device):
    from .pose_model import PoseExtractor
    net = PoseExtractor(3).to(device)
    ckpt = "./checkpoints/celeba_face_attributes/celeba_pose.pt"
    ckpt = torch.load(ckpt, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net
  
def apply_face_pose_estimator(face_pose_estimator, device, img):
    if (img.shape[0] != 1024) or (img.shape[1] != 1024):
      raise Exception()
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
      res = face_pose_estimator(img_tensor)
    return res[0].cpu().numpy()
