from torch import nn, optim
import torch
import numpy as np
from torch.utils import data
from styleflow.flow import cnf
from math import log, pi
import os
from tqdm import tqdm

import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as dset
import argparse


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2


class MyDataset(Dataset):
    def __init__(self, latents, attributes, transform=None):
        self.latents = latents
        self.attributes = attributes
        self.transform = transform

    def __getitem__(self, index):
        x = self.latents[index]
        y = self.attributes[index]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def __len__(self):
        return len(self.latents)

def read_annotations(annotations_file, annotations):
    ann = open(annotations_file, "r")
    n_lines = int(ann.readline())
    header = ann.readline().strip().split()
    for _ in range(n_lines):
        l = ann.readline().strip().split()
        f_name = l[0]
        fields = l[1:]
        if f_name not in annotations:
            annotations[f_name] = {}
        for field, val in zip(header, fields):
            annotations[f_name][field] = float(val)

FIELDS = [  "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Chubby",
            "Double_Chin", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Receding_Hairline",
            "Rosy_Cheeks", "Sideburns", "Smiling", "Wavy_Hair", "Yaw", "Pitch", "Raw"
         ]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="StyleFlow trainer")

    parser.add_argument("--latents_dir", default='/home/user/ML/GAN/diffusion_distiller/data/celeba_128m_latents/', type=str, help="path to the latents")
    parser.add_argument("--attributes_file", default="/media/user/Datasets/CelebAMask-HQ_/CelebAMask-HQ-attribute-anno.txt", type=str)
    parser.add_argument("--pose_file", default="/media/user/Datasets/CelebAMask-HQ_/CelebAMask-HQ-pose-anno.txt", type=str)
    parser.add_argument(
        "--batch", type=int, default=3, help="batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs"
    )

    parser.add_argument("--flow_modules", type=str, default='512-512-512-512-512')
    parser.add_argument("--cond_size", type=int, default=29)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--name", type=str, default="styleflow")
    parser.add_argument("--start_epoch", type=int, default=0)

    args = parser.parse_args()
    torch.manual_seed(0)
    print(f"Loading data...")
    annotations = {}
    read_annotations(args.attributes_file, annotations)
    read_annotations(args.pose_file, annotations)

    sg_latents = []
    sg_attributes = []
    for f_name in os.listdir(args.latents_dir):
        f_path = os.path.join(args.latents_dir, f_name)
        D = torch.load(f_path, map_location=torch.device("cpu"))
        latents = D["latents"][0]
        attrs = []
        file_attrs = annotations[f_name[:-3]]
        for f in FIELDS:
            attrs.append(file_attrs[f])

        sg_latents.append(latents)
        sg_attributes.append(torch.FloatTensor(attrs))
#    print(f"Attributes count {len(header)}")

    sg_latents = torch.stack(sg_latents, 0)[:, None, None, ...]
    sg_attributes = torch.stack(sg_attributes, 0).unsqueeze(-1)

    my_dataset = MyDataset(latents=sg_latents.cuda(), attributes=sg_attributes.float().cuda())
    train_loader = data.DataLoader(my_dataset, shuffle=False, batch_size=args.batch)

    print(f"Training {args.name}...")

    prior = cnf(512, args.flow_modules, args.cond_size, 1)
    out_dir = f"checkpoints/{args.name}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    optimizer = optim.Adam(prior.parameters(), lr=args.lr)


    for epoch in range(args.epochs):
        pbar = tqdm(train_loader)
        i = -1
        for x in pbar:
            i += 1
            a, b = x

            approx21, delta_log_p2 = prior(x[0].squeeze(1), x[1], torch.zeros(args.batch, x[0].shape[2], 1).to(x[0]))

            approx2 = standard_normal_logprob(approx21).view(args.batch, -1).sum(1, keepdim=True)

            delta_log_p2 = delta_log_p2.view(args.batch, x[0].shape[2], 1).sum(1)
            log_p2 = (approx2 - delta_log_p2)

            loss = -log_p2.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                f'logP: {loss:.5f}')

            if i % 1000 == 0:
                torch.save(
                    prior.state_dict(), os.path.join(out_dir, f'{str(i).zfill(6)}_{str(epoch).zfill(2)}.pt')
                )