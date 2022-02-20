import argparse
import importlib
from v_diffusion import make_beta_schedule
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
import os
import cv2
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import glob
from sample import calc_latents
from tqdm import tqdm

def process_images(args, make_model):
    device = torch.device("cuda")

    teacher_ema = make_model().to(device)

    teacher = make_model().to(device)

    ckpt = torch.load(args.checkpoint)
    teacher.load_state_dict(ckpt["G"])
    del ckpt
    print("Model loaded.")

    print("Scanning images directory...")
    files = glob.glob(args.images_dir + "/**", recursive=True)
    images = []
    ext = [".jpg", ".png"]
    for f in files:
        if not os.path.splitext(f)[-1] in ext:
            continue
        images.append(f)

    for f in tqdm(images):
        rel_path = os.path.relpath(f, args.images_dir)
        rel_dir = os.path.dirname(rel_path)
        f_name = os.path.basename(f)
        dst_dir = os.path.join(args.output_dir, rel_dir)
        dst_path = os.path.join(dst_dir, f_name + ".pt")
        try:
            latents, _ = calc_latents(teacher, f, device)
        except:
            print(f"{f} skipped.")
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        torch.save({"latents": latents.cpu().detach()}, dst_path)

    print("Finished.")

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, required=True)
    parser.add_argument("--images_dir", help="Images directory.", type=str, required=True)
    parser.add_argument("--output_dir", help="Images directory.", type=str, required=True)
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()

    args = parser.parse_args()

    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")

    process_images(args, make_model)