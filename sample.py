import argparse
import importlib
from v_diffusion import make_beta_schedule
from moving_average import init_ema_model
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import torch
import os
from train_utils import p_sample_loop
import cv2
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

def make_visualization(x0, latents, diffusion, device, image_size, need_tqdm=False, eta=0, clip_value=1.2):
    extra_args = {}

    noise = x0.repeat(image_size[0], 1, 1, 1).to(device)

    imgs = p_sample_loop(latents, diffusion, noise, extra_args, "cuda", samples_to_capture=5, need_tqdm=need_tqdm, eta=eta, clip_value=clip_value)
    images_ = []
    for images in imgs:
        images = images.split(1, dim=0)
        images = torch.cat(images, -1)
        images_.append(images)
    images_ = torch.cat(images_, 2)

    images_ = images_[0].permute(1, 2, 0).cpu().numpy()
    images_ = (255 * (images_ + 1) / 2).clip(0, 255).astype(np.uint8)
    return images_

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", help="Model module.", type=str, required=True)
    parser.add_argument("--checkpoint", help="Path to checkpoint.", type=str, required=True)
    parser.add_argument("--imageA", help="Path to the first image.", type=str, required=True)
    parser.add_argument("--imageB", help="Path to the second image.", type=str, required=True)
    parser.add_argument("--imageC", help="Path to the second image.", type=str, default="")
    parser.add_argument("--coeff", help="(C - B) coefficient", type=float, default=1)
    parser.add_argument("--out_file", help="Path to image.", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    parser.add_argument("--diffusion", help="Diffusion model.", type=str, default="GaussianDiffusion")
    parser.add_argument("--time_scale", help="Diffusion time scale.", type=int, default=1)
    parser.add_argument("--clipped_sampling", help="Use clipped sampling mode.", type=bool, default=False)
    parser.add_argument("--clipping_value", help="Noise clipping value.", type=float, default=1.2)
    parser.add_argument("--eta", help="Amount of random noise in clipping sampling mode(recommended non-zero values only for not distilled model).", type=float, default=0)
    return parser

def calc_latents(teacher, image_path, device):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = cv2.imread(image_path, 1)
    img_tensor = to_tensor(img).unsqueeze(0)
    size = teacher.image_size[2:4]
    img_tensor = F.interpolate(img_tensor, size=size, mode="bicubic", align_corners=False).clip(-1, 1)
    latents = teacher.make_latents(img_tensor.to(device))
    return latents, img_tensor


def sample_images(args, make_model):
    device = torch.device("cuda")

    teacher_ema = make_model().to(device)

    def make_diffusion(args, model, n_timestep, time_scale, device):
        betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=n_timestep).to(device)
        M = importlib.import_module("v_diffusion")
        D = getattr(M, args.diffusion)
        sampler = "ddpm"
        if args.clipped_sampling:
            sampler = "clipped"
        return D(model, betas, time_scale=time_scale, sampler=sampler)

    teacher = make_model().to(device)

    ckpt = torch.load(args.checkpoint)
    teacher.load_state_dict(ckpt["G"])
    n_timesteps = ckpt["n_timesteps"]//args.time_scale
    time_scale = ckpt["time_scale"]*args.time_scale
    del ckpt
    print("Model loaded.")

    teacher_diffusion = make_diffusion(args, teacher, n_timesteps, time_scale, device)
    image_size = deepcopy(teacher.image_size)
    image_size[0] = args.batch_size

    latents_a, image_a = calc_latents(teacher, args.imageA, device)
    latents_b, _ = calc_latents(teacher, args.imageB, device)
    if args.imageC != "":
        latents_c, _ = calc_latents(teacher, args.imageC, device)

    print("Reverse DDIM sampling for finding x_0...")
    a0 = teacher_diffusion.reverse_sample(latents_a, image_a.to(device), {}, need_tqdm=True)

    alpha = torch.linspace(0, 1, image_size[0])[:, None].to(device)
    if args.imageC == "":
        latents = alpha*latents_b.repeat(alpha.shape[0], 1) + (1 - alpha)*latents_a.repeat(alpha.shape[0], 1)
    else:
        latents = latents_a.repeat(alpha.shape[0], 1) + args.coeff*alpha*(latents_c - latents_b).repeat(alpha.shape[0], 1)

    print("Forward sampling...")
    img = make_visualization(a0, latents, teacher_diffusion, device, image_size, need_tqdm=True, eta=args.eta, clip_value=args.clipping_value)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    cv2.imwrite(args.out_file, img)

    print("Finished.")

if __name__ == "__main__":
    parser = make_argument_parser()

    args = parser.parse_args()

    M = importlib.import_module(args.module)
    make_model = getattr(M, "make_model")

    sample_images(args, make_model)