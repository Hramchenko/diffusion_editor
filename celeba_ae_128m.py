from train_utils import *
from ae_ddpm import *
from celeba_dataset import CelebaWrapper

BASE_NUM_STEPS = 1024
BASE_TIME_SCALE = 1

def make_model():
    encoder = EncoderM(in_channel=3,
                      channel=128,
                      channel_multiplier=[1, 2, 2, 4, 4],
                      n_res_blocks=2,
                      attn_strides=[8, 16],
                      attn_heads=4,
                      dropout=0,
                      fold=1)
    channel = 128
    unet = UNet(in_channel=3,
                channel=channel,
                channel_multiplier=[1, 2, 4, 4],
                n_res_blocks=2,
                attn_strides=[8, 16],
                attn_heads=4,
                use_affine_time=True,
                dropout=0,
                fold=1,
                time_dim=512 + channel * 4,
                time_embed_dim=channel * 4)
    net = Autoencoder(encoder, unet)
    net.image_size = [1, 3, 128, 128]
    return net

def make_dataset():
    return CelebaWrapper(dataset_dir="./data/celeba_128/", resolution=128)