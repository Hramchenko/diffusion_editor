# Face Attribute Manipulation with Diffusion Autoencoders and StyleFlow
🚀 Unofficial implementation of [Diffusion Autoencoders: Toward a Meaningful and Decodable Representation](https://arxiv.org/abs/2111.15640) for face attribute manipulation.

## Image generation

![output](./data/output.png)

Download pretrained [weights](https://disk.yandex.ru/d/SjciA92PbGtVxw) to `checkpoints` directory.

Run `styleflow_script.ipynb`.


## Training 

Download Celeba-HQ dataset.

Run `celeba_ae_script.ipynb`.

## Requirements
* pytorch
* __torchdiffeq==0.0.1__
* kornia

## Acknowledgement
1. [Diffusion Autoencoders: Toward a Meaningful and Decodable Representation](https://arxiv.org/abs/2111.15640).
2. [rosinality/denoising-diffusion-pytorch](https://github.com/rosinality/denoising-diffusion-pytorch).
3. [StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows (ACM TOG 2021)](https://github.com/RameenAbdal/StyleFlow).