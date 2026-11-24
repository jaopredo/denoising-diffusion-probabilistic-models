# Denoising Diffusion Probabilistic Models

This repository brings an implementation of the [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) paper.

## Dependences
The only requirement for this project is for you to have the `uv` package controller installed. If you don't, you can install it in the official page: [click here](https://docs.astral.sh/uv/)

## Setup the project
To setup the project, you just need to run the following command
```
make setup
```

This command syncs the project dependences using the `uv` package controller and installs the `torch` and `torchvision` packages based on your GPU (That means, if you have a NVIDEA GPU, it will install CUDA related packages, if you have AMD: ROCm related, if you have none, just the standart cpu packages)
