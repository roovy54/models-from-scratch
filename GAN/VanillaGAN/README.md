# Generative Adversarial Networks (GANs)

## Introduction

Vanilla GAN is the architecture that was initially introduced in the 2014 paper. Main features of a GAN is the discriminator, generator and its loss functions.

## Generator

Generator part of a GAN is initially fed in with latents which is then used to convert it to a image. The loss of the generator is: 

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_G%20=%20\mathbb{E}_{z\sim%20p(z)}[\log(1-D(G(z)))]" />
</p>

## Discriminator 

Discriminator part of a GAN is given the real image and a fake image simultaneously and its job is to predict a probability of the image being a real image or not. The loss of the discriminator is as follows: 

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_D%20=%20-\mathbb{E}_{x\sim%20p_{\text{data}}}[\log%20D(x)]%20-%20\mathbb{E}_{z\sim%20p(z)}[\log(1-D(G(z)))]" />
</p>
