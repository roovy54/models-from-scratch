# Deep Convolutional GAN

## Introduction

DCGAN is an architecture that is built on top of Vanilla GAN by changing Dense Layers to CNNs and weights are initialized from a normal distribution.

## Generator

Generator part of a DCGAN is made of blocks where each block is made of a ConvTranspose2D Layer to upsample from smaller latents to a image, BatchNorm2D and ReLU activation.

## Discriminator 

Discriminator part of a DCGAN is also made of blocks and each block consists of Conv2D, BatchNorm2D and LeakyReLU activation. The initial block does not contain BatchNorm2D layer.
