# Wasserstein GAN

## Introduction

Wasserstein GAN's has Wasserstein's Distance as loss function to train the GAN. If this loss tends to zero for a discriminator then the generator can produce high quality outputs. The discriminator is also called critic in this case and is trained more than the generator.

## Loss function

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?W(P_r,%20P_g)%20=%20\sup_{\|f\|_L%20\leq%201}%20\mathbb{E}_{x%20\sim%20P_r}[f(x)]%20-%20\mathbb{E}_{x%20\sim%20P_g}[f(x)]" />
</p>

The main features of this loss function is that this requires the gradients to follow certain constraint called the **lipschitz constraint**. In order to enforce lipschitz they introduce gradient clipping and gradient penalty.
