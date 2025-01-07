# Variational Autoencoder (VAE)

## Introduction
VAE was created to solve the latent space issue that was present regular autoencoders. Key features of a VAE is its **encoder**, **decoder**, **loss function** and **reparameterization**.

## Encoder 
The encoder denoted by $q_\phi(z|x)$ converts the input to latent distribution z. The dimensions of z denotes the complexity of the latent distribution and it is equal to the number of latent variables. Each latent variable has two values assigned to it $\mu$ and $\sigma^2$ denoting the mean and variance of a gaussian distribution.

## Decoder 
The decoder denoted by $p_\theta(x|z)$ reconstructs the input from the latent distribution z. The reconstruction quality highly depends on the organisation of the latent space.

## Loss function
The loss function consists of two parts the **reconstruction loss** and the **kl divergence loss**.

### Reconstruction loss
The reconstruction loss is used to quantify the reconstruction quality of the decoder by comparing it with the input. This expression turns out to be MSE between the pixel values of the output and the input.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{recon}}%20=%20-\mathbb{E}_{q_\phi(z|x)}[\log%20p_\theta(x|z)]" />
</p>

### KL divergence loss
The KL divergence loss is used to make the latent space organised such that if we sample near a respective class we will get image of that class. This was missing in regular autoencoders.

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\text{KL}}%20=%20\text{KL}(q_\phi(z|x)||p(z))%20=%20\frac{1}{2}\sum_{j=1}^J(\sigma_j^2%20+%20\mu_j^2%20-%20\log(\sigma_j^2)%20-%201)" />
</p>

## Reparameterization 
The reparameterization trick is used to make backpropagation possible within the network by moving the randomness outside of the model. 


