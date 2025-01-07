import torch
from torch import nn


# input image -> hidden dimension -> mean, std -> parameterization -> decoder -> output
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        # activation function
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img2_hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sig(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return self.hid_2img(h)

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma * epsilon
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma


if __name__ == "__main__":
    x = torch.randn(1, 784)
