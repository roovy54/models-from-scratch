import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: batch_size, channels, 64, 64
            nn.Conv2d(
                channels_img + 1, features_d, kernel_size=4, stride=2, padding=1
            ),  # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 16 x 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 4 x 4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),  # 1 x 1
        )
        self.embedding = nn.Embedding(num_classes, img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(
            labels.shape[0], 1, self.img_size, self.img_size
        )
        x = torch.cat([x, embedding], dim=1)  # N x C x img_size (H) x img_size (W)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(
        self, z_dim, channels_img, features_g, num_classes, img_size, embed_size
    ):
        super(Generator, self).__init__()
        self.img_size = img_size
        # input: n, z_dim, 1, 1
        self.gen = nn.Sequential(
            self._block(
                z_dim + embed_size, features_g * 16, 4, 1, 0
            ),  # n, f_g * 16, 4, 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8 x 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16 x 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32 x 32
            nn.ConvTranspose2d(
                features_g * 2,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),  # [-1, 1]
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # latent vector z: N x noise_dim x 1 x 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0, 0.02)
