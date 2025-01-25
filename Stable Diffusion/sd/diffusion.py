import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        
        x = self.linear_1(x)
        
        x = F.silu(x)
        
        x = self.linear_2(x)
        
        # (1, 1280)
        return x

class UNET_ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels: 
            self.residual_layer = nn.Identity()
        else: 
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature (batch size, in_channels, height, width)
        # time (1, 1280)
        
        residue = feature
        
        feature = self.groupnorm_feature(feature)
        
        feature = F.silu(feature)
        
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnorm_merged(merged)
        
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)
            
class UNET_AttentionBlock(nn.Module):
    
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # (batch size, features, height, width)
        # context: (batch size, seq len, dim)
        
        residue_long = x
        
        x = self.groupnorm(x)
        
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (batch size, features, height, width) -> (batch size, features, height * width)
        x = x.view((n, c, h * w))
        
        # (batch size, features, height * width) -> (batch size, height * width, features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self Attention with skip connections
        
        residue_short = x
        
        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short
        
        residue_short = x
        
        # Normalization + Cross Attention with skip connection
        x = self.layernorm_2(x)
        
        # Cross Attention
        self.attention_2(x, context)
        
        x += residue_short
        
        residue_short = x
        
        # Normalization + FF with GeGLU and skip connection
        
        x = self.layernorm_3(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        
        x = self.linear_geglu_2(x)
        
        x += residue_short
        
        # (batch size, height * width, features) -> (batch size, features, height * width)
        x = x.transpose(-1, -2)
        
        x = x.view((n, c, h, w))
        
        return self.conv_output(x) + residue_long
        
        
class UpSample(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # (batch size, features, height, width) -> (batch size, features, height * 2, width * 2)
        
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
    
    
class SwitchSequential(nn.Sequential):
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor: 
        for layer in self: 
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else: 
                x = layer(x)
        return x
    
class UNET(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoders = nn.ModuleList([
            # (batch size, 4, height / 8, width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)), 
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)), 
            
            # (batch size, 320, height / 8, width / 8) -> (batch size, 320, height / 16, width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (batch size, 640, height / 16, width / 16) -> (batch size, 640, height / 32, width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (batch size, 1280, height / 32, width / 32) -> (batch size, 1280, height / 64, width / 64)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (batch size, 1280, height / 64, width / 64) -> (batch size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
        ])
        
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            
            UNET_AttentionBlock(8, 160),
            
            UNET_ResidualBlock(1280, 1280),
        )
        
        self.decoders = nn.ModuleList([
            # (batch size, 2560, height / 64, width / 64) -> (batch size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),
             
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
              
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
             
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)), 
            
        ])
        
        
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # (batch size, 320, height / 8, width / 8)
        
        x = self.groupnorm(x)
        
        x = F.silu(x)
        
        x = self.conv(x)
        
        # (batch size, 4, height / 8, width / 8)
        
        return x
    
    
class Diffusion(nn.Module):
    
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
        
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (batch size, 4, height / 8, width / 8)
        # context: (batch size, seq len, dim)
        # time: (1, 320)
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (batch, 4, height / 8, width / 8) -> (batch, 320, height / 8, width / 8)
        output = self.unet(latent, context, time)
        
        # (batch 320, height / 8, width / 8) -> (batch, 4, height / 8, width / 8)
        output = self.final(output)
        
        return output
    
    