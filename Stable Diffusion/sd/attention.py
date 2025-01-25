import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads 
        self.d_head = d_embed // n_heads
        
    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (batch size, seq len, dim)
        input_shape = x.shape
        
        batch_size, sequence_length, d_embed = input_shape
        
        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # (batch size, seq len, dim) -> (batch size, seq len, dim * 3) -> 3 tensors of shape (batch size, seq len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=1)
        
        # (batch size, seq len, dim) -> (batch size, seq len, h, dim / h) -> (batch size, h, seq len, dim / h)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)
        
        # (batch size, h, seq len, seq len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask: 
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        # (batch size, h, seq len, seq len) @ (batch size, h, seq len, dim / h) -> (batch size, h, seq len, dim / h)
        output = weight @ v
        
        # (batch size, h, seq len, dim / h) -> (batch size, seq len, h, dim / h)
        output = output.transpose(1, 2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        # (batch size, seq len, dim)
        return output
        
class CrossAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().___init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
    def forward(self, x, y):
        # x: (latent): (batch size, seq len q, dim q)
        # y: (context): (batch size, seq len k v, dim k v) = (batch size, 77, 768)
        
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        
        output = output.transpose(1, 2).continuous()
        
        output = output.view(input_shape)
        
        output = self.out_proj(output)
        
        return output