#composition of unet architecture
import torch
import torch.nn as nn
from .fourier_encoder import FourierEncoder
from .modules import Encoder, Midcoder, Decoder
from typing import List

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        # For SiLU (ReLU-like), fan_in mode is recommended
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        # Linear layers benefit from fan_out mode
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        # Initialize normalization scale to 1 and bias to 0
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        # Embeddings initialized to small normal noise
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class MFUNet(nn.Module):
    def __init__(self, in_channels, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_embed_dim: int, num_classes: int):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )
        self.t_embedder = FourierEncoder(t_embed_dim)
        self.r_embedder = FourierEncoder(t_embed_dim)
        self.y_embedder = nn.Embedding(num_classes+1, y_embed_dim)

        encoders = []
        decoders = []
        for (curr_channel, next_channel) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_channel, next_channel, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(2*next_channel, curr_channel, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))
        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)

        self.final_conv = nn.Conv2d(channels[0],in_channels, kernel_size=3, padding=1)
        self.apply(initialize_weights)


    def forward(self, x: torch.Tensor, t: torch.Tensor, r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        # Embed t and y
        res = []
        t_embed = self.t_embedder(t) # (bs, t_embed_dim)
        r_embed = self.r_embedder(r) # (bs, t_embed_dim)
        y = y.to(torch.long)
        y_embed = self.y_embedder(y) # (bs, y_embed_dim)
        
        x = self.init_conv(x) # (bs, in_channels, h, w)
        
        for encoder in self.encoders:
            x = encoder(x, t_embed,r_embed, y_embed) # (bs, c[i], h, w) -> (bs, c[i+1], h//2, w//2)
            res.append(x.clone()) # push res into the stack

        x = self.midcoder(x, t_embed, r_embed, y_embed)

        for decoder in self.decoders:
            x_res = res.pop() # (bs, c[i], h, w) last-in, first-out
            # do concatenation , double the channels
            x = torch.cat([x, x_res], dim=1) # (bs, c[i]*2, h, w)
            x = decoder(x, t_embed,r_embed, y_embed) # (bs, c[i-1], 2*h, 2*w)

        x = self.final_conv(x)

        return x
    
