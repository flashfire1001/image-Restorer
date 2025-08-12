import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.vision_transformer import Attention
import torch.nn.functional as F
from einops import repeat, pack, unpack
from torch.cuda.amp import autocast


def modulate(x, scale, shift):
    """
    modulation is for adapting the blocks to timestep or class

    Args:
        x : tensor with shape (N, T, D)
        scale : learnable standard deviation
        shift : learnable average value

    Returns:
        Normed X with modulate.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t.reshape(-1,1) * freqs.reshape(1,-1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t*1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        # label embedding is a typical raw nn.embedding to learn from the train
        self.embedding = nn.Embedding(num_classes + 1, dim)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding(labels)
        return embeddings


class RMSNorm(nn.Module):
    #Replaces standard LayerNorm with RMSNorm for computational stability and simplicity.
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        x: (N, T, D), N for batch_size T for num_tokens and D for embedding dimension
        First normalize the x to a unit vector (L2 norm = 1)
        then Scale by self.g to be real RMS Normalization
        Finally scale by self.g (init with 1), which is set learnable for enhanced flexibilty.
        """
        return F.normalize(x, dim=-1) * self.scale * self.g


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        # first norm is for attention layers
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        # flasth attn can not be used with jvp
        self.attn.fused_attn = False
        
        # second norm is for FeedForward layers
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_dim, 
            act_layer=approx_gelu, 
            drop=0
        ) #serve as feedforward network
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        # genarate 6 vectors for shift scale and gate
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        
        # residaal layer addition 1
        # do attention of the (norm+modulated x)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        # residaal layer addition 2
        # do feedforward of the (norm+modulated x)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim) 
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MFDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        num_classes=10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # instantiate the embedders
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim) # do convolution, flatten, projection and reshape . from (N, C, H, W) to (N, T = H*W/P**2, D = 2* P**2)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)

        # set y_embedder for cfg task
        self.use_cond = num_classes is not None
        #don't use cfg self.use_cond = False
        
        self.y_embedder = LabelEmbedder(num_classes, dim) if self.use_cond else None
        
        # D = H*W/P**2
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding for position:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=True)

        # the core transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C) # output token tensors
        imgs: (N, H, W, C)
        this code:
        Reshapes them into 2D patch grids
        Reorders them to match image layout
        Stitches the small patches into full-sized image
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x) # x: (N, C, H_patch, P, W_patch, P) → (N, C, h, p, w, p)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, r, y=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        H, W = x.shape[-2:]

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        t = self.t_embedder(t)                   # (N, D)
        r = self.r_embedder(r)
        # t = torch.cat([t, r], dim=-1)
        t = t + r

        # condition
        c = t
        if self.use_cond:
            y = self.y_embedder(y)               # (N, D)
            c = c + y                                # (N, D)

        for i, block in enumerate(self.blocks):
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels): get ready for unpatchifying
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


# Positional embedding from:
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb