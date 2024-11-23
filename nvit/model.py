import math
import inspect
from dataclasses import dataclass
from typing import Tuple, cast

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention


@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0 ** 0.5)    # 1 / sqrt(n_embd)
    use_nViT: int = 0
    sz_init_value: float = 1.00
    sz_init_scaling: float = 1.0
    dropout: float = 0.0
    bias: bool = False
    channels: int = 3
    num_classes: int = 1000


class Block(nn.Module):

    def __init__(self, config: ViTConfig, iblock: int) -> None:
        super(Block, self).__init__()
        self.config = config

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias)
        self.silu = nn.SiLU()
        self.mlp_c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        if (config.use_nViT == 0):
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)

        if (config.use_nViT == 1):
            self.attn_alpha_init_value = torch.scalar_tensor(0.05, dtype=torch.float32)
            self.attn_alpha_init_scaling = torch.scalar_tensor(config.base_scale, dtype=torch.float32)
            self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.mlp_alpha_init_value = torch.scalar_tensor(0.05, dtype=torch.float32)
            self.mlp_alpha_init_scaling = torch.scalar_tensor(config.base_scale, dtype=torch.float32)
            self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.sqk_init_value = torch.scalar_tensor(1.0, dtype=torch.float32)
            self.sqk_init_scaling = torch.scalar_tensor(config.base_scale, dtype=torch.float32)
            self.sqk = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.suv_init_value = torch.scalar_tensor(1.0, dtype=torch.float32)
            self.suv_init_scaling = torch.scalar_tensor(1.0, dtype=torch.float32)
            self.suv = torch.nn.Parameter(self.suv_init_scaling*torch.ones(2 * 4 * config.n_embd, dtype=torch.float32))

    
    def justnorm(self, x: torch.Tensor) -> torch.Tensor:
        #return F.normalize(x, p=2, dim=-1)
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, C = h.size()  # batch, sequence_length, embedding_dim

        if (self.config.use_nViT == 0):
            h = self.rmsnorm_att(h)
        
        # Project to q, k, v
        q = self.query(h)  # [B, T, C]
        k = self.key(h)    # [B, T, C]
        v = self.value(h)  # [B, T, C]

        # Split embedding dim into heads: [B, T, C] -> [B, H, T, D]
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.config.n_head)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.config.n_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.config.n_head)

        if (self.config.use_nViT == 1):
            sqk = (self.sqk * (self.sqk_init_value/self.sqk_init_scaling))
            sqk = rearrange(sqk, '(h d) -> 1 h 1 d', h=self.config.n_head)
            q = sqk * self.justnorm(q)  
            k = sqk * self.justnorm(k)  

        head_size = C // self.config.n_head
        sqrt_head_dim = head_size ** 0.5
        softmax_scale = 1.0 / sqrt_head_dim if self.config.use_nViT == 0 else sqrt_head_dim
        
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        
        # Get attention output [B, H, T, D]
        attn_output = cast(torch.Tensor,flex_attention(q, k, v, scale=softmax_scale))
        
        # Merge heads back: [B, H, T, D] -> [B, T, C]
        h_att = rearrange(attn_output, 'b h t d -> b t (h d)')
        
        # Project attention output
        h_att = self.att_c_proj(h_att)

        if (self.config.use_nViT == 0):
            h = h + h_att
        if (self.config.use_nViT == 1):
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)
            
            A_norm = self.justnorm(h)
            B_norm = self.justnorm(h_att)
                
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        # MLP block
        if (self.config.use_nViT == 0):
            h = self.rmsnorm_mlp(h)
        
        uv = self.c_fc(h)
        if (self.config.use_nViT == 1):
            suv = (self.suv * ((self.suv_init_value/self.suv_init_scaling) * (self.config.n_embd ** 0.5))) 
            uv = suv * uv  
        
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        if (self.config.use_nViT == 0):
            h = h + h_mlp
        if (self.config.use_nViT == 1):
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = self.justnorm(h)
            B_norm = self.justnorm(h_mlp)
                
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        return h

class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


def posemb_sincos_2d(h: int, w: int, dim: int, temperature: int = 10000, dtype = torch.float32) -> torch.Tensor:
    """2D Sinusoidal Position Embedding"""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        
        # Replace linear patch embedding with CNN
        self.patch_embed = nn.Conv2d(
            config.channels, 
            config.n_embd,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Position embedding
        self.h = config.image_size // config.patch_size
        self.pos_embedding = posemb_sincos_2d(
            h=self.h,
            w=self.h,
            dim=config.n_embd
        )

        # Transformer blocks
        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, il) for il in range(config.n_layer)])
        ))

        # Output head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.num_classes)
        )
        
        if self.config.use_nViT == 1:
            self.sz = torch.nn.Parameter(
                self.config.sz_init_scaling * torch.ones(config.num_classes, dtype=torch.float32)
            )

        # Initialize weights
        self.apply(self._init_weights)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        if self.config.use_nViT == 1 and isinstance(module, nn.Linear):
            torch.nn.init.constant_(self.sz, self.config.sz_init_value)

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str) -> torch.optim.AdamW:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        if self.config.use_nViT == 1:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_dict.items() if 'sz' not in n and p.dim() >= 2], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_dict.items() if 'sz' not in n and p.dim() < 2], 'weight_decay': 0.0},
                {'params': [self.sz], 'weight_decay': 0.0}
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_dict.items() if p.dim() >= 2], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_dict.items() if p.dim() < 2], 'weight_decay': 0.0}
            ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, betas=betas, fused=True if device_type == "cuda" else False)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> Tuple[float, float]:
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # First estimate the number of flops we do per iteration.
        # See: https://github.com/pytorch/pytorch/issues/110656
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.image_size//cfg.patch_size * cfg.image_size//cfg.patch_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu, flops_achieved

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        device = img.device
        
        # CNN patch embedding
        x = self.patch_embed(img)  # [B, n_embd, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, n_embd]
        
        # Add position embeddings
        x = x + self.pos_embedding.to(device, dtype=x.dtype)
        
        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        # Pool and classify
        x = x.mean(dim=1)  # Global average pooling
        logits = self.mlp_head(x)

        if self.config.use_nViT == 1:
            sz = self.sz * (self.config.sz_init_value / self.config.sz_init_scaling)
            logits = sz * logits

        return logits

    @property
    def num_params(self) -> int:
        """Return number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())
