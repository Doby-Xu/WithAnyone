
import torch
from einops import rearrange
from torch import Tensor

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from torch import Tensor
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
# a return class
@dataclass
class AttentionReturnQAndMAP:
    result: Tensor
    attention_map: Tensor
    Q: Tensor


def attention_return_Q_and_map():
    pass

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask = None, token_aug_idx = -1, text_length = None, image_length = None, return_map = False) -> Tensor:
    q, k = apply_rope(q, k, pe)
    # if mask is not None:
    #     print("mask is not None")
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def attention_aug_bbox(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask = None, token_aug_idx = -1, text_length = -1, image_length = -1, return_map = False) -> Tensor:
    if pe is not None:
        q, k = apply_rope(q, k, pe)
    
    # Scale factor based on key dimension
    d_k = k.size(-1)
    
    # Compute attention scores: (batch, heads, seq_len_q, seq_len_k)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    
    
    # Apply mask if provided
    if mask is not None:
        # print(f"token_aug_idx: {token_aug_idx}, text_length: {text_length}, image_length: {image_length}")
        if mask.dtype == torch.bool:
            # Boolean mask: False values are masked out
            # print("Got boolean mask")
            attn_scores = attn_scores.masked_fill(~mask, -float('inf'))
        else:
            # Float mask: values are added directly to scores
            attn_scores = attn_scores + mask
        

    # Apply softmax to get attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attn_weights, v)
    
    # Rearrange dimensions as in the original function
    output = rearrange(output, "B H L D -> B L (H D)")


    
    return output


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
