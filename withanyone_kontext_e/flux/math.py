

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

# Example usage in your model:
# model.some_attention_layer.register_forward_hook(hook_attention_for_visualization)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask = None, token_aug_idx = -1, text_length = None, image_length = None, return_map = False) -> Tensor:
    q, k = apply_rope(q, k, pe)
    # if mask is not None:
    #     print("mask is not None")
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def attention_return_Q_and_map(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask = None, return_map = False):
    if pe is not None:
        q, k = apply_rope(q, k, pe)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    x = rearrange(x, "B H L D -> B L (H D)")
    if return_map:
        # Compute attention map, and only use txt and img tokens
        # 1. discard lq tokens and id embedding tokens
        #   txt: 512, img: 1024, lq: 32, id: 16
        #   they are concatenated as: 512-32-1024-32-16
        #   we need to get the attention map for txt and img tokens only
        # if q.shape[2] == 512 + 32 + 1024 + 32 + 16:
        if q.shape[2] == 1616:
            q_map = torch.cat((q[:, :, :512], q[:, :, 544: 512+32+1024]), dim=2)
            k_map = torch.cat((k[:, :, :512], k[:, :, 544: 512+32+1024]), dim=2)
        elif q.shape[2] == 1536:
            q_map = q
            k_map = k
        else:
            raise ValueError(f"Unexpected q shape: {q.shape}, expected 1616 or 1536")
        # 2. compute the attention map

        d_k = k_map.size(-1)
        attn_scores = torch.matmul(q_map, k_map.transpose(-2, -1)) / (d_k ** 0.5)
        # attn_weights = torch.softmax(attn_scores, dim=-1)
        # attn_map = attn_scores.mean(dim=1)
        q_map = rearrange(q_map, "B H L D -> B L (H D)")
        return AttentionReturnQAndMAP(result=x, attention_map=attn_scores, Q=q_map)
    else:
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
        
        # # augment the attention scores that are not masked, and row or column index is in the last 2048 tokens
        # if token_aug_idx > 0:
        #     assert text_length > 0 and image_length > 0
        #     # augment [-2048, -1] rows that are not masked (*1.3), exlude the last 2048 tokens
        #     # print("Applying augmentation to attention scores")
        #     attn_scores[:, :, text_length + image_length:, text_length:text_length + image_length] *= 15
        #     # augment [-2048, -1] columns that are not masked (*1.3)
        #     attn_scores[:, :, text_length:text_length + image_length, text_length + image_length:] *= 15

    
    # Apply softmax to get attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attn_weights, v)
    
    # Rearrange dimensions as in the original function
    output = rearrange(output, "B H L D -> B L (H D)")

    if return_map:
        # Return attention weights for visualization and router loss
        # (b, head, token, dim/head) -> (b, token, dim)
        # attn_scores = rearrange(attn_scores, "b h s d -> b s (h d)")
        return output, attn_scores
    
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
