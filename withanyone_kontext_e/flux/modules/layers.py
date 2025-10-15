

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

# from ..math import attention, rope
from ..math import rope
from ..math import  attention, attention_aug_bbox, attention_return_Q_and_map
# from ..math import attention
import torch.nn.functional as F

TOKEN_AUG_IDX = 2048

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

def reshape_tensor(x, heads):
    # print("x in reshape_tensor", x.shape)
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x
class PerceiverAttentionCA(nn.Module):
    def __init__(self, *, dim=3072, dim_head=64, heads=16, kv_dim=2048):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, mask=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        # print("x, latents in PerceiverAttentionCA", x.shape, latents.shape)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        # print("q, k, v in PerceiverAttentionCA", q.shape, k.shape, v.shape)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # # attention
        # scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        # weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        # print("is there any nan in weight:", torch.isnan(weight).any())
        # if mask is not None:
        #     # Mask shape should be [batch_size, num_heads, q_len, kv_len]
        #     # weight = weight.masked_fill(mask == 0, float("-inf"))
        #     if mask.dtype == torch.bool:
        #         # Boolean mask: False values are masked out
        #         # print("Got boolean mask")
        #         weight = weight.masked_fill(~mask, -float('inf'))
        #     else:
        #         # Float mask: values are added directly to scores
        #         weight = weight + mask
        # print("is there any nan in weight after mask:", torch.isnan(weight).any())
        # weight = torch.softmax(weight, dim=-1)
        # print("is there any nan in weight after softmax:", torch.isnan(weight).any())
        # out = weight @ v

        # use sdpa
        # if mask is not None:
        #     print("mask shape in PerceiverAttentionCA", mask.shape)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)




class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return ((x * rrms) * self.scale.float()).to(dtype=x_dtype)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x)
        return x

class LoraFluxAttnProcessor(nn.Module):

    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight


    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x) + self.qkv_lora(x) * self.lora_weight
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x) + self.proj_lora(x) * self.lora_weight
        return x
# class IDCrossAttention(nn.Module):
#     def __init__(self, *, dim=3072, dim_head=64, heads=16, kv_dim=3072):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.dim_head = dim_head
#         self.heads = heads
#         inner_dim = dim_head * heads

#         self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
#         self.norm2 = nn.LayerNorm(dim)

#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)

#     def forward(self, x, y):
#         x = self.norm1(x)
#         y = self.norm2(y)

#         q = self.to_q(x)
#         k, v = self.to_kv(y).chunk(2, dim=-1
    
class IDSelfAttention(nn.Module):
    def __init__(self, *, dim=3072, dim_head=64, heads=16, kv_dim=3072):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
    def forward(self, x, mask=None, pe=None, return_map=False):
        # Normalize the input
        x_norm = self.norm2(x)
        
        # Project to get query, key, value
        qkv = self.to_qkv(x_norm)
        
        # Split into q, k, v and reshape
        chunks = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), chunks)
        
        # Apply attention using the provided function
        
        out = attention_aug_bbox(q, k, v, pe=pe, mask=mask, token_aug_idx=-1, text_length=-1, image_length=-1, return_map=return_map)
   
        
        # Process output based on whether we're returning attention map
        if return_map:
            attn_out, attn_map = out
            return self.to_out(attn_out), attn_map
        else:
            return self.to_out(out), None

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
    def forward():
        pass

@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )
    
class DiffusersFluxDoubleStreamLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 32, network_alpha=None, lora_weight: float = 1):
        super().__init__()
        # Image stream LoRA modules
        self.img_q_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.img_k_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.img_v_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.img_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # Text stream LoRA modules
        self.txt_q_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.txt_k_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.txt_v_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.txt_proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # Modulation LoRA
        self.img_mod_lora = LoRALinearLayer(dim, dim * 6, rank, network_alpha)
        self.txt_mod_lora = LoRALinearLayer(dim, dim * 6, rank, network_alpha)
        
        # MLP LoRA
        self.img_mlp_proj_lora = LoRALinearLayer(dim, dim * 4, rank, network_alpha)
        self.img_mlp_out_lora = LoRALinearLayer(dim * 4, dim, rank, network_alpha)
        self.txt_mlp_proj_lora = LoRALinearLayer(dim, dim * 4, rank, network_alpha)
        self.txt_mlp_out_lora = LoRALinearLayer(dim * 4, dim, rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, mask, text_length, image_length, return_map, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # Prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        
        # Apply LoRA to each QKV component separately for image
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = torch.split(img_qkv, attn.hidden_size, dim=-1)
        
        img_q = img_q + self.img_q_lora(img_modulated) * self.lora_weight
        img_k = img_k + self.img_k_lora(img_modulated) * self.lora_weight
        img_v = img_v + self.img_v_lora(img_modulated) * self.lora_weight
        
        img_qkv = torch.cat([img_q, img_k, img_v], dim=-1)
        
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # Prepare text for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        
        # Apply LoRA to each QKV component separately for text
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = torch.split(txt_qkv, attn.hidden_size, dim=-1)
        
        txt_q = txt_q + self.txt_q_lora(txt_modulated) * self.lora_weight
        txt_k = txt_k + self.txt_k_lora(txt_modulated) * self.lora_weight
        txt_v = txt_v + self.txt_v_lora(txt_modulated) * self.lora_weight
        
        txt_qkv = torch.cat([txt_q, txt_k, txt_v], dim=-1)
        
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # Run attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        if return_map:
            results = attention_return_Q_and_map(q, k, v, pe=pe, mask=mask, return_map=True)
            attn1 = results.result
            attn_map = results.attention_map
            Q_double_lyaer = results.Q
        else:
            attn1 = attention(q, k, v, pe=pe, mask=mask, token_aug_idx=TOKEN_AUG_IDX, 
                             text_length=text_length, image_length=image_length, return_map=return_map)
        
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # Calculate the image blocks with LoRA
        img_proj = attn.img_attn.proj(img_attn) + self.img_proj_lora(img_attn) * self.lora_weight
        img = img + img_mod1.gate * img_proj
        
        # Apply LoRA to MLP for image
        img_mlp_in = (1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift
        
        # Calculate the text blocks with LoRA
        txt_proj = attn.txt_attn.proj(txt_attn) + self.txt_proj_lora(txt_attn) * self.lora_weight
        txt = txt + txt_mod1.gate * txt_proj
        
        # Apply LoRA to MLP for text
        txt_mlp_in = (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
        
        # Handle MLP LoRA for img and txt
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        if return_map:
            return img, txt, attn_map, Q_double_lyaer
        return img, txt

class DoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, mask, text_length, image_length, return_map, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # attn1 = attention(q, k, v, pe=pe, mask=mask, token_aug_idx=TOKEN_AUG_IDX, text_length=text_length, image_length=image_length, return_map=return_map)
        if return_map:
            results = attention_return_Q_and_map(q, k, v, pe=pe, mask=mask, return_map=True)
            attn1 = results.result
            attn_map = results.attention_map
            Q_double_lyaer = results.Q
        else:
            attn1 = attention(q, k, v, pe=pe, mask=mask, token_aug_idx=TOKEN_AUG_IDX, text_length=text_length, image_length=image_length, return_map=return_map)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * (attn.img_attn.proj(img_attn) + self.proj_lora1(img_attn) * self.lora_weight)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * (attn.txt_attn.proj(txt_attn) + self.proj_lora2(txt_attn) * self.lora_weight)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        if return_map:
            return img, txt, attn_map, Q_double_lyaer
        return img, txt

class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, mask, text_length, image_length, return_map, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        if return_map:
            # attn1, attn_map = attention(q, k, v, pe=pe, mask=attention_kwargs.get("mask"), token_aug_idx=TOKEN_AUG_IDX,text_length=text_length, image_length=image_length, return_map=return_map)
            results = attention_return_Q_and_map(q, k, v, pe=pe, mask=attention_kwargs.get("mask"), return_map=True)
            attn1 = results.result
            attn_map = results.attention_map
            Q_double_lyaer = results.Q
        else:

            attn1 = attention(q, k, v, pe=pe, mask=attention_kwargs.get("mask"), token_aug_idx=TOKEN_AUG_IDX,text_length=text_length, image_length=image_length, return_map=return_map)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        if return_map:
            return img, txt, attn_map, Q_double_lyaer
        return img, txt

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
        mask: Tensor | None = None,
        text_length: int = None,
        image_length: int = None,
        return_map: bool = False,
        **attention_kwargs
    ) -> tuple[Tensor, Tensor]:
        if image_proj is None:

            return self.processor(self, img, txt, vec, pe, mask, text_length, image_length, return_map=return_map)
        else:

            return self.processor(self, img, txt, vec, pe, mask, text_length, image_length, image_proj, ip_scale, return_map=return_map)


class DiffusersFluxSingleStreamLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 32, network_alpha=None, lora_weight: float = 1):
        super().__init__()
        # Separate QKV LoRA modules
        self.q_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.k_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.v_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        
        # For proj_out
        self.proj_out_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        
        # For norm.linear (modulation)
        self.modulation_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, 
                mask=None, text_length=None, image_length=None, return_map=False, **attention_kwargs) -> Tensor:
        # Apply LoRA to modulation if needed
        # Note: This isn't directly used in the forward pass yet as we need more information on how to integrate it
        
        # Get basic modulation
        mod, _ = attn.modulation(vec)
        
        # Apply modulation
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        
        # Get base QKV and MLP components
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        # Split QKV for separate LoRA application
        q, k, v = torch.split(qkv, attn.hidden_size, dim=-1)
        
        # Apply LoRA to each component
        q = q + self.q_lora(x_mod) * self.lora_weight
        k = k + self.k_lora(x_mod) * self.lora_weight
        v = v + self.v_lora(x_mod) * self.lora_weight
        
        # Recombine QKV
        qkv = torch.cat([q, k, v], dim=-1)
        
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)
        
        # Compute attention
        if return_map:
            results = attention_return_Q_and_map(q, k, v, pe=pe, mask=mask, return_map=True)
            attn_1 = results.result
            attn_map = results.attention_map
            Q_double_lyaer = results.Q
        else:
            attn_1 = attention(q, k, v, pe=pe, mask=mask, token_aug_idx=TOKEN_AUG_IDX, 
                              text_length=text_length, image_length=image_length, return_map=return_map)
        
        # Compute output with LoRA
        concat_features = torch.cat((attn_1, attn.mlp_act(mlp)), 2)
        output = attn.linear2(concat_features)
        output = output + self.proj_out_lora(concat_features) * self.lora_weight
        
        output = x + mod.gate * output
        
        if return_map:
            return output, attn_map, Q_double_lyaer
        
        return output

class SingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, mask = None, text_length = None, image_length = None, return_map=False) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        if return_map:
            results = attention_return_Q_and_map(q, k, v, pe=pe, mask=mask, return_map=True)
            attn_1 = results.result
            attn_map = results.attention_map
            Q_double_lyaer = results.Q
        else:
            attn_1 = attention(q, k, v, pe=pe, mask=mask, token_aug_idx=TOKEN_AUG_IDX,text_length=text_length, image_length=image_length, return_map=return_map)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = output + self.proj_lora(torch.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weight
        output = x + mod.gate * output
        if return_map:
            return output, attn_map, Q_double_lyaer
        
        return output


class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, mask: Tensor, text_length, image_length, return_map=False, **attention_kwargs) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        if return_map:
            results = attention_return_Q_and_map(q, k, v, pe=pe, mask=mask, return_map=True)
            attn_1 = results.result
            attn_map = results.attention_map
            Q_double_lyaer = results.Q
        else:
            attn_1 = attention(q, k, v, pe=pe, mask=mask, token_aug_idx=TOKEN_AUG_IDX,text_length=text_length, image_length=image_length)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        if return_map:
            return output, attn_map, Q_double_lyaer
        return output

class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        mask: Tensor | None = None,
        text_length: int | None = None,
        image_length: int | None = None,
        return_map: bool = False,
    ) -> Tensor:
        if image_proj is None:
            return self.processor(self, x, vec, pe, mask, text_length=text_length, image_length=image_length, return_map=return_map)
        else:
            return self.processor(self, x, vec, pe, mask, image_proj, ip_scale, text_length=text_length, image_length=image_length, return_map=return_map)



class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
