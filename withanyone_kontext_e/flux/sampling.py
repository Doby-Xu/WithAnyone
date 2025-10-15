
import math
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm

from .model import Flux
from .modules.conditioner import HFEmbedder

import numpy as np
from PIL import Image

from .modules.autoencoder import AutoEncoder

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare_kontext(
    t5: HFEmbedder,
    clip: HFEmbedder,
    prompt: str | list[str],
    ae: AutoEncoder,
    seed: int,
    device: torch.device,
    img_cond: Image.Image | None = None,
    img_cond_path: str | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    bs: int = 1,
) -> tuple[dict[str, Tensor], int, int]:
    # load and encode the conditioning image
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    assert img_cond is not None or img_cond_path is not None, "Either img_cond or img_cond_path must be provided."

    if img_cond is None:
        img_cond = Image.open(img_cond_path).convert("RGB")
    width, height = img_cond.size if isinstance(img_cond, Image.Image) else img_cond[0].size
    aspect_ratio = width / height 

    assert aspect_ratio > 0.9 and aspect_ratio < 1.1, "Your model is trained on square images, please provide a square image."
    # # Kontext is trained on specific resolutions, using one of them is recommended
    # _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)



    width = 2 * int(width / 16)
    height = 2 * int(height / 16)
    if not isinstance(img_cond, list):
        img_cond = img_cond.resize((8 * width, 8 * height), Image.Resampling.LANCZOS)
        img_cond = np.array(img_cond)
        img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
        img_cond = rearrange(img_cond, "h w c -> 1 c h w")
        img_cond_orig = img_cond.clone()
    else: # batch input
        batched_img_cond = []
        for img in img_cond:
            img = img.resize((8 * width, 8 * height), Image.Resampling.LANCZOS)
            img = np.array(img)
            img = torch.from_numpy(img).float() / 127.5 - 1.0
            img = rearrange(img, "h w c -> 1 c h w")
            batched_img_cond.append(img)
        img_cond = torch.cat(batched_img_cond, dim=0).to(device).to(torch.bfloat16)

    with torch.no_grad():
        img_cond = ae.encode(img_cond.to(device))

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    # image ids are the same as base image with the first dimension set to 1
    # instead of 0
    img_cond_ids = torch.zeros(height // 2, width // 2, 3)
    img_cond_ids[..., 0] = 1
    img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(height // 2)[:, None]
    img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(width // 2)[None, :]
    img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)

    if target_width is None:
        target_width = 8 * width
    if target_height is None:
        target_height = 8 * height

    img = get_noise(
        1,
        target_height,
        target_width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond_seq"] = img_cond
    return_dict["img_cond_seq_ids"] = img_cond_ids.to(device)
    # return_dict["img_cond_orig"] = img_cond_orig
    return return_dict, target_height, target_width

def prepare(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ref_img: None | Tensor=None,
    pe: Literal['d', 'h', 'w', 'o'] ='d'
) -> dict[str, Tensor]:
    assert pe in ['d', 'h', 'w', 'o']
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if ref_img is not None:
        _, _, ref_h, ref_w = ref_img.shape
        ref_img = rearrange(ref_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if ref_img.shape[0] == 1 and bs > 1:
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)
        ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)
        # img id分别在宽高偏移各自最大值
        h_offset = h // 2 if pe in {'d', 'h'} else 0
        w_offset = w // 2 if pe in {'d', 'w'} else 0
        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None] + h_offset
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :] + w_offset
        ref_img_ids = repeat(ref_img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    if ref_img is not None:
        return {
            "img": img,
            "img_ids": img_ids.to(img.device),
            "ref_img": ref_img,
            "ref_img_ids": ref_img_ids.to(img.device),
            "txt": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
            "vec": vec.to(img.device),
        }
    else:
        return {
            "img": img,
            "img_ids": img_ids.to(img.device),
            "txt": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
            "vec": vec.to(img.device),
        }

def prepare_multi_ip(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ref_imgs: list[Tensor] | None = None,
    pe: Literal['d', 'h', 'w', 'o'] = 'd'
) -> dict[str, Tensor]:
    assert pe in ['d', 'h', 'w', 'o']
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    ref_img_ids = []
    ref_imgs_list = []
    pe_shift_w, pe_shift_h = w // 2, h // 2
    # if ref_imgs is None:
    #     ref_imgs = None
    for ref_img in ref_imgs:
        _, _, ref_h1, ref_w1 = ref_img.shape
        ref_img = rearrange(ref_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if ref_img.shape[0] == 1 and bs > 1:
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)
        ref_img_ids1 = torch.zeros(ref_h1 // 2, ref_w1 // 2, 3)
        # img id分别在宽高偏移各自最大值
        h_offset = pe_shift_h if pe in {'d', 'h'} else 0
        w_offset = pe_shift_w if pe in {'d', 'w'} else 0
        ref_img_ids1[..., 1] = ref_img_ids1[..., 1] + torch.arange(ref_h1 // 2)[:, None] + h_offset
        ref_img_ids1[..., 2] = ref_img_ids1[..., 2] + torch.arange(ref_w1 // 2)[None, :] + w_offset
        ref_img_ids1 = repeat(ref_img_ids1, "h w c -> b (h w) c", b=bs)
        ref_img_ids.append(ref_img_ids1)
        ref_imgs_list.append(ref_img)

        # 更新pe shift
        pe_shift_h += ref_h1 // 2
        pe_shift_w += ref_w1 // 2

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "ref_img": tuple(ref_imgs_list),
        "ref_img_ids": [ref_img_id.to(img.device) for ref_img_id in ref_img_ids],
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,


    arcface_embeddings = None,
    siglip_embeddings = None,
    bboxes: Tensor = None,
    return_map: bool = False,
    shortcut: bool = False,
    arc_only: bool = False,
    max_num_ids: int = 4,  # Default to support 2 identities
    img_width: int = 512,
    img_height: int = 512,
):
    

    i = 0

    
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1):
        
        # kontext input processing
        img_input = img
        img_input_ids = img_ids
        real_img_len = img.shape[1]

        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"

            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)





        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            siglip_embeddings=siglip_embeddings,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            arcface_embeddings=arcface_embeddings,
            bbox_lists=bboxes,
            arc_only=arc_only,
            max_num_ids=max_num_ids,
            real_img_len=real_img_len,
            original_width=img_width,
            original_height=img_height,
        )
        # kontext output processing
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred
        i += 1

    
    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
