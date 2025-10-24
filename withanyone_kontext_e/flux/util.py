

import os
from dataclasses import dataclass

import torch
import json
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file as load_sft

from withanyone_kontext_e.flux.model import Flux, FluxParams
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from .modules.conditioner import HFEmbedder

import re
from withanyone_kontext_e.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor, DiffusersFluxSingleStreamLoraProcessor, DiffusersFluxDoubleStreamLoraProcessor

def load_model(ckpt, device='cpu'):
    if ckpt.endswith('safetensors'):
        from safetensors import safe_open
        pl_sd = {}
        with safe_open(ckpt, framework="pt", device=device) as f:
            for k in f.keys():
                pl_sd[k] = f.get_tensor(k)
    else:
        pl_sd = torch.load(ckpt, map_location=device)
    return pl_sd

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]

def load_checkpoint(local_path, repo_id, name):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    elif repo_id is not None and name is not None:
        print(f"Loading checkpoint {name} from repo id {repo_id}")
        checkpoint = load_from_repo_id(repo_id, name)
    else:
        raise ValueError(
            "LOADING ERROR: you must specify local_path or repo_id with name in HF to download"
        )
    return checkpoint


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None

configs = {
    "flux-kontext": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        repo_id_ae="black-forest-labs/FLUX.1-Kontext-dev",
        repo_flow="flux1-kontext-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fp8": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-krea": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Krea-dev",
        repo_id_ae="black-forest-labs/FLUX.1-Krea-dev",
        repo_flow="flux1-krea-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

def load_from_repo_id(repo_id, checkpoint_name):
    ckpt_path = hf_hub_download(repo_id, checkpoint_name)
    sd = load_sft(ckpt_path, device='cpu')
    return sd

def load_flow_model_no_lora(
    name: str,
    path: str,
    ipa_path: str ,
    device: str | torch.device = "cuda",
    hf_download: bool = True,
    lora_rank: int = 16,
    use_fp8: bool = False
):
    # Loading Flux
    print("Init model")
    # ckpt_path = path
    ckpt_path = os.path.join(path, "flux1-kontext-dev.safetensors") if path is not None else None
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow.replace("sft", "safetensors"))
    
    ipa_ckpt_path = ipa_path


    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params)


    # model = set_lora(model, lora_rank, device="meta" if ipa_ckpt_path is not None else device)

    if ckpt_path is not None:
        print("Loading lora")
        lora_sd = load_sft(ipa_ckpt_path, device=str(device)) if ipa_ckpt_path.endswith("safetensors")\
            else torch.load(ipa_ckpt_path, map_location='cpu')
         
        
        print("Loading main checkpoint")
        # load_sft doesn't support torch.device

        if ckpt_path.endswith('safetensors'):
            if use_fp8:
                print(
                    "####\n"
                    "We are in fp8 mode right now, since the fp8 checkpoint of XLabs-AI/flux-dev-fp8 seems broken\n"
                    "we convert the fp8 checkpoint on flight from bf16 checkpoint\n"
                    "If your storage is constrained"
                    "you can save the fp8 checkpoint and replace the bf16 checkpoint by yourself\n"
                )
                sd = load_sft(ckpt_path, device="cpu")
                sd = {k: v.to(dtype=torch.float8_e4m3fn, device=device) for k, v in sd.items()}
            else:
                sd = load_sft(ckpt_path, device=str(device))
            


            # # Then proceed with the update
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        else:
            dit_state = torch.load(ckpt_path, map_location='cpu')
            sd = {}
            for k in dit_state.keys():
                sd[k.replace('module.','')] = dit_state[k]
            sd.update(lora_sd)        
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            model.to(str(device))
        print_load_warning(missing, unexpected)
    return model


def merge_to_flux_model(
    loading_device, working_device, flux_state_dict, model, ratio, merge_dtype, save_dtype, mem_eff_load_save=False
):

    lora_name_to_module_key = {}
    keys = list(flux_state_dict.keys())
    for key in keys:
        if key.endswith(".weight"):
            module_name = ".".join(key.split(".")[:-1])
            lora_name = "lora_unet" + "_" + module_name.replace(".", "_")
            lora_name_to_module_key[lora_name] = key


    print(f"loading: {model}")
    lora_sd = load_sft(model, device=loading_device) if model.endswith("safetensors")\
                else torch.load(model, map_location='cpu')

    print(f"merging...")
    for key in list(lora_sd.keys()):
        if "lora_down" in key:
            lora_name = key[: key.rfind(".lora_down")]
            up_key = key.replace("lora_down", "lora_up")
            alpha_key = key[: key.index("lora_down")] + "alpha"

            if lora_name not in lora_name_to_module_key:
                print(f"no module found for LoRA weight: {key}.  LoRA for Text Encoder is not supported yet.")
                continue

            down_weight = lora_sd.pop(key)
            up_weight = lora_sd.pop(up_key)

            dim = down_weight.size()[0]
            alpha = lora_sd.pop(alpha_key, dim)
            scale = alpha / dim

            # W <- W + U * D
            module_weight_key = lora_name_to_module_key[lora_name]
            if module_weight_key not in flux_state_dict:
                # weight = flux_file.get_tensor(module_weight_key)
                print(f"no module found for LoRA weight: {module_weight_key}")
            else:
                weight = flux_state_dict[module_weight_key]

            weight = weight.to(working_device, merge_dtype)
            up_weight = up_weight.to(working_device, merge_dtype)
            down_weight = down_weight.to(working_device, merge_dtype)

            if len(weight.size()) == 2:
                # linear
                weight = weight + ratio * (up_weight @ down_weight) * scale
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight
                    + ratio
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = weight + ratio * conved * scale

            flux_state_dict[module_weight_key] = weight.to(loading_device, save_dtype)
            del up_weight
            del down_weight
            del weight

    if len(lora_sd) > 0:
        print(f"Unused keys in LoRA model: {list(lora_sd.keys())}")

    return flux_state_dict



def load_flow_model_diffusers(
    name: str,
    path: str,
    ipa_path: str ,
    device: str | torch.device = "cuda",
    hf_download: bool = True,
    lora_rank: int = 16,
    use_fp8: bool = False,
    additional_lora_ckpt: str | None = None,
    lora_weight: float = 1.0,
):
    # Loading Flux
    print("Init model")

    ckpt_path = os.path.join(path, "flux1-kontext-dev.safetensors") if path is not None else None
    print("Loading checkpoint from", ckpt_path)
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow.replace("sft", "safetensors"))
    

    ipa_ckpt_path = ipa_path



    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params)

    # if additional_lora_ckpt is not None:
    #     model = set_lora(model, lora_rank, device="meta" if ipa_ckpt_path is not None else device)
    assert additional_lora_ckpt is not None, "additional_lora_ckpt should have been provided. this must be a bug"

    if ckpt_path is not None:
        print("Loading lora")
        ipa_lora_sd = load_sft(ipa_ckpt_path, device=str(device)) if ipa_ckpt_path.endswith("safetensors")\
            else torch.load(ipa_ckpt_path, map_location='cpu')

        extra_lora_path = additional_lora_ckpt
        
        print("Loading main checkpoint")
        # load_sft doesn't support torch.device

        if ckpt_path.endswith('safetensors'):
            if use_fp8:
                print(
                    "####\n"
                    "We are in fp8 mode right now, since the fp8 checkpoint of XLabs-AI/flux-dev-fp8 seems broken\n"
                    "we convert the fp8 checkpoint on flight from bf16 checkpoint\n"
                    "If your storage is constrained"
                    "you can save the fp8 checkpoint and replace the bf16 checkpoint by yourself\n"
                )
                sd = load_sft(ckpt_path, device="cpu")
                sd = {k: v.to(dtype=torch.float8_e4m3fn, device=device) for k, v in sd.items()}
            else:
                sd = load_sft(ckpt_path, device=str(device))
            
            if extra_lora_path is not None:
                print("Merging extra lora to main checkpoint")
                lora_ckpt_path = extra_lora_path
                sd = merge_to_flux_model("cpu", device, sd, lora_ckpt_path, lora_weight, torch.float8_e4m3fn if use_fp8 else torch.bfloat16, torch.float8_e4m3fn if use_fp8 else torch.bfloat16)
            # # Then proceed with the update
            sd.update(ipa_lora_sd)

            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            model.to(str(device))
        else:
            dit_state = torch.load(ckpt_path, map_location='cpu')
            sd = {}
            for k in dit_state.keys():
                sd[k.replace('module.','')] = dit_state[k]

            if extra_lora_path is not None:
                print("Merging extra lora to main checkpoint")
                lora_ckpt_path = extra_lora_path
                sd = merge_to_flux_model("cpu", device, sd, lora_ckpt_path, 1.0, torch.float8_e4m3fn if use_fp8 else torch.bfloat16, torch.float8_e4m3fn if use_fp8 else torch.bfloat16)
            sd.update(ipa_lora_sd)
            
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            model.to(str(device))
        print_load_warning(missing, unexpected)

    return model


def set_lora(
    model: Flux,
    lora_rank: int,
    double_blocks_indices: list[int] | None = None,
    single_blocks_indices: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> Flux:
    double_blocks_indices = list(range(model.params.depth)) if double_blocks_indices is None else double_blocks_indices
    single_blocks_indices = list(range(model.params.depth_single_blocks)) if single_blocks_indices is None \
                            else single_blocks_indices
    
    lora_attn_procs = {}
    with torch.device(device):
        for name, attn_processor in  model.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))

            if name.startswith("double_blocks") and layer_index in double_blocks_indices:
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=model.params.hidden_size, rank=lora_rank)
            elif name.startswith("single_blocks") and layer_index in single_blocks_indices:
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=model.params.hidden_size, rank=lora_rank)
            else:
                lora_attn_procs[name] = attn_processor
    model.set_attn_processor(lora_attn_procs)
    return model




def load_t5(t5_path, device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    version = t5_path
    return HFEmbedder(version, max_length=max_length, torch_dtype=torch.bfloat16).to(device)

def load_clip(clip_path, device: str | torch.device = "cuda") -> HFEmbedder:
    version = clip_path
    return HFEmbedder(version, max_length=77, torch_dtype=torch.bfloat16).to(device)


def load_ae(flux_path, name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:

    
    if flux_path == "black-forest-labs/FLUX.1-dev" or flux_path == "black-forest-labs/FLUX.1-schnell" or flux_path == "black-forest-labs/FLUX.1-Krea-dev" or flux_path == "black-forest-labs/FLUX.1-Kontext-dev":
        ckpt_path = hf_hub_download("black-forest-labs/FLUX.1-dev", "ae.safetensors")
    else:
        ckpt_path =  os.path.join(flux_path, "ae.safetensors")
        if not os.path.exists(ckpt_path):
            # try diffusion_pytorch_model.safetensors
            ckpt_path =  os.path.join(flux_path, "vae", "ae.safetensors")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Cannot find ae checkpoint in {flux_path}/ae.safetensors or {flux_path}/vae/ae.safetensors")
            

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    # if ckpt_path is not None:
    assert ckpt_path is not None, "ckpt_path should have been provided. this must be a bug"
    sd = load_sft(ckpt_path, device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return ae