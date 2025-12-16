import dataclasses
import gc
import logging
import os
import random
from copy import deepcopy
from typing import TYPE_CHECKING, Literal, Tuple, Any, Dict, List

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from einops import rearrange
from safetensors.torch import load_file
from tqdm import tqdm

# Custom loader import
from withanyone.negativeloader_full import efficient_loader as loader

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

logger = get_logger(__name__)


@dataclasses.dataclass
class TrainArgs:
    """Arguments for training configuration."""
    # Accelerator & Logging
    project_dir: str | None = None
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_accumulation_steps: int = 1
    seed: int = 42
    wandb_project_name: str | None = None
    wandb_run_name: str | None = None

    # Model Configuration
    model_name: Literal["flux-dev", "flux-kontext", "flux-schnell", "flux-krea"] = "flux-dev"



    gradient_checkpoint: bool = True
    ema: bool = False
    ema_interval: int = 1
    ema_decay: float = 0.99

    # Optimizer
    learning_rate: float = 1e-4
    adam_betas: list[float] = dataclasses.field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # LR Scheduler
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 100
    max_train_steps: int = 200000

    # Data & Training
    batch_size: int = 1
    text_dropout: float = 0.1
    resolution: int = 512
    dataset_path: str = "./dataset/"
    
    # Checkpointing
    resume_from_checkpoint: str | None | Literal["latest"] = None
    checkpointing_steps: int = 1000

    # ID Loss Configuration
    id_loss_weight: float = 1.0
    use_GT_aligned_id_loss: bool = False
    id_weight: float = 1.0
    nce_loss_weight: float = 0.01  # Weight for NCE loss
    exneg_pool: bool = True        # Use negative pool for ID loss
    gather: bool = False           # Whether to gather tensors across devices

    # Prompt
    drop_content: str = ""

    # Architecture Choices
    arc_only: bool = True
    ipa_choice: Literal["arcface", "siglip"] = "arcface"

    #
    cp_ratio: float = 0.0  # percentage of data to use for copy-paste augmentation


def import_model_modules(ipa_choice: str):
    """Dynamically imports model modules based on the IPA choice."""
    if ipa_choice == "arcface":
        from withanyone.flux_arc.sampling import get_schedule, prepare
        from withanyone.flux_arc.util import load_ae, load_clip, load_flow_model, load_flow_model_01, load_t5
        from withanyone.flux_arc.model import Flux, SiglipEmbedding
    elif ipa_choice == "siglip":
        from withanyone.flux_sig.sampling import get_schedule, prepare
        from withanyone.flux_sig.util import load_ae, load_clip, load_flow_model, load_flow_model_01, load_t5
        from withanyone.flux_sig.model import Flux, SiglipEmbedding
    else:
        raise ValueError(f"Unknown ipa_choice: {ipa_choice}")
    
    return get_schedule, prepare, load_ae, load_clip, load_flow_model, load_flow_model_01, load_t5, Flux, SiglipEmbedding


def get_models(
    name: str, 
    device: torch.device, 
    load_funcs: tuple, 
    offload: bool = False, 
    init_01: bool = False
):
    """Loads the necessary models (T5, CLIP, Flow Model, VAE)."""
    _, _, load_ae, load_clip, load_flow_model, load_flow_model_01, load_t5, _, _ = load_funcs
    
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    
    
    
    model = load_flow_model(name, device="cpu")
        
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip


def resume_from_checkpoint(
    checkpoint_path: str | None | Literal["latest"],
    project_dir: str,
    accelerator: Accelerator,
    dit: torch.nn.Module,
    dit_ema_dict: dict | None = None,
) -> Tuple[torch.nn.Module, dict | None, int]:
    """Resumes training from a specific checkpoint or the latest one."""
    if checkpoint_path is None:
        return dit, dit_ema_dict, 0

    if checkpoint_path == "latest":
        dirs = os.listdir(project_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        if len(dirs) == 0:
            accelerator.print(f"Checkpoint '{checkpoint_path}' does not exist. Starting a new training run.")
            return dit, dit_ema_dict, 0
        path = dirs[-1]
    else:
        path = checkpoint_path

    accelerator.print(f"Resuming from checkpoint {path}")
    
    # Load ipa weights
    ipa_path = os.path.join(path, 'ipa.safetensors')
    if not os.path.exists(ipa_path):
         # Fallback to project dir structure if path is just a folder name
         ipa_path = os.path.join(project_dir, path, 'ipa.safetensors')

    ipa_state = load_file(ipa_path, device=str(accelerator.device))
    unwrapped_dit = accelerator.unwrap_model(dit)
    unwrapped_dit.load_state_dict(ipa_state, strict=False)

    # Load EMA weights if applicable
    if dit_ema_dict is not None:
        ema_path = os.path.join(path, 'ipa_ema.safetensors')
        if not os.path.exists(ema_path):
             ema_path = os.path.join(project_dir, path, 'ipa_ema.safetensors')
             
        loaded_ema = load_file(ema_path, device=str(accelerator.device))
        if dit is not unwrapped_dit:
            # Filter keys that match the current model
            dit_ema_dict = {f"module.{k}": v for k, v in loaded_ema.items() if k in unwrapped_dit.state_dict()}
        else:
            dit_ema_dict = loaded_ema

    try:
        global_step = int(path.split("-")[-1])
    except ValueError:
        global_step = 0
        
    return dit, dit_ema_dict, global_step


def main(args: TrainArgs):
    # 1. Import specific modules based on architecture choice
    model_modules = import_model_modules(args.ipa_choice)
    get_schedule, prepare, _, _, _, _, _, Flux, SiglipEmbedding = model_modules

    # 2. Import ID Loss
    if args.use_GT_aligned_id_loss:
        from withanyone.id_loss_nofa import IDLoss
    else:
        from withanyone.id_loss import IDLoss

    # 3. Setup Accelerator
    deepspeed_plugins = {
        "dit": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero2_config.json'),
        "t5": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero3_config.json'),
        "clip": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero3_config.json'),
        "siglip": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero3_config.json'),
        "vae": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero3_config.json'),
        "id_loss_fn": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero3_config.json'),
    }
    
    accelerator = Accelerator(
        project_dir=args.project_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        deepspeed_plugins=deepspeed_plugins,
        log_with="wandb",
    )
    set_seed(args.seed, device_specific=True)
    
    accelerator.init_trackers(
        project_name=args.wandb_project_name,
        config=args.__dict__,
        init_kwargs={
            "wandb": {
                "name": args.wandb_run_name,
                "dir": accelerator.project_dir,
            },
        },
    )

    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }.get(accelerator.mixed_precision, torch.float32)

    # 4. Logging Setup
    logging.basicConfig(
        format=f"[RANK {accelerator.process_index}] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True
    )
    logger.info(accelerator.state)

    # 5. Load Models
    dit, vae, t5, clip = get_models(
        name=args.model_name,
        device=accelerator.device,
        load_funcs=model_modules,
        init_01=False,
    )
    if args.ipa_choice == "siglip":
        siglip = SiglipEmbedding()

    # Freeze base models
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit.requires_grad_(False)

    # Set trainable parameters
    trainable_keywords = ["arcface", "ipa", "siglip", "learnable"]
    for name, param in dit.named_parameters():
        if any(keyword in name for keyword in trainable_keywords):
            param.requires_grad = True
            
            # Handle meta device initialization for DeepSpeed/Accelerate
            if param.device.type == 'meta':
                param.data = torch.zeros_like(param.data, device=accelerator.device)
            
    trainable_params = [name for name, param in dit.named_parameters() if param.requires_grad]
    if accelerator.is_main_process:
        print(f"Trainable parameters count: {len(trainable_params)}")

    dit.train()
    dit.gradient_checkpointing = args.gradient_checkpoint

    # 6. EMA Setup
    dit_ema_dict = None
    if args.ema:
        dit_ema_dict = {
            f"module.{k}": deepcopy(v).requires_grad_(False) 
            for k, v in dit.named_parameters() if v.requires_grad
        }

    # 7. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in dit.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=args.adam_betas,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_eps,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # 8. Resume Training
    dit, dit_ema_dict, global_step = resume_from_checkpoint(
        args.resume_from_checkpoint,
        args.project_dir,
        accelerator,
        dit,
        dit_ema_dict
    )

    # 9. Data Loader
    tar_dir_list = [
        args.dataset_path
    ]

    
    dataloader = loader(
        tar_dir_list,
        args.batch_size,
        is_distributed=(accelerator.num_processes > 1),
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        cp_ratio=args.cp_ratio,
    )
    dataloader = accelerator.prepare(dataloader)

    # 10. Prepare for Distributed Training
    id_loss_fn = IDLoss(accelerator.device)

    accelerator.state.select_deepspeed_plugin("dit")
    dit, optimizer, lr_scheduler = accelerator.prepare(dit, optimizer, lr_scheduler)
    
    accelerator.state.select_deepspeed_plugin("t5")
    t5 = accelerator.prepare(t5)
    
    accelerator.state.select_deepspeed_plugin("clip")
    clip = accelerator.prepare(clip)
    if args.ipa_choice == "siglip":
        accelerator.state.select_deepspeed_plugin("siglip")
        siglip = accelerator.prepare(siglip)
    
    accelerator.state.select_deepspeed_plugin("vae")
    vae = accelerator.prepare(vae)
    
    accelerator.state.select_deepspeed_plugin("id_loss_fn")
    id_loss_fn.netArc = accelerator.prepare(id_loss_fn.netArc)

    # 11. Noise Scheduler
    timesteps = get_schedule(
        999,
        (args.resolution // 8) * (args.resolution // 8) // 4,
        shift=True,
    )
    timesteps = torch.tensor(timesteps, device=accelerator.device)

    # 12. Training Loop
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
    )

    train_loss = 0.0

    while global_step < args.max_train_steps:
        for step, batch in enumerate(dataloader):
            if batch is None:
                continue

            # Data Preparation
            prompts = [txt_ if random.random() > args.text_dropout else args.drop_content for txt_ in batch["txt"]]
            img = batch["img"].to(accelerator.device).to(torch.bfloat16)
            arcface_embeddings = batch["arcface_embedding"].permute(1, 0, 2).to(accelerator.device)
            # gt_embeddings = batch["gt_embeddings"].to(accelerator.device)
            gt_embeddings = batch["gt_embeddings"].to(accelerator.device)
            
            negative_pool = None
            if args.exneg_pool:
                negative_pool = batch["negative_pool"].to(accelerator.device)
                
            rec_bboxes = batch["bboxes"]
            rec_bboxes_A = [b[0] for b in rec_bboxes]
            rec_bboxes_B = [b[1] for b in rec_bboxes]

            # Encode Images
            with torch.no_grad():
                x_1 = vae.encode(img.to(accelerator.device).to(torch.bfloat16))
                h = x_1.shape[2] // 2
                w = x_1.shape[3] // 2
                inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bs = img.shape[0]
                t_idx = torch.randint(0, 1000, (bs,), device=accelerator.device)
                t = timesteps[t_idx]
                
                x_0 = torch.randn_like(x_1, device=accelerator.device)
                x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0
                guidance_vec = torch.full((x_t.shape[0],), 1, device=x_t.device, dtype=x_t.dtype)
                if args.ipa_choice == "siglip":
                    siglip_embeddings = siglip([ref_img for ref_img in batch["ref_imgs"]]).to(accelerator.device, dtype=weight_dtype)
                    
                else:
                    siglip_embeddings = None

            # Forward Pass
            with accelerator.accumulate(dit):
                model_pred = dit(
                    img=x_t.to(weight_dtype),
                    img_ids=inp['img_ids'].to(weight_dtype),
                    siglip_embeddings=siglip_embeddings,
                    txt=inp['txt'].to(weight_dtype),
                    txt_ids=inp['txt_ids'].to(weight_dtype),
                    y=inp['vec'].to(weight_dtype),
                    timesteps=t.to(weight_dtype),
                    guidance=guidance_vec.to(weight_dtype),
                    arcface_embeddings=arcface_embeddings.to(weight_dtype),
                    bbox_A=rec_bboxes_A,
                    bbox_B=rec_bboxes_B,
                    use_mask=True,
                    return_map=False,
                    id_weight=args.id_weight,
                    arc_only=args.arc_only
                )

                # Diffusion Loss
                diff_loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # ID Loss Calculation
                x_pred = x_t - model_pred * t[:, None, None]
                x_pred = rearrange(x_pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)
                x_pred = vae.decode(x_pred.to(torch.bfloat16))
                x_pred_id = 127.5 * (x_pred + 1.0)

                

                if args.use_GT_aligned_id_loss:
                    generated_arcface_embeddings, gt_arcface_embeddings = id_loss_fn.get_arcface_embeddings(
                        x_pred_id.to(torch.bfloat16),
                        gt_images=img,
                        check_side_views=False,
                        original_bboxes= batch["bboxes"],
                    )
                    id_loss = id_loss_fn.compute_id_loss_with_embeddings(
                        generated_arcface_embeddings,
                        gt_arcface_embeddings, 
                        original_bboxes= batch["bboxes"],
                    )
                    force_align = False # GT aligned
                else:
                    generated_arcface_embeddings, force_align = id_loss_fn.get_arcface_embeddings(
                        x_pred_id.to(torch.bfloat16),
                        check_side_views=False,
                        original_bboxes=batch["bboxes"],
                    )
                    id_loss = id_loss_fn.compute_id_loss_with_embeddings(
                        generated_arcface_embeddings,
                        gt_embeddings,
                        force_align=force_align,
                    )


                # Format embeddings for NCE Loss
                generated_arcface_embeddings = torch.stack(
                    generated_arcface_embeddings,
                    dim=0,
                ).to(accelerator.device, dtype=weight_dtype)

                # Gather for distributed training
                if accelerator.sync_gradients and args.gather:
                    generated_arcface_embeddings_gathered = accelerator.gather(generated_arcface_embeddings)
                    gt_embeddings_gathered = accelerator.gather(gt_embeddings)
                    exneg_gathered = accelerator.gather(negative_pool) if args.exneg_pool else None
                else:
                    generated_arcface_embeddings_gathered = generated_arcface_embeddings
                    gt_embeddings_gathered = gt_embeddings
                    exneg_gathered = negative_pool if args.exneg_pool else None

                contrastive_loss = id_loss_fn.compute_info_nce_loss(
                    generated_arcface_embeddings_gathered,
                    gt_embeddings_gathered,
                    extend_negative_pool=exneg_gathered,
                )

                # Dynamic weighting based on timestep
                if force_align or t[0] > 0.6:
                    epsilon = 1e-5
                else:
                    epsilon = 1 - 1e-5

                loss = diff_loss + epsilon * args.id_loss_weight * id_loss + epsilon * contrastive_loss * args.nce_loss_weight
                
                # Logging accumulation
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Post-step operations
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    "diffusion_loss": diff_loss.detach().item(),
                    "id_loss": id_loss.detach().item(),
                    "nce": contrastive_loss.detach().item(),
                }, step=global_step)
                train_loss = 0.0

            # EMA Update
            if accelerator.sync_gradients and dit_ema_dict is not None and global_step % args.ema_interval == 0:
                src_dict = dit.state_dict()
                for tgt_name in dit_ema_dict:
                    dit_ema_dict[tgt_name].data.lerp_(src_dict[tgt_name].to(dit_ema_dict[tgt_name]), 1 - args.ema_decay)

            # Checkpointing
            if accelerator.sync_gradients and global_step % args.checkpointing_steps == 0:
                logger.info("Waiting for all processes to finish saving")
                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    save_path = os.path.join(args.project_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    logger.info(f"Saving model to {save_path}")
                    
                    unwrapped_model = accelerator.unwrap_model(dit)
                    unwrapped_model_state = unwrapped_model.state_dict()
                    requires_grad_key = [k for k, v in unwrapped_model.named_parameters() if v.requires_grad]
                    unwrapped_model_state = {k: unwrapped_model_state[k] for k in requires_grad_key}

                    accelerator.save(
                        unwrapped_model_state,
                        os.path.join(save_path, 'ipa.safetensors'),
                        safe_serialization=True
                    )
                    
                    unwrapped_opt = accelerator.unwrap_model(optimizer)
                    accelerator.save(unwrapped_opt.state_dict(), os.path.join(save_path, 'optimizer.bin'))

                    if args.ema:
                        accelerator.save(
                            {k.split("module.")[-1]: v for k, v in dit_ema_dict.items()},
                            os.path.join(save_path, 'ipa_ema.safetensors'),
                            safe_serialization=True
                        )
                    logger.info(f"Saved state to {save_path}")

                torch.cuda.empty_cache()
                gc.collect()
                torch.set_grad_enabled(True)
                dit.train()
                accelerator.wait_for_everyone()

            logs = {
                "step_loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0],
                "nce": contrastive_loss.detach().item(),
                "id_loss": id_loss.detach().item(),
                "at time step": t[0].item()
            }
            progress_bar.set_postfix(**logs)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser([TrainArgs])
    args_tuple = parser.parse_args_into_dataclasses(args_file_flag="--config")
    main(*args_tuple)