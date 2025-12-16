from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .modules.layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock, timestep_embedding, PerceiverAttentionCA, IDSelfAttention
from transformers import AutoTokenizer, AutoProcessor, SiglipModel
import math
from transformers import AutoModelForImageSegmentation
from einops import rearrange

from torchvision import transforms
from PIL import Image
from torch.cuda.amp import autocast

def create_person_cross_attention_mask(
    batch_size, num_heads, img_len, id_len,
    bbox_A_list, bbox_B_list, original_width, original_height,
    vae_scale_factor=8, patch_size=2
):

    mask = torch.zeros((batch_size, num_heads, img_len, id_len), dtype=torch.bool)
    

    latent_width = original_width // vae_scale_factor
    latent_height = original_height // vae_scale_factor
    patches_width = latent_width // patch_size
    patches_height = latent_height // patch_size
    

    def bbox_to_token_indices(bbox):
        x1, y1, x2, y2 = bbox
        

        if type(x1) == torch.Tensor:
            x1_patch = max(0, int(x1.item()) // vae_scale_factor // patch_size)
            y1_patch = max(0, int(y1.item()) // vae_scale_factor // patch_size)
            x2_patch = min(patches_width, math.ceil(int(x2.item()) / vae_scale_factor / patch_size))
            y2_patch = min(patches_height, math.ceil(int(y2.item()) / vae_scale_factor / patch_size))   
        elif type(x1) == int:
            x1_patch = max(0, x1 // vae_scale_factor // patch_size)
            y1_patch = max(0, y1 // vae_scale_factor // patch_size)
            x2_patch = min(patches_width, math.ceil(x2 / vae_scale_factor / patch_size))
            y2_patch = min(patches_height, math.ceil(y2 / vae_scale_factor / patch_size))
        else:
            raise TypeError(f"Unsupported type: {type(x1)}")
        

        indices = []
        for y in range(y1_patch, y2_patch):
            for x in range(x1_patch, x2_patch):
                idx = y * patches_width + x
                indices.append(idx)
        
        return indices

    person_A_ids = slice(0, id_len // 2)
    person_B_ids = slice(id_len // 2, id_len)
    
    for b in range(batch_size):

        bbox_A = bbox_A_list[b] if b < len(bbox_A_list) else bbox_A_list[0]
        bbox_B = bbox_B_list[b] if b < len(bbox_B_list) else bbox_B_list[0]
        

        indices_A = bbox_to_token_indices(bbox_A)
        indices_B = bbox_to_token_indices(bbox_B)
        
        for h in range(num_heads):

            for idx in indices_A:
                mask[b, h, idx, person_A_ids] = True
            

            for idx in indices_B:
                mask[b, h, idx, person_B_ids] = True
    
    return mask

def create_person_cross_attention_mask_varlen(
    batch_size, num_heads, img_len, id_len,
    bbox_lists, original_width, original_height,
    max_num_ids=2,  # Default to support 2 identities
    vae_scale_factor=8, patch_size=2
):
    """
    Create boolean attention masks limiting image tokens to interact only with corresponding person ID tokens
    
    Parameters:
    - batch_size: Number of samples in batch
    - num_heads: Number of attention heads
    - img_len: Length of image token sequence
    - id_len: Length of EACH identity embedding (not total)
    - bbox_lists: List where bbox_lists[i] contains all bboxes for batch i
                  Each batch may have a different number of bboxes/identities
    - max_num_ids: Maximum number of identities to support (for padding)
    - original_width/height: Original image dimensions
    - vae_scale_factor: VAE downsampling factor (default 8)
    - patch_size: Patch size for token creation (default 2)
    
    Returns:
    - Boolean attention mask of shape [batch_size, num_heads, img_len, total_id_len]
    """
    # Total length of ID tokens based on maximum number of identities
    total_id_len = max_num_ids * id_len
    
    # Initialize mask to block all attention
    mask = torch.zeros((batch_size, num_heads, img_len, total_id_len), dtype=torch.bool)
    
    # Calculate VAE dimensions
    latent_width = original_width // vae_scale_factor
    latent_height = original_height // vae_scale_factor
    patches_width = latent_width // patch_size
    patches_height = latent_height // patch_size


    
    # Convert boundary box to token indices
    def bbox_to_token_indices(bbox):
        x1, y1, x2, y2 = bbox
        
        # Convert to patch space coordinates
        if isinstance(x1, torch.Tensor):
            x1_patch = max(0, int(x1.item()) // vae_scale_factor // patch_size)
            y1_patch = max(0, int(y1.item()) // vae_scale_factor // patch_size)
            x2_patch = min(patches_width, math.ceil(int(x2.item()) / vae_scale_factor / patch_size))
            y2_patch = min(patches_height, math.ceil(int(y2.item()) / vae_scale_factor / patch_size))
        elif isinstance(x1, int):
            x1_patch = max(0, x1 // vae_scale_factor // patch_size)
            y1_patch = max(0, y1 // vae_scale_factor // patch_size)
            x2_patch = min(patches_width, math.ceil(x2 / vae_scale_factor / patch_size))
            y2_patch = min(patches_height, math.ceil(y2 / vae_scale_factor / patch_size))
        elif isinstance(x1, float):
            x1_patch = max(0, int(x1) // vae_scale_factor // patch_size)
            y1_patch = max(0, int(y1) // vae_scale_factor // patch_size)
            x2_patch = min(patches_width, math.ceil(x2 / vae_scale_factor / patch_size))
            y2_patch = min(patches_height, math.ceil(y2 / vae_scale_factor / patch_size))
        else:
            raise TypeError(f"Unsupported type: {type(x1)}")
        
        # Create list of all token indices in this region
        indices = []
        for y in range(y1_patch, y2_patch):
            for x in range(x1_patch, x2_patch):
                idx = y * patches_width + x
                indices.append(idx)
        
        return indices
    
    for b in range(batch_size):
        # Get all bboxes for this batch item
        batch_bboxes = bbox_lists[b] if b < len(bbox_lists) else []
        
        # Process each bbox in the batch up to max_num_ids
        for identity_idx, bbox in enumerate(batch_bboxes[:max_num_ids]):
            # Get image token indices for this bbox
            image_indices = bbox_to_token_indices(bbox)
            
            # Calculate ID token slice for this identity
            id_start = identity_idx * id_len
            id_end = id_start + id_len
            id_slice = slice(id_start, id_end)
            
            # Enable attention between this region's image tokens and the identity's tokens
            for h in range(num_heads):
                for idx in image_indices:
                    mask[b, h, idx, id_slice] = True
    
    return mask




# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )



@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class SiglipEmbedding(nn.Module):
    def __init__(self, use_matting=False):
        super().__init__()
        self.model = SiglipModel.from_pretrained("google/siglip-base-patch16-256-i18n").vision_model.to(torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-i18n")
        self.model.to(torch.cuda.current_device())
        
        # BiRefNet matting setup
        self.use_matting = use_matting
        if self.use_matting:
            self.birefnet = AutoModelForImageSegmentation.from_pretrained(
                'briaai/RMBG-2.0', trust_remote_code=True).to(torch.cuda.current_device(), dtype=torch.bfloat16)
            # Apply half precision to the entire model after loading
            # self.birefnet = self.birefnet.half()
            self.matting_transform = transforms.Compose([
                # transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


    def apply_matting(self, image):
        """Apply BiRefNet matting to remove background from image"""
        if not self.use_matting:
            return image
            
        # Convert to input format and move to GPU
        input_image = self.matting_transform(image).unsqueeze(0).to(torch.cuda.current_device(), dtype=torch.bfloat16)


        # Generate prediction
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            preds = self.birefnet(input_image)[-1].sigmoid().cpu()
        
        # Process the mask
        pred = preds[0].squeeze().float()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        binary_mask = mask.convert("L")
        
        # Create a new image with black background
        result = Image.new("RGB", image.size, (0, 0, 0))
        result.paste(image, (0, 0), binary_mask)


        
        return result

    def get_id_embedding(self, refimage):
        """
        Extracts identity embeddings from reference images using SigLIP.
        Processes the entire batch in parallel for efficiency.
        
        Args:
            refimage: List[List[PIL.Image]] of shape (batch_size, num_persons)
                        OR torch.Tensor of shape (batch_size, num_persons, C, H, W)
        
        Returns:
            torch.Tensor: Embeddings of shape (batch_size, num_persons, 256, 768)
        """
        if self.use_matting:
            # Convert the model to half precision if matting is used
            if hasattr(self, "convert_birefnet_to_half"):
                self.convert_birefnet_to_half()

        # 1. Flatten inputs to a single list of images for batch processing
        # We want a flat list ordered by: [Batch0_Person0, Batch0_Person1, Batch1_Person0, ...]
        if isinstance(refimage, list):
            batch_size = len(refimage)
            if batch_size == 0:
                return torch.empty(0, 0, 256, 768, device=torch.cuda.current_device(), dtype=torch.bfloat16)
            num_persons = len(refimage[0])
            # Flatten: Iterate batch first, then persons. 
            # Result: [B0_P0, B0_P1, B1_P0, B1_P1, ...]
            flat_images = [img for batch in refimage for img in batch]
            
            if len(flat_images) != batch_size * num_persons:
                raise ValueError(f"Inconsistent number of persons across batch. Expected {batch_size}*{num_persons}={batch_size*num_persons}, got {len(flat_images)}")
                
        elif isinstance(refimage, torch.Tensor):
            batch_size, num_persons = refimage.shape[:2]
            # Ensure contiguous memory before view
            refimage = refimage.contiguous()
            # Flatten tensor to (B*N, ...)
            # Result order matches memory layout: B0_P0, B0_P1, B1_P0...
            flat_tensor = refimage.view(-1, *refimage.shape[2:])
            # Convert to list of tensors for consistency
            flat_images = [img for img in flat_tensor]
        else:
            raise ValueError(f"Unsupported input type for refimage: {type(refimage)}")

        # 2. Apply Matting (if enabled)
        if self.use_matting:
            # Apply matting to each image in the flattened list
            flat_images = [self.apply_matting(img) for img in flat_images]

        # 3. Process images (Batch Parallel)
        # Processor handles list of PIL images or Tensors
        inputs = self.processor(images=flat_images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch.cuda.current_device(), dtype=torch.bfloat16)

        # 4. Forward Pass through SigLIP
        # Output shape: (batch_size * num_persons, seq_len, hidden_dim)
        outputs = self.model(pixel_values)
        last_hidden_state = outputs.last_hidden_state

        # 5. Reshape / Rearrange
        # Explicitly map the flattened dimension (b*n) back to (b, n)
        # Since our input order was B0_P0, B0_P1..., the decomposition (b n) implies n is the inner dimension.
        embeddings = rearrange(last_hidden_state, '(b n) s d -> b n s d', b=batch_size, n=num_persons)

        return embeddings
    
    def forward(self, refimage):
        return self.get_id_embedding(refimage)

class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.gradient_checkpointing = False




        # use cross attention
        self.ipa = nn.ModuleList([
            PerceiverAttentionCA(dim=self.hidden_size, kv_dim=self.hidden_size, heads=self.num_heads) 
            for _ in range(self.params.depth_single_blocks + self.params.depth)
        ])



        self.arcface_in = nn.Sequential(
            nn.Linear(512, 4 * self.hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(4 * self.hidden_size),
            nn.Linear(4 * self.hidden_size, 8 * self.hidden_size, bias=True),
        )

        self.siglip_in = nn.Sequential(
            nn.Linear(768, self.hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
        )
        

    def lq_in(self, txt_lq, siglip_embeddings, arcface_embeddings, arc_only=False):
        """
        Process the siglip and arcface embeddings.
        """

        
        # shape of arcface: (num_refs, bs, 512)
        arcface_embeddings = self.arcface_in(arcface_embeddings)  

        arcface_embeddings =  rearrange(arcface_embeddings, 'b n (t d) -> b n t d', t=8, d=self.hidden_size)
        # (num_ref, tokens, hidden_size) -> (bs, num_refs*tokens, hidden_size)
        # siglip_embeddings = rearrange(siglip_embeddings, 'b n h d -> b (n h) d')
        arcface_embeddings = arcface_embeddings.permute(1, 0, 2, 3) # (n, b, t, d) -> (b, n, t, d)
        siglip_embeddings = self.siglip_in(siglip_embeddings)  # (bs, num_refs, 256, 768) -> (bs, num_refs, 4*hidden_size)


        # concat in token dimension
        arcface_embeddings = torch.cat((siglip_embeddings, arcface_embeddings), dim=2)  # (bs, num_refs, 4, hidden_size) cat (bs, num_refs, 4, hidden_size) -> (bs, num_refs, 8, hidden_size)


        arcface_embeddings = rearrange(arcface_embeddings, 'b n t d -> b (n t) d')

        return arcface_embeddings



    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self):
        # set recursively
        processors = {}  # type: dict[str, nn.Module]

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)



    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        # ref_img: Tensor | None = None, 
        # ref_img_ids: Tensor | None = None, 
        siglip_embeddings: Tensor | None = None, # (bs, num_refs, 256, 768)
        skip_layers: list[int] = None,
        arcface_embeddings: Tensor | None = None, # (bs, num_refs, 512)
        # mask: Tensor | None = None, # create mask right here
        bbox_A: Tensor | None = None, # (bs,  4)
        bbox_B: Tensor | None = None, # (bs, 4)
        bbox_lists: list | None = None, # list of list of bboxes, bbox_lists[i] is for the i-th batch, each has different number of bboxes (ids), which should align with the dim1 of arcface_embeddings. This is used to replace bbox_A and bbox_B, which should be discarded, but remained for compatibility.
        use_mask: bool = True,
        return_map: bool = False,
        shortcut: bool = False,
        id_weight: float = 1.0,
        arc_only: bool = False,
        max_num_ids: int = 2,  # max number of identities in a batch, used when bbox_lists is not None
        img_height: int = 512,
        img_width: int = 512,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        text_length = txt.shape[1]
        img_length = img.shape[1]

        #        concat ref_img/img
        img_end = img.shape[1]

        use_ip = arcface_embeddings is not None

        if use_ip:

            id_embeddings = self.lq_in(None, siglip_embeddings, arcface_embeddings, arc_only=arc_only)  
            text_length = txt.shape[1]  # update text_length after adding learnable query

            id_len = 8 + 256

            if bbox_A is not None and use_mask and bbox_lists is None:
                
                mask = create_person_cross_attention_mask(
                    batch_size=img.shape[0],
                    num_heads=self.params.num_heads,
                    # txt_len=text_length,
                    img_len=img_length,
                    id_len=id_len * 2 if bbox_B is not None else id_len,  # if bbox_B 
                    bbox_A_list=bbox_A,
                    bbox_B_list=bbox_B,
                    original_width=img_width,
                    original_height=img_height,
                ).to(img.device)
            elif bbox_lists is not None and use_mask:

                mask = create_person_cross_attention_mask_varlen(
                    batch_size=img.shape[0],
                    num_heads=self.params.num_heads,
                    # txt_len=text_length,
                    img_len=img_length,
                    id_len=id_len,  # if bbox_B is None, then we only have one person
                    bbox_lists=bbox_lists,
                    # max_num_ids=max_num_ids,
                    max_num_ids=len(bbox_lists[0]),
                    original_width=img_width,
                    original_height= img_height,
                ).to(img.device)
            else:
                mask = None
        else:
            mask = None

            
            # update text_ids and id_ids
            txt_ids =  torch.zeros((txt.shape[0], text_length, 3)).to(img_ids.device)  
        ids = torch.cat((txt_ids, img_ids), dim=1)  # (bs, T + I + ID, 3) 

        pe = self.pe_embedder(ids)


        # ipa
        ipa_idx = 0

        for index_block, block in enumerate(self.double_blocks):
            if self.training and self.gradient_checkpointing:

                img, txt = torch.utils.checkpoint.checkpoint(
                    block,
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe, 
                    text_length=text_length,
                    image_length=img_length,
                    return_map = False,
                    use_reentrant=False,
                )

            else:

                img, txt= block(
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe,
                    text_length=text_length,
                    image_length=img_length,
                    return_map=False,
                )

            if use_ip:
                img = img + id_weight * self.ipa[ipa_idx](id_embeddings, img, mask)
                ipa_idx += 1 

        # for block in self.single_blocks:
        img = torch.cat((txt, img), 1)

        for index_block, block in enumerate(self.single_blocks):
            if self.training and self.gradient_checkpointing:
                img = torch.utils.checkpoint.checkpoint(
                    block,
                    img, vec=vec, pe=pe, #mask=mask,
                    text_length=text_length,
                    image_length=img_length,
                    use_reentrant=False
                )

            else:
                img = block(img, vec=vec, pe=pe,text_length=text_length, image_length=img_length)




            # IPA
            if use_ip:
                txt, real_img = img[:, :text_length, :], img[:, text_length:, :]
                id_ca = id_weight * self.ipa[ipa_idx](id_embeddings, real_img, mask)

                real_img = real_img + id_ca
                img = torch.cat((txt, real_img), dim=1)
                ipa_idx += 1
        


       

        img = img[:, txt.shape[1] :, ...]

        img = img[:, :img_end, ...]

        img = self.final_layer(img, vec)  

        return img
    