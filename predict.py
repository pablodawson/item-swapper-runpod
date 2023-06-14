''' StableDiffusion-v1 Predict Module '''

import os
from typing import List
import time
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler
)
import json
from utils import *

from PIL import Image
from cog import BasePredictor, Input, Path
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    '''Predictor class for StableDiffusion-v1'''

    def setup(self):
        '''
        Load the model into memory to make running multiple predictions efficient
        '''
        print("Loading pipeline...")

        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                            "runwayml/stable-diffusion-inpainting",
                            cache_dir=MODEL_CACHE,
                            local_files_only=True,
                            ).to("cuda")
        
        self.inpaint_pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        width: int = Input(
            description="Output image width; max 1024x768 or 768x1024 due to memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        image: Path = Input(
            description="Image to inpaint on.",
            default=None,
        ),
        seg: Path = Input(
            description="Segmentation map to base our masks on.",
            default=None,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="K-LMS",
            choices=["DDIM", "DDPM", "DPM-M", "DPM-S", "EULER-A", "EULER-D",
                     "HEUN", "IPNDM", "KDPM2-A", "KDPM2-D", "PNDM",  "K-LMS"],
            description="Choose a scheduler. If you use an init image, PNDM will be used",
        ),
        swap : str = Input(
            description="Items to be swapped, json format",
            default=None
        ),
        output_format: str = Input(
            description="Output format",
            default="all-in-one",
            choices=["all-in-one", "instances"]
        ),
        lora_folder: str = Input(
            description="Folder where the loras are stored",
            default="loras"
        )
    ) -> List[Path]:
        '''
        Run a single prediction on the model
        '''        
        # Convert swap json to list
        swap = json.loads(swap)

        seed = int.from_bytes(os.urandom(2), "big")

        if not image:
            raise ValueError("No image provided")

        image = Image.open(image).convert("RGB")
        seg = Image.open(seg).convert("RGB")

        #Resize image while mantaining aspect ratio
        if min(image.width, image.height) < 512:
            ratio = 512 / min(image.width, image.height)
            image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.BICUBIC)

        self.inpaint_pipe.scheduler = make_scheduler(scheduler, self.inpaint_pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)
        
        output_dirs = []
        
        if (not os.path.exists("tmp")):
            os.mkdir("tmp")
        
        for item in swap:
            lora = item.get("lora")
            weight = item.get("weight", 1.35)
            prompt = item.get("prompt", f"A photo of {lora}")
            color = item.get("color")
            token = item.get("token", lora)

            mask = create_mask(np.array(seg), color, convex_hull=item.get("convex_hull", False))

            image_ref, mask_ref, image_bbox = create_reference_images(image, mask, min_size=200, padding=30, width=width)

            image_ref.save("test.jpg")
            mask_ref.save("test_mask.jpg")
            
            timestart = time.time()
            apply_lora(self.inpaint_pipe, f"{lora_folder}/{lora}.safetensors", weight=weight)
            print("Time to load lora: ", time.time() - timestart)

            timestart = time.time()

            output = self.inpaint_pipe(prompt, 
                        image_ref.resize((width,width)), num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, mask_image=mask_ref, 
                        width=width, height=width).images[0]
            output = output.resize(image_ref.size)

            # "Pegar" la imagen solo en la mascara
            if (output_format == "all-in-one"):
                paste_mask = mask_ref.resize(image_ref.size).convert("L")
                image.paste(output, image_bbox, paste_mask)

            # Agregar la imagen con mascara a la lista de outputs
            else:
                output = apply_mask(output, mask)
                output.save(f"tmp/{token}.png", optimize=True, quality=50)
                output_dirs.append(f"tmp/{token}.png")
            
            print("Time to inpaint item: ", time.time() - timestart)
        
        if (output_format == "all-in-one"):
            output_path = f"tmp/output.png"
            image.save(output_path, optimize=True, quality=30)
            output_dirs.append(output_path)

        return output_dirs


def make_scheduler(name, config):
    '''
    Returns a scheduler from a name and config.
    '''
    return {
        "DDIM": DDIMScheduler.from_config(config),
        "DDPM": DDPMScheduler.from_config(config),
        "DPM-M": DPMSolverMultistepScheduler.from_config(config),
        "DPM-S": DPMSolverSinglestepScheduler.from_config(config),
        "EULER-A": EulerAncestralDiscreteScheduler.from_config(config),
        "EULER-D": EulerDiscreteScheduler.from_config(config),
        "HEUN": HeunDiscreteScheduler.from_config(config),
        "IPNDM": IPNDMScheduler.from_config(config),
        "KDPM2-A": KDPM2AncestralDiscreteScheduler.from_config(config),
        "KDPM2-D": KDPM2DiscreteScheduler.from_config(config),
        "PNDM": PNDMScheduler.from_config(config),
        "K-LMS": LMSDiscreteScheduler.from_config(config)
    }[name]
