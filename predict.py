''' StableDiffusion-v1 Predict Module '''

import os
from typing import List

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
                            torch_dtype=torch.float16,
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
        swap : list = Input(
            description="Items to be swapped",
            default=None
        )
    ) -> Path:
        '''
        Run a single prediction on the model
        '''        
        seed = int.from_bytes(os.urandom(2), "big")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits."
            )

        extra_kwargs = {}

        if not image:
            raise ValueError("No image provided")

        pipe = self.inpaint_pipe

        image = Image.open(image).convert("RGB")
        seg = Image.open(seg).convert("RGB")

        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        height = int(width * image.height / image.width)
        height = height - (height % 8)
        
        image = image.resize((width, height), Image.BILINEAR)

        for item in swap:
            lora = item.get("lora")
            weight = item.get("weight", 1.35)
            prompt = item.get("prompt", f"A photo of {lora}")
            color = item.get("color")

            mask = create_mask(np.array(seg), color, convex_hull=item.get("convex_hull", False)).resize((width, height))
            apply_lora(pipeline, f"loras/{lora}.safetensors", weight=weight)

            timestart = time.time()
        
            image = pipeline(prompt, 
                        image, num_inference_steps=inference_steps, guidance_scale=guidance_scale, mask_image=mask, 
                        width=width, height=height).images[0]

            print("Time to add item: ", time.time() - timestart)
        
        output_path = f"/tmp/output.png"
        image.save(output_path)

        return output_path


def make_scheduler(name, config):
    '''
    Returns a scheduler from a name and config.
    '''
    return {
        "DDIM": DDIMScheduler.from_config(config),
        "DDPM": DDPMScheduler.from_config(config),
        # "DEIS": DEISMultistepScheduler.from_config(config),
        "DPM-M": DPMSolverMultistepScheduler.from_config(config),
        "DPM-S": DPMSolverSinglestepScheduler.from_config(config),
        "EULER-A": EulerAncestralDiscreteScheduler.from_config(config),
        "EULER-D": EulerDiscreteScheduler.from_config(config),
        "HEUN": HeunDiscreteScheduler.from_config(config),
        "IPNDM": IPNDMScheduler.from_config(config),
        "KDPM2-A": KDPM2AncestralDiscreteScheduler.from_config(config),
        "KDPM2-D": KDPM2DiscreteScheduler.from_config(config),
        # "KARRAS-VE": KarrasVeScheduler.from_config(config),
        "PNDM": PNDMScheduler.from_config(config),
        # "RE-PAINT": RePaintScheduler.from_config(config),
        # "SCORE-VE": ScoreSdeVeScheduler.from_config(config),
        # "SCORE-VP": ScoreSdeVpScheduler.from_config(config),
        # "UN-CLIPS": UnCLIPScheduler.from_config(config),
        # "VQD": VQDiffusionScheduler.from_config(config),
        "K-LMS": LMSDiscreteScheduler.from_config(config)
    }[name]