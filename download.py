import os
import sys

import torch
from diffusers import StableDiffusionInpaintPipeline

os.makedirs("diffusers-cache", exist_ok=True)


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    cache_dir="diffusers-cache")