import os
import torch
import sys
from diffusers import StableDiffusionPipeline


model_id = "nitrosocke/archer-diffusion"
cache_dir = "archer-diffusion-cache"
os.makedirs(cache_dir, exist_ok=True)


pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    use_auth_token=sys.argv[1],
)
