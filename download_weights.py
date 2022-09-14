import os
# import argparse
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler


# if __name__ == "__main__":



model_id = "hakurei/waifu-diffusion"
cache_dir = "waifu-diffusion-cache"
# os.makedirs(cache_dir, exist_ok=True)


# pipe = StableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     cache_dir="waifu-diffusion-cache",
#     revision="fp16",
#     torch_dtype=torch.float16,
#     use_auth_token=sys.argv[1],
# )

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    revision="fp16",
    scheduler=DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    ),
)

# print("All done!") # The permission of the cache-dir may need to change for the demo