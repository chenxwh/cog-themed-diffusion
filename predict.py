import os
from typing import List

import torch
from diffusers import (
    StableDiffusionPanoramaPipeline,
    EulerDiscreteScheduler,
    DDIMScheduler,
)
from cog import BasePredictor, Input, Path


MODEL_ID = "stabilityai/stable-diffusion-2-base"
MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pipe = StableDiffusionPanoramaPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt", default="a photo of the dolomites"
        ),
        negative_prompt: str = Input(
            description="The prompt or prompts not to guide the image generation (what you do not want to see in the generation). Ignored when not using guidance.",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Lower the setting if out of memory.",
            default=4096,
        ),
        height: int = Input(
            description="Height of output image. Lower the setting if out of memory.",
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 4], default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DDIM",
            choices=[
                "DDIM",
                "K_EULER",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width == height == 1024:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        output_paths = []

        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


def make_scheduler(name, config):
    return {
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
    }[name]
