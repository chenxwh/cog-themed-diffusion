import os
from typing import List

import torch
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        model_id = "Linaqruf/anything-v3.0"
        MODEL_CACHE = "diffusers-cache"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision="diffusers", 
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="1girl, brown hair, green eyes, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden",
        ),
        negative_prompt: str = Input(
            description="The prompt or prompts not to guide the image generation (what you do not want to see in the generation). Ignored when not using guidance.",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 4], default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=12
        ),
        # scheduler: str = Input(
        #     default="DPMSolverMultistep",
        #     choices=["DDIM", "K_EULER", "DPMSolverMultistep", "K_EULER_ANCESTRAL", "PNDM", "KLMS"],
        #     description="Choose a scheduler.",
        # ),
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

        # self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs if negative_prompt is not None else None,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )


        samples = [
            output.images[i]
            for i, nsfw_flag in enumerate(output["nsfw_content_detected"])
            if not nsfw_flag
        ]

        if len(samples) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        print(
            f"NSFW content detected in {num_outputs - len(samples)} outputs, showing the rest {len(samples)} images..."
        )
        output_paths = []

        for i, sample in enumerate(samples):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths

# def make_scheduler(name, config):
#     return {
#         "PNDM": PNDMScheduler.from_config(config),
#         "KLMS": LMSDiscreteScheduler.from_config(config),
#         "DDIM": DDIMScheduler.from_config(config),
#         "K_EULER": EulerDiscreteScheduler.from_config(config),
#         "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
#         "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
#     }[name]