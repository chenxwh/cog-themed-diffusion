import os
from typing import List
import PIL
import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
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

        model_id = "timbrooks/instruct-pix2pix"
        MODEL_CACHE = "diffusers-cache"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            safety_checker=None,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="An image which will be repainted according to prompt",
        ),
        prompt: str = Input(
            description="Prompt to guide the image generation",
        ),
        negative_prompt: str = Input(
            description="The prompt or prompts not to guide the image generation (what you do not want to see in the generation). Ignored when not using guidance.",
            default=None,
        ),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 4], default=1
        ),
        num_inference_steps: int = Input(
            description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
            ge=1,
            le=500,
            default=100,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance. Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.",
            ge=1,
            le=20,
            default=7.5,
        ),
        image_guidance_scale: float = Input(
            description="Image guidance scale is to push the generated image towards the inital image. Higher image guidance scale encourages to generate images that are closely linked to the source image, usually at the expense of lower image quality.",
            ge=1,
            default=1.5,
        ),
        scheduler: str = Input(
            default="K_EULER_ANCESTRAL",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
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

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        image = PIL.Image.open(str(image))
        image = PIL.ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        output = self.pipe(
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
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
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
