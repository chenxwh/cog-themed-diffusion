import os
from subprocess import call
import shutil
from typing import List
from zipfile import ZipFile
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from pytorch_lightning import seed_everything
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        custmoised_model_url: str = Input(
            description="Url of the ZIP file containing the model files",
            default="https://replicate.delivery/pbxt/zEnutTFTKUqGFBfum31OcIGMTUGzEUQVFTNfwjw0v1Rpw38PA/output.zip",
        ),
        new_ckpt: bool = Input(
            description="set true if your are uploading a different model (see below)",
            default=False,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="a magical princess with golden hair, archer style",
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
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        print("Loading pipeline...")

        cog_custmoised_model_zip = "cog_custmoised_model.zip"
        if new_ckpt or not os.path.exists(cog_custmoised_model_zip):
            run_cmd(f"wget -O {cog_custmoised_model_zip} {custmoised_model_url}")

        cog_custmoised_model = "cog_custmoised_model"

        if os.path.exists(cog_custmoised_model):
            shutil.rmtree(cog_custmoised_model)
        os.makedirs(cog_custmoised_model)

        with ZipFile(cog_custmoised_model_zip, "r") as zip_ref:
            zip_ref.extractall(cog_custmoised_model)

        # if this is a zip file uploaded from different epochs, need to future decide an epoch
        model_path = cog_custmoised_model
        file_list = os.listdir(cog_custmoised_model)
        if "model_index.json" not in file_list:
            subdir = str(max([int(epoch) for epoch in file_list]))
            model_path = f"{cog_custmoised_model}/{subdir}"

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch.float16,
        ).to("cuda")

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width == height == 1024:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        seed_everything(seed)

        output = pipe(
            prompt=[prompt] * num_outputs,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        output_paths = []
        for i, sample in enumerate(output["images"]):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)
