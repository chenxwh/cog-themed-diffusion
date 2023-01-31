import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from cog import BasePredictor, Input


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading model...")

        model_id = "FredZhang7/distilgpt2-stable-diffusion-v2"
        MODEL_CACHE = "model-cache"

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "distilgpt2", cache_dir=MODEL_CACHE
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = GPT2LMHeadModel.from_pretrained(
            "FredZhang7/distilgpt2-stable-diffusion-v2",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="The beginning of the prompt",
            default="a cat sitting",
        ),
        top_k: int = Input(
            description="Number of tokens to sample from at each step with top_k sampling",
            default=50,
        ),
        max_length: int = Input(
            description="Maximum number of tokens for the output of the model",
            default=80,
        ),
        temperature: float = Input(
            description="A higher temperature will produce more diverse results, but with a higher risk of less coherent text",
            default=0.9,
        ),
        repitition_penalty: float = Input(
            description="Penalty value for each repetition of a token", default=1.2
        ),
        num_return_sequences: int = Input(
            description="Number of results to generate", default=5
        ),
    ) -> str:
        """Run a single prediction on the model"""

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        output = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repitition_penalty,
            penalty_alpha=0.6,
            no_repeat_ngram_size=1,
            early_stopping=True,
        )

        out_sequence = ""
        for out in output:
            out_sequence += self.tokenizer.decode(out, skip_special_tokens=True) + "\n\n"

        return out_sequence
