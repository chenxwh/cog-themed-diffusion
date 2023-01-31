## DistilGPT2 Stable Diffusion V2 

Replicate demo and API: 
[![Replicate](https://replicate.com/cjwbw/distilgpt2-stable-diffusion-v2/badge)](https://replicate.com/cjwbw/distilgpt2-stable-diffusion-v2)


A cog implementation for [distilgpt2-stable-diffusion-v2](FredZhang7/distilgpt2-stable-diffusion-v2) in [Cog](https://github.com/replicate/cog), and pushing it to Replicate.


First, download the weights (update the corresponding `model_id` of the model, which need to be available on HuggingFace):

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt=" "

Or, push to a Replicate page:

    cog push r8.im/...




