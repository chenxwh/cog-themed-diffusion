# Themed Stable diffusion

[![Replicate](https://replicate.com/cjwbw/supermarionation/badge)](https://replicate.com/cjwbw/supermarionation)

A template for implementing customised stable diffusion model in [Cog](https://github.com/replicate/cog), and pushing it to Replicate.


First, download the weights (update the corresponding `model_id` of the model, which need to be available on HuggingFace):

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt=" "

Or, push to a Replicate page:

    cog push r8.im/...
