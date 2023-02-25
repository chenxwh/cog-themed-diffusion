# MultiDiffusion

A template for implementing [MultiDiffusion](https://github.com/chenxwh/MultiDiffusion) in [Cog](https://github.com/replicate/cog), and pushing it to Replicate.

[![Replicate](https://replicate.com/cjwbw/multidiffusion/badge)](https://replicate.com/cjwbw/multidiffusion)

First, download the weights (update the corresponding `model_id` of the model, which need to be available on HuggingFace):

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i prompt=" "

Or, push to a Replicate page:

    cog push r8.im/...
