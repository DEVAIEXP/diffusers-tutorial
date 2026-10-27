import os
import torch
from diffusers import DiffusionPipeline
from sdnq import SDNQConfig  # noqa: F401

model_id = "OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_4bit"

pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.bfloat16, 
    device_map="cuda",   
)

# 1. Enable model CPU offloading to reduce GPU memory usage
pipeline.enable_model_cpu_offload() #<-- Model CPU offloading

# 2. Set deterministic seed for exact reproducibility
seed = 43
generator = torch.Generator(device="cpu").manual_seed(seed)

# 3 Prompt setup
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

# Note: Use 1-4 steps for distilled/turbo variants; 20-25 steps for base models
image = pipeline(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=8,  # Switch to 4-8 if running distilled checkpoints
    #guidance_scale=7.5 # Switch to 0 if running distilled checkpoints, Qwen-Image-2.1 dont use this parameter
    true_cfg_scale=1.0
).images[0]

output_dir = "./outputs/qwen-image-2.1"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image.save(os.path.join(output_dir, "result_sdnq.png"))