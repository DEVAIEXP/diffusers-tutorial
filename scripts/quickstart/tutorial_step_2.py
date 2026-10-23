import os

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image", 
    dtype=torch.bfloat16, 
    device_map="cuda"
)
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
generator = torch.Generator(device="cpu").manual_seed(42)

image = pipeline(prompt=prompt, num_inference_steps=25).images[0]

output_dir = "./outputs/z_image"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image.save(os.path.join(output_dir, "result.png"))