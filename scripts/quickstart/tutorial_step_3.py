import os

from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
import torch

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer"],
)

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", 
    dtype=torch.bfloat16, 
    device_map="cuda",
    quantization_config=quant_config,
)
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
generator = torch.Generator(device="cpu").manual_seed(42)

pipeline.enable_model_cpu_offload() #<-- Model CPU offloading
image = pipeline(prompt=prompt, num_inference_steps=25).images[0]

output_dir = "./outputs/qwen-image-2.1"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image.save(os.path.join(output_dir, "result.png"))