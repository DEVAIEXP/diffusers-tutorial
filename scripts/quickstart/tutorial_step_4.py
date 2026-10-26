import os

from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig
import torch

quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer"],
)

model_id = "Qwen/Qwen-Image-2.1"  # Or distilled variants like FLUX.1-schnell

pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    dtype=torch.bfloat16, 
    device_map="cuda",
    quantization_config=quant_config,
)

# 2. Compile repetitive transformer blocks to optimize kernel execution
pipeline.transformer.compile_repeated_blocks(fullgraph=True)

# 3. Enable model CPU offloading to reduce GPU memory usage
pipeline.enable_model_cpu_offload() #<-- Model CPU offloading

# 4. Set deterministic seed for exact reproducibility
seed = 42
generator = torch.Generator(device="cpu").manual_seed(seed)

# 5. Prompt setup
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

# Note: Use 1-4 steps for distilled/turbo variants; 20-25 steps for base models
image = pipeline(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=25,  # Switch to 4-8 if running distilled checkpoints
    #guidance_scale=7.5 # Switch to 0 if running distilled checkpoints, Qwen-Image-2.1 dont use this parameter
    true_cfg_scale=5.0
).images[0]

output_dir = "./outputs/qwen-image-2.1"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image.save(os.path.join(output_dir, "result.png"))