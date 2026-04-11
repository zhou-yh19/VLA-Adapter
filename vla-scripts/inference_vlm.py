"""
inference_vlm.py

Simple VLM inference script that loads a local pretrained model and performs image+text inference.
Uses prismatic.load() for native model loading.

$ cd /home/nas/VLA-Adapter && python vla-scripts/inference_vlm.py \
    --model_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --image_path figure/Teaser.png \
    --prompt "What is shown in this image?" \
    --max_new_tokens 256
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prismatic import load


# Set device
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_vlm_model(model_path: Union[str, Path]) -> torch.nn.Module:
    """
    Load VLM model using prismatic.load().
    
    Args:
        model_path: Model path (directory containing config.json and checkpoints/)
    
    Returns:
        Loaded VLM model
    """
    model_path = Path(model_path)
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {DEVICE}")
    
    # Load model using prismatic.load()
    model = load(model_path)
    
    # Set model to eval mode
    model.eval()
    
    print("Model loaded successfully!")
    return model


def prepare_image(image_input: Union[str, Path, np.ndarray, Image.Image], target_size: int = 224) -> Image.Image:
    """
    Prepare image input, convert to PIL Image and resize.
    
    Args:
        image_input: Image input (path, numpy array, or PIL Image)
        target_size: Target image size
    
    Returns:
        PIL Image object
    """
    # If input is a path, load the image
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    # If input is numpy array
    elif isinstance(image_input, np.ndarray):
        # Ensure uint8 type
        if image_input.dtype != np.uint8:
            image_input = (image_input * 255).astype(np.uint8)
        # Ensure HWC format
        if len(image_input.shape) == 3 and image_input.shape[-1] == 3:
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported image shape: {image_input.shape}")
    # If already PIL Image
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    # Resize
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.LANCZOS)
    
    return image


def generate_text(
    model,
    image: Union[str, Path, np.ndarray, Image.Image],
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Generate text response using VLM.
    
    Args:
        model: VLM model
        image: Input image
        prompt: Text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
    
    Returns:
        Generated text
    """
    # Prepare image
    pil_image = prepare_image(image)
    
    # Generate response using model's generate method
    with torch.inference_mode():
        generated_text = model.generate(
            image=pil_image,
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
    
    return generated_text


@dataclass
class InferenceConfig:
    """Configuration for inference script."""
    model_path: str = "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
    image_path: Optional[str] = None
    prompt: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.7


def main():
    """Main function to run inference."""
    import draccus
    
    # Parse configuration
    cfg = draccus.parse(InferenceConfig)
    
    # Check required arguments
    if not cfg.image_path or not cfg.prompt:
        print("Error: Both --image_path and --prompt are required")
        print(f"Usage: python {sys.argv[0]} --image_path <path> --prompt <text>")
        return
    
    # Load model
    model = load_vlm_model(cfg.model_path)
    
    # Run inference
    print(f"\nImage: {cfg.image_path}")
    print(f"Prompt: {cfg.prompt}\n")
    
    response = generate_text(
        model,
        cfg.image_path,
        cfg.prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
    )
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
