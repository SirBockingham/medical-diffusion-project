import sys
from pathlib import Path

import torch
from diffusers import UNet2DModel  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config


def build_unet(
        sample_size: int,
        in_channels: int,
        out_channels: int,
        layers_per_block: int,
        block_out_channels: list[int],
        down_block_types: list[str],
        up_block_types: list[str]
) -> UNet2DModel:
    
    block_count = len(block_out_channels)
    
    if (len(down_block_types) != block_count) or (len(up_block_types) != block_count):
        raise ValueError("Inconsistent block count")
    
    
    model = UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=tuple(block_out_channels),
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types)
    )
    
    return model



def build_unet_from_config(config: dict | None = None) -> UNet2DModel:
    if config is None:
        config = load_config()
 
    model_config = config["model"]
    image_size = config["data"]["image_size"]
 
    return build_unet(
        sample_size=image_size,
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        layers_per_block=model_config["layers_per_block"],
        block_out_channels=model_config["block_out_channels"],
        down_block_types=model_config["down_block_types"],
        up_block_types=model_config["up_block_types"],
    )
    


def count_parameters(model: UNet2DModel) -> int:
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total += parameter.numel()
    return total



if __name__ == "__main__":
    config = load_config()
    model = build_unet_from_config(config)
    
    parameter_count = count_parameters(model)
    print("UNet2DModel built.")
    print(f"Trainable parameter count: {parameter_count:,}")
    
    model_config = config["model"]
    batch_size = 2
    sample_size = config["data"]["image_size"]
    in_channels = model_config["in_channels"]
    
    dummy_images = torch.randn(batch_size, in_channels, sample_size, sample_size)
    dummy_timesteps = torch.randint(low=0, high=1000, size=(batch_size,))
    
    print(f"\nTest input shape: {tuple(dummy_images.shape)}")
    output = model(dummy_images, dummy_timesteps).sample
    print(f"Test output shape: {tuple(output.shape)}")
    
    if tuple(output.shape) == tuple(dummy_images.shape):
        print("\nMatching input and output shapes.")
    else:
        print("\nWarning: input and output shapes do not match.")