import sys
from pathlib import Path
from typing import Literal, get_args

import torch
from diffusers import DDPMScheduler  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config

BetaSchedule = Literal["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"]
PredictionType = Literal["epsilon", "sample", "v_prediction"]

VALID_BETA_SCHEDULES = list(get_args(BetaSchedule))
VALID_PREDICTION_TYPES = list(get_args(PredictionType))

def build_scheduler(
        num_train_timesteps: int,
        beta_schedule: BetaSchedule,
        beta_start: float,
        beta_end: float,
        prediction_type: PredictionType
) -> DDPMScheduler:
    
    
    if beta_schedule not in VALID_BETA_SCHEDULES:
        raise ValueError(
            f"Unknown beta_schedule value: {beta_schedule}."
            f"Supported values: {BetaSchedule}."
        )
        
    if prediction_type not in VALID_PREDICTION_TYPES:
        raise ValueError(
                    f"Unknown prediction_type value: {prediction_type}."
                    f"Supported values: {PredictionType}."
                )
    
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        prediction_type=prediction_type
    )
    
    return noise_scheduler



def build_scheduler_from_config(config: dict | None = None) -> DDPMScheduler:
    if config is None:
        config = load_config()
        
    
    scheduler_config = config["scheduler"]
    
    
    return build_scheduler(
        num_train_timesteps=scheduler_config["num_train_timesteps"],
        beta_schedule=scheduler_config["beta_schedule"],
        beta_start=scheduler_config["beta_start"],
        beta_end=scheduler_config["beta_end"],
        prediction_type=scheduler_config["prediction_type"],
    )
    
    
    
if __name__ == "__main__":
    config = load_config()
    scheduler_config = config["scheduler"]
    noise_scheduler = build_scheduler_from_config()
    
    
    print("\nDDPMScheduler built from config.")
    print(f"  num_train_timesteps: {scheduler_config["num_train_timesteps"]}")
    print(f"  beta_schedule:       {scheduler_config["beta_schedule"]}")
    print(f"  prediction_type:     {scheduler_config["prediction_type"]}")
    
    image_size = config["data"]["image_size"]
    in_channels = config["model"]["in_channels"]
    
    clean_image = torch.randn(1, in_channels, image_size, image_size)
    noise = torch.randn_like(clean_image)
    
    
    print("\nNoising check with different timesteps")
    
    max_timestep = scheduler_config["num_train_timesteps"] - 1
    timesteps_to_check = [0, max_timestep // 4, max_timestep // 2, max_timestep]
    
    for timestep_val in timesteps_to_check:
        timestep = torch.IntTensor([timestep_val])
        noisy_image = noise_scheduler.add_noise(clean_image, noise, timestep)
        print(f"    t = {timestep_val:4d}   ->  dispersion: {noisy_image.std().item():.4f}")