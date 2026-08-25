import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
# sys.path.insert(0, str(SRC_DIR / "data"))
# sys.path.insert(0, str(SRC_DIR / "models"))

from diffusers import DDPMScheduler, UNet2DModel  # type: ignore

from data.data_loading import build_dataloaders_from_config
from models.scheduler import build_scheduler_from_config
from models.unet import build_unet_from_config, count_parameters
from utils.config import load_config


def resolve_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    return torch.device(device_setting)



def save_checkpoint(
        checkpoint_dir: Path,
        epoch: int,
        model: UNet2DModel,
        optimizer: torch.optim.Optimizer
    ) -> Path:
    
    epoch_dir = checkpoint_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(epoch_dir))
    
    
    training_state = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(training_state, epoch_dir / "training_state.pt")
    
    return epoch_dir



def load_checkpoint(
        checkpoint_path: Path,
        model: UNet2DModel,
        optimizer: torch.optim.Optimizer
    ) -> int:
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    
    loaded_model = UNet2DModel.from_pretrained(str(checkpoint_path))
    model.load_state_dict(loaded_model.state_dict())
    
    training_state_path = checkpoint_path / "training_state.pt"
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location="cpu")
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        completed_epoch = training_state["epoch"]
    else:
        print("Warning: no training.pt found in the checkpoint. Epoch starting from zero.")
        completed_epoch = 0
        
        
    return completed_epoch



def train_one_epoch(
        model: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_train_timesteps: int,
        max_grad_norm: float,
        log_every_n_steps: int,
        max_batches: int,
        epoch: int
    ) -> float:
    
    model.train()
    
    total_loss = 0.0
    batch_count = 0
    epoch_start_time = time.time()
    
    for batch_index, batch in enumerate(dataloader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        
        clean_images = batch["image"].to(device)
        
        noise = torch.randn_like(clean_images)
        
        current_batch_size = clean_images.shape[0]
        timesteps = torch.randint(
                                        low=0,
                                        high=num_train_timesteps,
                                        size=(current_batch_size,),
                                        device=device
                                    ).long()
        
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        noise_prediction = model(noisy_images, timesteps).sample
        
        loss = F.mse_loss(noise_prediction, noise)
        
        optimizer.zero_grad()
        loss.backward()
        
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        if log_every_n_steps > 0 and batch_index % log_every_n_steps == 0:
            elapsed_seconds = time.time() - epoch_start_time
            print(f"    [epoch {epoch}] batch {batch_index:5d}  "
                  f"loss: {loss.item():.4f}  ({elapsed_seconds:.1f}s)")
            
    
    if batch_count == 0:
        return 0.0
    
    average_loss = total_loss / batch_count
    return average_loss



@torch.no_grad()
def evaluate(
        model:UNet2DModel,
        noise_scheduler: DDPMScheduler,
        dataloader: DataLoader,
        device: torch.device,
        num_train_timesteps: int,
        max_batches: int
    ) -> float:
    
    model.eval()
    
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    
    total_loss = 0.0
    batch_count = 0
    
    for batch_index, batch in enumerate(dataloader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        
        clean_images = batch["image"].to(device)
        
        noise = torch.randn(
            clean_images.shape,
            device=device,
            generator=generator
        )
        
        current_batch_size = clean_images.shape[0]
        timesteps = torch.randint(
                                        low=0,
                                        high=num_train_timesteps,
                                        size=(current_batch_size,),
                                        device=device,
                                        generator=generator
                                    ).long()
        
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_perdiction = model(noisy_images, timesteps).sample
        loss = F.mse_loss(noise_perdiction, noise)
        
        total_loss += loss.item()
        batch_count += 1
        
    if batch_count == 0:
        return 0.0
    
    return total_loss / batch_count



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    
    
    config = load_config()
    training_config = config["training"]
    
    
    def resolve(cli_value, config_value):
            return cli_value if cli_value is not None else config_value

    epochs = resolve(args.epochs, training_config["epochs"])
    learning_rate = resolve(args.learning_rate, training_config["learning_rate"])
    device_setting = resolve(args.device, training_config["device"])
    max_batches_per_epoch = resolve(args.max_batches_per_epoch, training_config["max_batches_per_epoch"])
    
    weight_decay = training_config["weight_decay"]
    max_grad_norm = training_config["max_grad_norm"]
    save_every_n_epochs = training_config["save_every_n_epochs"]
    log_every_n_steps = training_config["log_every_n_steps"]
    num_train_timesteps = config["scheduler"]["num_train_timesteps"]
    checkpoint_dir = config["paths"]["checkpoints"]
    
    device = resolve_device(device_setting)
    
    
    print("=== Training start ===")
    print(f"    Device:             {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"    Number of epochs:   {epochs}")
    print(f"    Learning rate:      {learning_rate}")
    print(f"    Checkpoint dir:     {checkpoint_dir}")
    
    print("\n--- Building dataloaders ---")
    dataloaders = build_dataloaders_from_config(config)
    
    
    print("\n--- Building model and scheduler ---")
    model = build_unet_from_config(config)
    model.to(device) # type: ignore
    noise_scheduler = build_scheduler_from_config(config)
    
    parameter_count = count_parameters(model)
    print(f"    UNet trainable parameters:  {parameter_count}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    
    start_epoch = 1
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        completed_epoch = load_checkpoint(resume_path, model, optimizer)
        start_epoch = completed_epoch + 1
        print(f"Resuming training after epoch {start_epoch}, from: {resume_path}")
        
    
    print("\n--- Training cycle ---")
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        
        train_loss = train_one_epoch(
            model=model,
            noise_scheduler=noise_scheduler,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            num_train_timesteps=num_train_timesteps,
            max_grad_norm=max_grad_norm,
            log_every_n_steps=log_every_n_steps,
            max_batches=max_batches_per_epoch,
            epoch=epoch
        )
        
        val_loss = evaluate(
            model=model,
            noise_scheduler=noise_scheduler,
            dataloader=dataloaders["val"],
            device=device,
            num_train_timesteps=num_train_timesteps,
            max_batches=max_batches_per_epoch
        )
        
        
        epoch_duration = time.time() - epoch_start_time
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train loss: {train_loss:.4f}  "
              f"val loss: {val_loss:.4f}  "
              f"({epoch_duration:.1f}s)")
        
        
        is_sceduled_save = save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0
        is_last_epoch = epoch == epochs
        
        if is_sceduled_save or is_last_epoch:
            saved_path = save_checkpoint(checkpoint_dir, epoch, model, optimizer)
            print(f"    Checkpoint saved to: {saved_path}")
            
    print("\n--- Training finished ---")
    
    

if __name__ == "__main__":
    main()