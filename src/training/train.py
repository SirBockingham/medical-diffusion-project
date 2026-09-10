import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from diffusers import DDPMScheduler, UNet2DModel  # type: ignore
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

from data.data_loading import build_dataloaders_from_config
from data.dataset import MedicalImageDataset
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
        optimizer: torch.optim.Optimizer,
        class_names: list[str] | None = None,
        ema_model: EMAModel | None = None
    ) -> Path:
    
    epoch_dir = checkpoint_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
        model.save_pretrained(str(epoch_dir))
        ema_model.restore(model.parameters())
        
        raw_weights_dir = epoch_dir / "raw_weights"
        raw_weights_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(raw_weights_dir))
    else:
        model.save_pretrained(str(epoch_dir))
    
    
    training_state = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict()
    }
    
    if ema_model is not None:
        training_state["ema_state_dict"] = ema_model.state_dict()
        
    torch.save(training_state, epoch_dir / "training_state.pt")
    
    if class_names is not None:
        class_names_path = epoch_dir / "class_names.json"
        with open(class_names_path, "w", encoding="utf-8") as f:
            json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)
    
    return epoch_dir



def load_checkpoint(
        checkpoint_path: Path,
        model: UNet2DModel,
        optimizer: torch.optim.Optimizer,
        ema_model: EMAModel | None = None
    ) -> int:
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    
    raw_weights_dir = checkpoint_path / "raw_weights"
    if raw_weights_dir.exists():
        weights_source = raw_weights_dir
    else:
        weights_source = checkpoint_path
    
    loaded_model = UNet2DModel.from_pretrained(str(checkpoint_path))
    model.load_state_dict(loaded_model.state_dict())
    
    training_state_path = checkpoint_path / "training_state.pt"
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location="cpu")
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        completed_epoch = training_state["epoch"]
        
        if ema_model is not None:
            if "ema_state_dict" in training_state:
                ema_model.load_state_dict(training_state["ema_state_dict"])
            else:
                print("Warning: checkpoint does not contain EMA state! EMA starting from current weights.")
    else:
        print("Warning: no training.pt found in the checkpoint! Epoch starting from zero.")
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
        epoch: int,
        conditional: bool = False,
        label_dropout_chance: float = 0.0,
        null_class_index: int = 0,
        ema_model: EMAModel | None = None,
        lr_scheduler = None
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
        
        
        if conditional:
            class_labels = batch["class_index"].to(device)
            
            if label_dropout_chance > 0:
                dropout_mask = torch.rand(class_labels.shape, device=device) < label_dropout_chance
                class_labels = torch.where(
                    dropout_mask,
                    torch.full_like(class_labels, null_class_index),
                    class_labels
                )
                
            noise_prediction = model(noisy_images, timesteps, class_labels=class_labels).sample
        else:
            noise_prediction = model(noisy_images, timesteps).sample
        
        
        loss = F.mse_loss(noise_prediction, noise)
        
        optimizer.zero_grad()
        loss.backward()
        
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        if ema_model is not None:
            ema_model.step(model.parameters())
        
        total_loss += loss.item()
        batch_count += 1
        
        if log_every_n_steps > 0 and batch_index % log_every_n_steps == 0:
            elapsed_seconds = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"    [epoch {epoch}] batch {batch_index:5d}  "
                  f"loss: {loss.item():.4f}     lr: {current_lr:.2e}  "
                  f"({elapsed_seconds:.1f}s)")
            
    
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
        max_batches: int,
        conditional: bool = False,
        ema_model: EMAModel | None = None
    ) -> float:
    
    model.eval()
    
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    
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
        
        
        if conditional:
            class_labels = batch["class_index"].to(device)
            noise_prediction = model(noisy_images, timesteps, class_labels=class_labels).sample
        else:
            noise_prediction = model(noisy_images, timesteps).sample
            
            
        loss = F.mse_loss(noise_prediction, noise)
        
        total_loss += loss.item()
        batch_count += 1
    
    if ema_model is not None:
        ema_model.restore(model.parameters())
        
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
    
    conditional = config["conditional"]["enabled"]
    label_dropout_chance = config["conditional"]["label_dropout_chance"]
    
    use_ema = training_config["use_ema"]
    ema_decay = training_config["ema_decay"]
    lr_warmup_steps = training_config["lr_warmup_steps"]
    
    device = resolve_device(device_setting)
    
    
    print("=== Training start ===")
    print(f"    Device:             {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"    Number of epochs:   {epochs}")
    print(f"    Learning rate:      {learning_rate}")
    print(f"    Checkpoint dir:     {checkpoint_dir}")
    if conditional:
        print(f"    Mode:               conditional (label_dropout: {label_dropout_chance})")
    else:
        print("    Mode:               unconditional")
    if use_ema:
        print(f"    EMA:                on (decay: {ema_decay})")
    else:
        print("     EMA:                off")
    if lr_warmup_steps > 0:
        print(f"    LR warmup:          {lr_warmup_steps} steps + cosine decay")
    
    print("\n--- Building dataloaders ---")
    dataloaders = build_dataloaders_from_config(config)
    
    
    train_dataset: MedicalImageDataset = dataloaders["train"].dataset # type: ignore
    if conditional:
        num_class_embeds = train_dataset.num_class_embeds
        class_names = train_dataset.class_names
        null_class_index = train_dataset.null_class_index
    else:
        num_class_embeds = None
        class_names = None
        null_class_index = 0
    
    
    print("\n--- Building model and scheduler ---")
    model = build_unet_from_config(config, num_class_embeds=num_class_embeds)
    model.to(device) # type: ignore
    noise_scheduler = build_scheduler_from_config(config)
    
    parameter_count = count_parameters(model)
    print(f"    UNet trainable parameters:  {parameter_count}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    if use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=ema_decay,
            use_ema_warmup=True,
            inv_gamma=1.0,
            power=3.0 / 4.0,
            model_cls=UNet2DModel,
            model_config=model.config,
        )
    else:
        ema_model = None
        
    batches_per_epoch = len(dataloaders["train"])
    if max_batches_per_epoch > 0:
        batches_per_epoch = min(batches_per_epoch, max_batches_per_epoch)
    total_training_steps = batches_per_epoch * epochs
 
    if lr_warmup_steps > 0:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = None
        
    
    start_epoch = 1
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        completed_epoch = load_checkpoint(resume_path, model, optimizer, ema_model)
        start_epoch = completed_epoch + 1
        print(f"Resuming training after epoch {start_epoch}, from: {resume_path}")
        
    
    print("\n--- Training cycle ---")
    training_start = time.time()
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
            epoch=epoch,
            conditional=conditional,
            label_dropout_chance=label_dropout_chance,
            null_class_index=null_class_index,
            ema_model=ema_model,
            lr_scheduler=lr_scheduler
        )
        
        val_loss = evaluate(
            model=model,
            noise_scheduler=noise_scheduler,
            dataloader=dataloaders["val"],
            device=device,
            num_train_timesteps=num_train_timesteps,
            max_batches=max_batches_per_epoch,
            conditional=conditional,
            ema_model=ema_model
        )
        
        
        epoch_duration = time.time() - epoch_start_time
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train loss: {train_loss:.4f}  "
              f"val loss: {val_loss:.4f}  "
              f"({epoch_duration:.1f}s)")
        
        
        is_sceduled_save = save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0
        is_last_epoch = epoch == epochs
        
        if is_sceduled_save or is_last_epoch:
            saved_path = save_checkpoint(checkpoint_dir, epoch, model, optimizer, class_names=class_names, ema_model=ema_model)
            print(f"    Checkpoint saved to: {saved_path}")
            
    print(f"\n--- Training finished in {time.time() - training_start:.1f}s ---")
    
    

if __name__ == "__main__":
    main()
