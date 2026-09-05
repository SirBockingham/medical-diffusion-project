import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from diffusers import DDPMScheduler, UNet2DModel  # type: ignore

from models.scheduler import build_scheduler_from_config
from utils.config import load_config


def resolve_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    return torch.device(device_setting)



def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint folder not found: {checkpoint_dir}"
        )
        
    checkpoint_folders = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("epoch_"):
            checkpoint_folders.append(item)
            
    if len(checkpoint_folders) == 0:
        raise FileNotFoundError(
            f"No checkpoints found in: {checkpoint_dir}"
        )
        
    checkpoint_folders.sort(key=lambda folder: folder.name)
    return checkpoint_folders[-1]



def load_class_names(checkpoint_path : Path) -> list[str] | None:
    class_names_path = checkpoint_path / "class_names.json"
    if not class_names_path.exists():
        return None
    
    with open(class_names_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data["class_names"]



def resolve_label_to_index(label: str, class_names: list[str]) -> int:
    for index, class_name in enumerate(class_names):
        if class_name.lower() == label.lower():
            return index
    
    raise ValueError(
        f"Unknown label: {label}.\n"
        f"Available labels: {class_names}"
    )



def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    image_tensor = (image_tensor / 2.0) + 0.5
    image_tensor = image_tensor.clamp(0.0, 1.0)
    image_tensor = (image_tensor * 255.0).round().to(torch.uint8)
    return image_tensor



@torch.no_grad()
def generate_batch(
        model: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        batch_size: int,
        image_size: int,
        in_channels: int,
        num_inference_steps: int,
        device: torch.device,
        generator: torch.Generator | None,
        class_index: int | None = None,
        null_class_index: int | None = None,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
    model.eval()
    
    image = torch.randn(
        (batch_size, in_channels, image_size, image_size),
        device=device,
        generator=generator
    )
    
    noise_scheduler.set_timesteps(num_inference_steps)
    
    total_steps = len(noise_scheduler.timesteps)
    start_time = time.time()
    
    conditional_labels = None
    null_labels = None
    
    if class_index is not None:
        conditional_labels = torch.full((batch_size,), class_index, device=device, dtype=torch.long)
        
        if guidance_scale > 1.0 and null_class_index is not None:
            null_labels = torch.full((batch_size,), null_class_index, device=device, dtype=torch.long)
    
    
    use_guidance = null_labels is not None
    use_conditioning = conditional_labels is not None
    
    for step_index, timestep in enumerate(noise_scheduler.timesteps):
        if use_guidance:
            #Classifier-Free Guidance
            conditional_prediction = model(image, timestep, class_labels=conditional_labels).sample
            unconditional_prediction = model(image,timestep, class_labels=null_labels).sample
            noise_prediction = unconditional_prediction + guidance_scale * (conditional_prediction - unconditional_prediction)
        elif use_conditioning:
            noise_prediction = model(image, timestep, class_labels=conditional_labels).sample
        else:
            noise_prediction = model(image, timestep).sample
        
        
        image = noise_scheduler.step(noise_prediction, timestep, image).prev_sample
        
        is_progress_step = step_index % max(1, total_steps // 10) == 0
        if is_progress_step:
            elapsed = time.time() - start_time
            percent = 100 * step_index / total_steps
            print(f"    denoising: {step_index:4d}/{total_steps}"
                  f"({percent:.0f}, {elapsed:.1f})")
            
    return image
    
    
    
def save_images(
        images: torch.Tensor,
        output_dir: Path,
        start_index: int,
    ) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    image_uint8 = denormalize(images.cpu())
    
    for offset in range(image_uint8.shape[0]):
        single_image = image_uint8[offset]
        
        image_array = single_image.permute(1, 2, 0).numpy()
        
        
        if image_array.shape[2] == 3:
            pil_image = Image.fromarray(image_array, mode="RGB").convert("L")
        else:
            pil_image = Image.fromarray(image_array[:, :, 0], mode="L")
        
        image_index = start_index + offset
        save_path = output_dir / f"sample_{image_index:04d}.png"
        pil_image.save(save_path)
        saved_paths.append(save_path)
        
    return saved_paths



def save_grid(images: torch.Tensor, output_path: Path) -> None:
    images_uint8 = denormalize(images.cpu())
    image_count = images_uint8.shape[0]
    
    columns = math.ceil(math.sqrt(image_count))
    rows = math.ceil(image_count / columns)
    
    _figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    
    
    if image_count == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flatten())
    
    for index in range(len(axes_list)):
        axis = axes_list[index]
        axis.axis("off")
        
        if index < image_count:
            image_array = images_uint8[index].permute(1, 2, 0).numpy()
            if image_array.shape[2] == 3:
                axis.imshow(image_array)
            else:
                axis.imashow(image_array[:, :, 0], cmap="gray")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100)
    plt.close()
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--list-labels", action="store_true")
    args = parser.parse_args()
    
    config = load_config()
    generation_config = config["generation"]
    
    
    def resolve(cli_value, config_value):
        return cli_value if cli_value is not None else config_value
            
    num_images = resolve(args.num_images, generation_config["num_images"])
    batch_size = resolve(args.batch_size, generation_config["batch_size"])
    num_inference_steps = resolve(args.num_inference_steps, generation_config["num_inference_steps"])
    seed = resolve(args.seed, config["seed"])
    if seed == -1:
        seed = None
    device_setting = resolve(args.device, config["training"]["device"])
    output_dir = resolve(args.output_dir, config["paths"]["outputs_generated_samples"])
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint(config["paths"]["checkpoints"])
    image_size = config["data"]["image_size"]
    in_channels = config["model"]["in_channels"]
    device = resolve_device(device_setting)
    guidance_scale = resolve(args.guidance_scale, config["conditional"]["guidance_scale"])
    class_names = load_class_names(checkpoint_path)
    
    if args.list_labels:
        if class_names is None:
            print(f"This checkpoint is unconditional (no class_names.json): {checkpoint_path}")
        else:
            print(f"Available classes in checkpoint: {checkpoint_path}")
            for index, class_name in enumerate(class_names):
                print(f"    [{index}] {class_name}")
        return
    
    
    class_index = None
    null_class_index = None
    
    if args.label is not None:
        if class_names is None:
            raise ValueError("--label used, but this checkpoint is unconditional.")
        class_index = resolve_label_to_index(args.label, class_names)
        null_class_index = len(class_names)
    
    
    print("=== Image Generation ===")
    print(f"    Checkpoint:         {checkpoint_path}")
    print(f"    Device:             {device}")
    print(f"    Number of Images:   {num_images}")
    print(f"    Image Size:         {image_size}")
    print(f"    Denoising Step:     {num_inference_steps}")
    if seed is not None:
        print(f"    Seed:               {seed}")
    else:
        print("    Seed:               not defined")
        
    if class_index is not None:
        print(f"    Category:           {args.label} (index: {class_index})")
        print(f"    Guidance Scale:     {guidance_scale}")
    elif class_names is not None:
        print("    Category:           not set")
    print(f"    Output directory    {output_dir}")
    
    
    print("\n--- Loading Model ---")
    model = UNet2DModel.from_pretrained(str(checkpoint_path))
    model.to(device)
    
    noise_scheduler = build_scheduler_from_config(config)
    
    
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None
        
    
    print("\n--- Generating ---")
    all_images = []
    images_remaining = num_images
    total_start_time = time.time()
    
    while images_remaining > 0:
        current_batch_size = min(batch_size, images_remaining)
        already_done = num_images - images_remaining
        
        print(f"    Batch: {already_done + 1}-{already_done + current_batch_size} / {num_images}")
        
        batch_images = generate_batch(
             model=model,
            noise_scheduler=noise_scheduler,
            batch_size=current_batch_size,
            image_size=image_size,
            in_channels=in_channels,
            num_inference_steps=num_inference_steps,
            device=device,
            generator=generator,
            class_index=class_index,
            null_class_index=null_class_index,
            guidance_scale=guidance_scale
        )
        
        all_images.append(batch_images.cpu())
        images_remaining -= current_batch_size
        
    
    generated_images = torch.cat(all_images, dim=0)
    total_duration = time.time() - total_start_time
    

    print("\n --- Saving ---")
    saved_paths = save_images(generated_images, output_dir, start_index=0)
    print(f"    {len(saved_paths)} image(s) saved to: {output_dir}")
    
    grid_path = output_dir / "grid.png"
    save_grid(generated_images, grid_path)
    print(f"    Grid summary: {grid_path}")
    
    print(f"\n Completed in {total_duration:.1f}s")
    
    
    
if __name__ == "__main__":
    main()