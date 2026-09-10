import argparse
import json
import sys
import time
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from utils.config import load_config
from data.data_loading import build_dataloaders_from_config
from models.scheduler import build_scheduler_from_config

from generation.generate import denormalize, find_latest_checkpoint, generate_batch, load_class_names

from diffusers import UNet2DModel, DDPMScheduler # type: ignore

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    print("Error: torchmetrics package not installed.")
    print("Install with:  pip install torchmetrics[image]")
    sys.exit(1)
    

def resolve_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    return torch.device(device_setting)



def prepare_for_fid(images_uint8: torch.Tensor) -> torch.Tensor:
    if images_uint8.shape[1] == 1:
        images_uint8 = images_uint8.repeat(1, 3, 1, 1)
        
    return images_uint8



@torch.no_grad()
def collect_real_images(
        dataloader,
        num_samples: int,
        device: torch.device,
        fid_metric: FrechetInceptionDistance,
        class_index_filter: int | None = None
) -> int:
    collected_count = 0
    
    for batch in dataloader:
        if collected_count >= num_samples:
            break
        
        images = batch["image"]
        
        if class_index_filter is not None:
            keep_mask = batch["class_index"] == class_index_filter
            if not bool(keep_mask.any()):
                continue
            images = images[keep_mask]
        
        remaining = num_samples - collected_count
        if images.shape[0] > remaining:
            images = images[:remaining]
        
        images_uint8 = denormalize(images)
        images_uint8 = prepare_for_fid(images_uint8).to(device)
        
        fid_metric.update(images_uint8, real=True)
        collected_count += images.shape[0]
        
    return collected_count



@torch.no_grad()
def collect_generated_images(
        model: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        num_samples: int,
        batch_size: int,
        image_size: int,
        in_channels: int,
        num_inference_steps: int,
        device: torch.device,
        fid_metric: FrechetInceptionDistance,
        class_index: int | None = None,
        null_class_index: int | None = None,
        guidance_scale: float = 1.0
) -> int:
    generated_count = 0
    start_time = time.time()
    
    while generated_count < num_samples:
        current_batch_size = min(batch_size, num_samples - generated_count)
        
        images = generate_batch(
            model=model,
            noise_scheduler=noise_scheduler,
            batch_size=current_batch_size,
            image_size=image_size,
            in_channels=in_channels,
            num_inference_steps=num_inference_steps,
            device=device,
            generator=None,
            class_index=class_index,
            null_class_index=null_class_index,
            guidance_scale=guidance_scale
        )
        
        images_uint8 = denormalize(images.cpu())
        images_uint8 = prepare_for_fid(images_uint8).to(device)
        
        fid_metric.update(images_uint8, real=False)
        generated_count += current_batch_size
        
        elapsed = time.time() - start_time
        percent = 100 * generated_count / num_samples
        print(f"    generating: {generated_count}/{num_samples} images ({percent:.0f}%, {elapsed:.0f}s)")
        
        
    return generated_count



def compute_fid(
        model: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        dataloader,
        num_samples: int,
        batch_size: int,
        image_size: int,
        in_channels: int,
        num_inference_steps: int,
        device: torch.device,
        class_index: int | None = None,
        null_class_index: int | None = None,
        guidance_scale: float = 1.0,
        real_class_filter: int | None = None,
) -> float | None:
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    
    print("  Loading real images...")
    real_count = collect_real_images(
        dataloader=dataloader,
        num_samples=num_samples,
        device=device,
        fid_metric=fid_metric,
        class_index_filter=real_class_filter
    )
    print(f"    {real_count} real images processed")
    
    minimum_required = 50
    if real_count < minimum_required:
        print(f"Not enough real images ({real_count} < {minimum_required}), FID can not be calculated realiably.")
        return None

    print("  Generating images...")
    collect_generated_images(
        model=model,
        noise_scheduler=noise_scheduler,
        num_samples=real_count,
        batch_size=batch_size,
        image_size=image_size,
        in_channels=in_channels,
        num_inference_steps=num_inference_steps,
        device=device,
        fid_metric=fid_metric,
        class_index=class_index,
        null_class_index=null_class_index,
        guidance_scale=guidance_scale
    )
    
    fid_value = float(fid_metric.compute().item())
    
    fid_metric.reset()
    del fid_metric
    
    return fid_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--no-per-class", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    
    config = load_config()
    evaluation_config = config["evaluation"]
    
    def resolve(cli_value, config_value):
        return cli_value if cli_value is not None else config_value
    
    num_samples = resolve(args.num_samples, evaluation_config["num_samples"])
    batch_size = resolve(args.batch_size, evaluation_config["batch_size"])
    num_inference_steps = resolve(args.num_inference_steps, evaluation_config["num_inference_steps"])
    guidance_scale = resolve(args.guidance_scale, config["conditional"]["guidance_scale"])
    device_setting = resolve(args.device, config["training"]["device"])
    
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(config["paths"]["checkpoints"])
        
    per_class = evaluation_config["per_class"]
    per_class_num_samples = evaluation_config["per_class_num_samples"]
    
    image_size = config["data"]["image_size"]
    in_channels = config["model"]["in_channels"]
    device = resolve_device(device_setting)
    
    class_names = load_class_names(checkpoint_path)
    is_conditional = class_names is not None
    
    
    print("=== FID evaluation ===")
    print(f"    Checkpoint:         {checkpoint_path}")
    print(f"    Device:             {device}")
    print(f"    Real images:        '{args.split}' split")
    print(f"    No. of samples:     {num_samples}")
    print(f"    Denoising steps.    {num_inference_steps}")
    if is_conditional:
        print(f"    Model:              conditional ({len(class_names)} categories)")
        print(f"    Guidance scale:     {guidance_scale}")
    else:
        print("    Model:              unconditional")
        
    
    # Loading model, scheduler, building data loader
    print("\n--- Loading ---")
    model = UNet2DModel.from_pretrained(str(checkpoint_path))
    model.to(device)
    model.eval()
    
    noise_scheduler = build_scheduler_from_config(config)
    dataloaders = build_dataloaders_from_config(config)
    dataloader = dataloaders[args.split]
    
    null_class_index = None
    if is_conditional:
        null_class_index = len(class_names)
    
    results: dict[str, float | None] = {}
    
    
    
    # Overall FID
    print("\n--- Overall FID ---")
    overall_start = time.time()
    
    if is_conditional:
        overall_class_index = null_class_index
    else:
        overall_class_index = None
    
    overall_fid = compute_fid(
        model=model,
        noise_scheduler=noise_scheduler,
        dataloader=dataloader,
        num_samples=num_samples,
        batch_size=batch_size,
        image_size=image_size,
        in_channels=in_channels,
        num_inference_steps=num_inference_steps,
        device=device,
        class_index=overall_class_index,
        null_class_index=None,
        guidance_scale=1.0
    )
    results["overall"] = overall_fid
    
    if overall_fid is not None:
        print(f"\n  FID (overall):   {overall_fid:.2f}")
    print(f"    ({time.time() - overall_start:.0f}s)")
    
    
    # Categorized FID
    if per_class and is_conditional:
        print("\n--- Categorized FID ---")
        print("comparing every category with real images of the same label")
        
        for class_index, class_name in enumerate(class_names):
            print(f"\n  [{class_index}] {class_name}")
            class_start = time.time()
            
            class_fid = compute_fid(
                model=model,
                noise_scheduler=noise_scheduler,
                dataloader=dataloader,
                num_samples=per_class_num_samples,
                batch_size=batch_size,
                image_size=image_size,
                in_channels=in_channels,
                num_inference_steps=num_inference_steps,
                device=device,
                class_index=class_index,
                null_class_index=null_class_index,
                guidance_scale=guidance_scale,
                real_class_filter=class_index
            )
            results[class_name] = class_fid
            
            if class_fid is not None:
                print(f"     FID: {class_fid:.2f}   ({time.time() - class_start:.0f}s)")
                
    
    print(f"\nEvaluation completed in {time.time() - overall_start:.0f}s")
    
    
    # Summary
    print("\n=== Results ===")
    for name, value in results.items():
        if value is None:
            print(f"    {name:25s} - (unable to calculate, not enough images)")
        else:
            print(f"    {name:25s}  {value:8.2f}")
            
    
    # Saving
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "num_samples": num_samples,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "results": results
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_path, f, ensure_ascii=False, indent=2)
            
        print(f"\nResults saved to: {output_path}")
        
        

if __name__ == "__main__":
    main()
