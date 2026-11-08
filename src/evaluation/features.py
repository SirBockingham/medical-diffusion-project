import sys
import time
from pathlib import Path
 
import numpy as np
import torch
 
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from torchmetrics.image.fid import FrechetInceptionDistance

from generation.generate import denormalize, generate_batch

# PIL's own fixed-point luma weights, so the result matches Image.convert("L") exactly
LUMA_WEIGHTS_FIXED = (19595, 38470, 7471)


def to_uint8_rgb(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float images -> uint8 [B, 3, H, W] on the CPU."""
    images_uint8 = denormalize(images.cpu())
    if images_uint8.shape[1] == 1:
        images_uint8 = images_uint8.repeat(1, 3, 1, 1)
    return images_uint8
 
 
def to_grayscale(images_uint8: torch.Tensor) -> np.ndarray:
    rgb = images_uint8.numpy().astype(np.int64)
    gray = (LUMA_WEIGHTS_FIXED[0] * rgb[:, 0] + LUMA_WEIGHTS_FIXED[1] * rgb[:, 1]
            + LUMA_WEIGHTS_FIXED[2] * rgb[:, 2] + 0x8000) >> 16
    return gray.astype(np.uint8)



class InceptionFeatureExtractor:
    FEATURE_DIM = 2048
    
    def __init__(self, device: torch.device):
        self.device = device
        fid_metric = FrechetInceptionDistance(feature=self.FEATURE_DIM, normalize=False)
 
        inception = getattr(fid_metric, "inception", None)
        if inception is None:
            raise RuntimeError(
                "This torchmetrics version has no 'inception' attribute on FrechetInceptionDistance, so the features can not be extracted."
            )
 
        self.inception = inception.to(device)
        self.inception.eval()
        
        
    @torch.no_grad()
    def extract(self, images_uint8: torch.Tensor) -> np.ndarray:
        """Expects the output of to_uint8_rgb: uint8 [B, 3, H, W]."""
 
        features = self.inception(images_uint8.to(self.device))
        return features.reshape(features.shape[0], -1).float().cpu().numpy()
    
    

def _stack(parts: list[np.ndarray], empty_shape: tuple[int, ...], dtype) -> np.ndarray:
    if not parts:
        return np.empty(empty_shape, dtype=dtype)
    return np.concatenate(parts, axis=0)

    

@torch.no_grad()
def collect_real_samples(
    dataloader,
    num_samples: int,
    extractor: InceptionFeatureExtractor,
    class_index_filter: int | None = None,
    keep_images: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    feature_parts: list[np.ndarray] = []
    image_parts: list[np.ndarray] = []
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
 
        images_uint8 = to_uint8_rgb(images)
        feature_parts.append(extractor.extract(images_uint8))
        if keep_images:
            image_parts.append(to_grayscale(images_uint8))
        collected_count += images.shape[0]
 
    features = _stack(feature_parts, (0, InceptionFeatureExtractor.FEATURE_DIM), np.float32)
    kept_images = _stack(image_parts, (0, 0, 0), np.uint8) if keep_images else None
    return features, kept_images


@torch.no_grad()
def generate_fake_samples(
    model,
    noise_scheduler,
    num_samples: int,
    batch_size: int,
    image_size: int,
    in_channels: int,
    num_inference_steps: int,
    device: torch.device,
    extractor: InceptionFeatureExtractor,
    class_index: int | None = None,
    null_class_index: int | None = None,
    guidance_scale: float = 1.0,
    keep_images: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    feature_parts: list[np.ndarray] = []
    image_parts: list[np.ndarray] = []
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
            guidance_scale=guidance_scale,
        )
 
        images_uint8 = to_uint8_rgb(images)
        feature_parts.append(extractor.extract(images_uint8))
        if keep_images:
            image_parts.append(to_grayscale(images_uint8))
        generated_count += current_batch_size
 
        elapsed = time.time() - start_time
        percent = 100 * generated_count / num_samples
        print(f"    generating: {generated_count}/{num_samples} images "
              f"({percent:.0f}%, {elapsed:.0f}s)")
 
    features = np.concatenate(feature_parts, axis=0)
    kept_images = np.concatenate(image_parts, axis=0) if keep_images else None
    return features, kept_images