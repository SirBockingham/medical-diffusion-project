import numpy as np
from scipy.ndimage import gaussian_filter
 
STANDARD_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
 
GAUSSIAN_SIGMA = 1.5
GAUSSIAN_TRUNCATE = 3.5
WINDOW_RADIUS = int(GAUSSIAN_TRUNCATE * GAUSSIAN_SIGMA + 0.5)
WINDOW_SIZE = 2 * WINDOW_RADIUS + 1
 
DATA_RANGE = 255.0
C1 = (0.01 * DATA_RANGE) ** 2
C2 = (0.03 * DATA_RANGE) ** 2
 
PAIR_BATCH_SIZE = 256
 
 
def max_scales_for_size(height: int, width: int) -> int:
    smaller_side = min(height, width)
    scales = 0
    while scales < len(STANDARD_WEIGHTS) and smaller_side / (2 ** scales) > WINDOW_SIZE - 1:
        scales += 1
    return scales
 
 
def scale_weights(num_scales: int) -> np.ndarray:
    weights = np.array(STANDARD_WEIGHTS[:num_scales], dtype=np.float64)
    return weights / weights.sum()
 
 
def _gaussian_valid(stack: np.ndarray) -> np.ndarray:
    filtered = gaussian_filter(
        stack, sigma=(0, GAUSSIAN_SIGMA, GAUSSIAN_SIGMA),
        truncate=GAUSSIAN_TRUNCATE, mode="reflect",
    )
    r = WINDOW_RADIUS
    return filtered[:, r:-r, r:-r]
 
 
def _ssim_and_contrast_structure(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu_x = _gaussian_valid(x)
    mu_y = _gaussian_valid(y)
 
    sigma_xx = _gaussian_valid(x * x) - mu_x * mu_x
    sigma_yy = _gaussian_valid(y * y) - mu_y * mu_y
    sigma_xy = _gaussian_valid(x * y) - mu_x * mu_y
 
    cs_map = (2.0 * sigma_xy + C2) / (sigma_xx + sigma_yy + C2)
    luminance_map = (2.0 * mu_x * mu_y + C1) / (mu_x * mu_x + mu_y * mu_y + C1)
    ssim_map = luminance_map * cs_map
 
    return ssim_map.mean(axis=(1, 2)), cs_map.mean(axis=(1, 2))
 
 
def _downsample(stack: np.ndarray) -> np.ndarray:
    height = stack.shape[1] - stack.shape[1] % 2
    width = stack.shape[2] - stack.shape[2] % 2
    stack = stack[:, :height, :width]
    return 0.25 * (stack[:, 0::2, 0::2] + stack[:, 1::2, 0::2]
                   + stack[:, 0::2, 1::2] + stack[:, 1::2, 1::2])
 
 
def ms_ssim_batch(x: np.ndarray, y: np.ndarray, num_scales: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(f"Expected two [B, H, W] arrays of equal shape, got {x.shape} and {y.shape}.")
 
    max_scales = max_scales_for_size(x.shape[1], x.shape[2])
    if num_scales is None:
        num_scales = max_scales
    if num_scales < 1 or num_scales > max_scales:
        raise ValueError(
            f"{x.shape[1]}x{x.shape[2]} images support 1..{max_scales} scales "
            f"with an {WINDOW_SIZE}x{WINDOW_SIZE} window, {num_scales} requested."
        )
 
    weights = scale_weights(num_scales)
    result = np.ones(x.shape[0], dtype=np.float64)
 
    for level in range(num_scales):
        ssim, cs = _ssim_and_contrast_structure(x, y)
 
        if level < num_scales - 1:
            result *= np.maximum(cs, 0.0) ** weights[level]
            x = _downsample(x)
            y = _downsample(y)
        else:
            result *= np.maximum(ssim, 0.0) ** weights[level]
 
    return result
 
 
def mean_pairwise_ms_ssim(
    images: np.ndarray,
    num_pairs: int,
    seed: int,
    num_scales: int | None = None,
) -> dict[str, float]:
    image_count = images.shape[0]
    if image_count < 2:
        raise ValueError(f"At least 2 images are needed for pairwise MS-SSIM, got {image_count}.")
 
    first, second = np.triu_indices(image_count, k=1)
    total_pairs = first.shape[0]
 
    if total_pairs > num_pairs:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(total_pairs, size=num_pairs, replace=False)
        first, second = first[chosen], second[chosen]
 
    values = []
    for start in range(0, first.shape[0], PAIR_BATCH_SIZE):
        end = start + PAIR_BATCH_SIZE
        values.append(ms_ssim_batch(images[first[start:end]], images[second[start:end]], num_scales))
    values = np.concatenate(values)
 
    used_scales = num_scales if num_scales is not None else max_scales_for_size(*images.shape[1:3])
    return {
        "mean": float(values.mean()),
        "sem": float(values.std(ddof=1) / np.sqrt(values.shape[0])) if values.shape[0] > 1 else 0.0,
        "pairs": int(values.shape[0]),
        "scales": int(used_scales),
    }
 
 
def compute_intra_class_ms_ssim(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    num_pairs: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    if real_images.shape[1:] != fake_images.shape[1:]:
        raise ValueError(
            f"Real and generated images differ in size: {real_images.shape[1:]} vs {fake_images.shape[1:]}."
        )
 
    generated = mean_pairwise_ms_ssim(fake_images, num_pairs, seed)
    real = mean_pairwise_ms_ssim(real_images, num_pairs, seed)
 
    return {
        "msssim_gen": generated["mean"],
        "msssim_real": real["mean"],
        "msssim_diff": generated["mean"] - real["mean"],
        "msssim_gen_sem": generated["sem"],
        "msssim_real_sem": real["sem"],
        "msssim_pairs": generated["pairs"],
        "msssim_scales": generated["scales"],
    }