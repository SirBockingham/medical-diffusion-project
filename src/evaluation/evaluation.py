import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "evaluation"))

from density_coverage import compute_density_coverage
from fid import compute_fid
from ms_ssim import compute_intra_class_ms_ssim
from utils.device import resolve_device

from utils.config import load_config, pick

MIN_REAL_SAMPLES = 50



def file_stem(name:str) -> str:
    """Filename-safe version of a class name ("No Finding" -> "no_finding")."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")



@dataclass
class Scope:
    name: str
    num_samples: int
    class_index: int | None
    real_class_filter: int | None
    null_class_index: int | None
    guidance_scale: float
    
    @property
    def file_stem(self) -> str:
        return file_stem(self.name)
    
    
    
@dataclass
class EvaluationData:
    scope_name: str
    real_features: np.ndarray
    fake_features: np.ndarray
    real_images: np.ndarray | None = None
    fake_images: np.ndarray | None = None
    
    
    
def run_fid(data: EvaluationData, settings: dict) -> dict[str, float]:
    return {"fid": compute_fid(data.real_features, data.fake_features)}



def run_density_coverage(data: EvaluationData, settings: dict) -> dict[str, float]:
    result = compute_density_coverage(
        data.real_features, data.fake_features, k=settings["nearest_k"]
    )
    return {
        "density": result["density"],
        "coverage": result["coverage"],
        "precision": result["precision"],
        "recall": result["recall"],
    }
    
    
    
def run_ms_ssim(data: EvaluationData, settings: dict) -> dict[str, float]:
    if data.real_images is None or data.fake_images is None:
        raise ValueError(
            "no images available for this scope (the saved features were stored without images,"
            "run the evaluation with generation"
        )
    return compute_intra_class_ms_ssim(
        data.real_images, data.fake_images,
        num_pairs=settings["msssim_pairs"], seed=settings["seed"],
    )
    
    
    
@dataclass
class MetricSpec:
    function: Callable[[EvaluationData, dict], dict[str, float]]
    columns: list[str]
    needs_images: bool = False
    


METRICS: dict[str, MetricSpec] = {
    "fid": MetricSpec(
        function=run_fid,
        columns=["fid"]
    ),
    "density_coverage": MetricSpec(
        function=run_density_coverage,
        columns=["density", "coverage", "precision", "recall"]
    ),
    "ms_ssim": MetricSpec(
        function=run_ms_ssim,
        columns=["msssim_gen", "msssim_real", "msssim_diff"],
        needs_images=True,
    ),
}



def run_metrics(
        data: EvaluationData,
        metric_names: list[str],
        settings: dict,
) -> dict:
    result: dict = {
        "num_real": int(data.real_features.shape[0]),
        "num_fake": int(data.fake_features.shape[0]),
    }
    errors: dict[str, str] = {}
 
    for name in metric_names:
        try:
            result.update(METRICS[name].function(data, settings))
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
            print(f"    Warning: metric '{name}' failed: {errors[name]}")
 
    if errors:
        result["errors"] = errors
 
    return result




class ModelFeatureSource:
    def __init__(self, config: dict, checkpoint_path: Path, split: str,
                 device_setting: str, batch_size: int, num_inference_steps: int,
                 features_dir: Path | None, keep_images: bool = False):
        from diffusers import UNet2DModel  # type: ignore
        from features import InceptionFeatureExtractor

        from data.data_loading import build_dataloaders_from_config
        from models.scheduler import build_scheduler_from_config
 
        self.device = resolve_device(device_setting)
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.image_size = config["data"]["image_size"]
        self.in_channels = config["model"]["in_channels"]
        self.features_dir = features_dir
        self.keep_images = keep_images
 
        print("\n--- Loading ---")
        self.model = UNet2DModel.from_pretrained(str(checkpoint_path))
        self.model.to(self.device) # type: ignore
        self.model.eval()
 
        self.noise_scheduler = build_scheduler_from_config(config)
        self.dataloader = build_dataloaders_from_config(config)[split]
        self.extractor = InceptionFeatureExtractor(self.device)
 
    def get(self, scope: Scope) -> EvaluationData | None:
        from features import collect_real_samples, generate_fake_samples
 
        print("  Loading real images...")
        real, real_images = collect_real_samples(
            self.dataloader, scope.num_samples, self.extractor,
            class_index_filter=scope.real_class_filter,
            keep_images=self.keep_images
        )
        print(f"    {real.shape[0]} real images processed")
 
        if real.shape[0] < MIN_REAL_SAMPLES:
            print(f"    Not enough real images ({real.shape[0]} < {MIN_REAL_SAMPLES}), "
                  "this category can not be evaluated reliably.")
            return None
 
        print("  Generating images...")
        fake, fake_images = generate_fake_samples(
            model=self.model,
            noise_scheduler=self.noise_scheduler,
            num_samples=real.shape[0],
            batch_size=self.batch_size,
            image_size=self.image_size,
            in_channels=self.in_channels,
            num_inference_steps=self.num_inference_steps,
            device=self.device,
            extractor=self.extractor,
            class_index=scope.class_index,
            null_class_index=scope.null_class_index,
            guidance_scale=scope.guidance_scale,
            keep_images=self.keep_images
        )
 
        if self.features_dir is not None:
            arrays = {"real": real, "fake": fake}
            if real_images is not None and fake_images is not None:
                arrays["real_images"] = real_images
                arrays["fake_images"] = fake_images
            np.savez_compressed(self.features_dir / f"{scope.file_stem}.npz", **arrays)
 
        return EvaluationData(scope.name, real, fake, real_images, fake_images)
 
 
class SavedFeatureSource:
    def __init__(self, features_dir: Path):
        self.features_dir = features_dir
 
    def get(self, scope: Scope) -> EvaluationData | None:
        path = self.features_dir / f"{scope.file_stem}.npz"
        if not path.exists():
            print(f"    No saved features for this scope ({path.name}): too few real images, or the run was interrupted.")
            return None
 
        stored = np.load(path)
        has_images = "real_images" in stored.files and "fake_images" in stored.files
        print(f"    loaded: {stored['real'].shape[0]} real, {stored['fake'].shape[0]} generated"
              f"{' (with images)' if has_images else ''}")
        return EvaluationData(
            scope.name, stored["real"], stored["fake"],
            stored["real_images"] if has_images else None,
            stored["fake_images"] if has_images else None,
        )
 
 
def build_scopes(class_names: list[str] | None, per_class: bool,
                 num_samples: int, per_class_num_samples: int,
                 guidance_scale: float) -> list[Scope]:
    is_conditional = class_names is not None
    null_class_index = len(class_names) if is_conditional else None
 
    scopes = [Scope(
        name="overall",
        num_samples=num_samples,
        class_index=null_class_index,
        real_class_filter=None,
        null_class_index=None,
        guidance_scale=1.0,
    )]
 
    if per_class and is_conditional:
        for class_index, class_name in enumerate(class_names):
            scopes.append(Scope(
                name=class_name,
                num_samples=per_class_num_samples,
                class_index=class_index,
                real_class_filter=class_index,
                null_class_index=null_class_index,
                guidance_scale=guidance_scale,
            ))
 
    return scopes
 
 
def format_value(value, column: str) -> str:
    if value is None:
        return "-"
    if column == "fid":
        return f"{value:.2f}"
    if column == "msssim_diff":
        return f"{value:+.3f}"
    return f"{value:.3f}"
 
 
def print_results_table(results: dict[str, dict | None], columns: list[str]) -> None:
    widths = {column: max(9, len(column) + 1) for column in columns}
    header = f"  {'category':<22}" + "".join(f"{c:>{widths[c]}}" for c in columns) + f"{'sample':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
 
    for name, result in results.items():
        if result is None:
            print(f"  {name:<22}  (unable to evaluate - too few real images)")
            continue
        cells = "".join(f"{format_value(result.get(c), c):>{widths[c]}}" for c in columns)
        print(f"  {name:<22}{cells}{result['num_real']:>7d}")
        if "errors" in result:
            for metric, message in result["errors"].items():
                print(f"  {'':<22}  ! {metric}: {message}")
 
 
def save_report(path: Path, metadata: dict, results: dict, complete: bool) -> None:
    report = dict(metadata)
    report["complete"] = complete
    report["results"] = results
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
 
 
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--nearest-k", type=int, default=None)
    parser.add_argument("--msssim-pairs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--no-per-class", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--from-features", default=None)
    return parser.parse_args()

 
 
def main():
    args = parse_args()
    config = load_config()
    evaluation_config = config["evaluation"]
 
    if args.metrics is not None:
        metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    else:
        metric_names = list(evaluation_config["metrics"])
 
    unknown = [name for name in metric_names if name not in METRICS]
    if unknown:
        print(f"Error: unknown metric(s): {unknown}. Available: {list(METRICS)}")
        sys.exit(1)
 
    nearest_k = pick(args.nearest_k, evaluation_config["nearest_k"])
    msssim_pairs = pick(args.msssim_pairs, evaluation_config["msssim_pairs"])
    settings = {
        "nearest_k": nearest_k,
        "msssim_pairs": msssim_pairs,
        "seed": config["seed"]
    }
 
    keep_images = any(METRICS[name].needs_images for name in metric_names)
 
    timestamp = time.strftime("%Y%m%d_%H%M")
 
    if args.from_features is not None:
        features_dir = Path(args.from_features)
        meta_path = features_dir / "features_meta.json"
        if not meta_path.exists():
            print(f"Error: not found: {meta_path}")
            sys.exit(1)
 
        stored_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        scopes = [Scope(**scope) for scope in stored_meta["scopes"]]
        make_source = lambda: SavedFeatureSource(features_dir)
 
        metadata = dict(stored_meta["collection"])
        metadata["source"] = f"saved_features: {features_dir}"
        default_output = features_dir.parent / f"{features_dir.name.replace('_features', '')}_recomputed_{timestamp}.json"
 
    else:
        from generation.generate import (
            check_checkpoint_compatibility,
            find_latest_checkpoint,
            load_class_names,
        )
 
        if args.checkpoint is not None:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = find_latest_checkpoint(config["paths"]["checkpoints"])
 
        if not check_checkpoint_compatibility(checkpoint_path, config):
            return
 
        class_names = load_class_names(checkpoint_path)
        num_samples = pick(args.num_samples, evaluation_config["num_samples"])
        per_class_num_samples = evaluation_config["per_class_num_samples"]
        num_inference_steps = pick(args.num_inference_steps, evaluation_config["num_inference_steps"])
        guidance_scale = pick(args.guidance_scale, config["conditional"]["guidance_scale"])
 
        scopes = build_scopes(
            class_names,
            per_class=evaluation_config["per_class"] and not args.no_per_class,
            num_samples=num_samples,
            per_class_num_samples=per_class_num_samples,
            guidance_scale=guidance_scale,
        )
 
        run_name = f"{checkpoint_path.parent.name}_{checkpoint_path.name}"
        default_output = config["paths"]["outputs_evaluations"] / f"eval_{run_name}_{timestamp}.json"
 
        metadata = {
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "num_samples": num_samples,
            "per_class_num_samples": per_class_num_samples,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "feature_extractor": "inception-v3-compat, 2048 dim",
            "conditional": class_names is not None,
        }
 
        features_dir = None
        if args.save_features:
            output_for_name = Path(args.output) if args.output else default_output
            features_dir = output_for_name.parent / f"{output_for_name.stem}_features"
            features_dir.mkdir(parents=True, exist_ok=True)
            (features_dir / "features_meta.json").write_text(json.dumps({
                "collection": metadata,
                "scopes": [asdict(scope) for scope in scopes],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
 
        make_source = lambda: ModelFeatureSource(
            config, checkpoint_path, args.split,
            device_setting=pick(args.device, config["training"]["device"]),
            batch_size=pick(args.batch_size, evaluation_config["batch_size"]),
            num_inference_steps=num_inference_steps,
            features_dir=features_dir,
            keep_images=keep_images
        )
 
    output_path = Path(args.output) if args.output else default_output
 
    metadata["metrics"] = metric_names
    metadata["nearest_k"] = nearest_k
    if "ms_ssim" in metric_names:
        metadata["msssim_pairs"] = msssim_pairs
    metadata["evaluated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
 
    print("\nEvaluation")
    for key in ["checkpoint", "source", "split", "num_samples", "num_inference_steps", "guidance_scale"]:
        if key in metadata:
            print(f"  {key:<20} {metadata[key]}")
    print(f"  {'metrics':<20} {', '.join(metric_names)}")
    print(f"  {'scopes':<20} {len(scopes)}")
    print(f"  {'output':<20} {output_path}")
 
    source = make_source()
 
    results: dict[str, dict | None] = {}
 
    for number, scope in enumerate(scopes, start=1):
        print(f"\n--- [{number}/{len(scopes)}] {scope.name} ---")
        scope_start = time.time()
 
        data = source.get(scope)
        if data is None:
            results[scope.name] = None
        else:
            results[scope.name] = run_metrics(data, metric_names, settings)
            summary = "  ".join(
                f"{column}: {format_value(results[scope.name].get(column), column)}"
                for name in metric_names for column in METRICS[name].columns
            )
            print(f"  {summary}  ({time.time() - scope_start:.0f}s)")
 
        save_report(output_path, metadata, results, complete=False)
 
    save_report(output_path, metadata, results, complete=True)
 
    columns = [column for name in metric_names for column in METRICS[name].columns]
 
    print("\nResults")
    print_results_table(results, columns)
 
    print(f"\nResults saved: {output_path}")
    if args.save_features and args.from_features is None:
        print(f"Features saved: {features_dir}")
 
 
if __name__ == "__main__":
    main()