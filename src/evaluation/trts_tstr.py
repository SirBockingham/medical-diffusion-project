import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "evaluation"))

from classification_metrics import (
    compute_classification_metrics,
    mean_and_std,
    mean_per_class,
)
from classifier import predict, train_classifier
from features import to_grayscale, to_uint8_rgb

from data.data_loading import build_datasets_from_config, collate_batch
from evaluation import file_stem
from generation.generate import (
    check_checkpoint_compatibility,
    find_latest_checkpoint,
    generate_batch,
    load_class_names,
    resolve_label_to_index,
)
from utils.config import load_config, pick
from utils.device import resolve_device

# which classifier (trained on real or synthetic images) is tested on which images
EXPERIMENTS = {
    "TRTR": {"train": "real", "test": "real"},
    "TSTR": {"train": "synthetic", "test": "real"},
    "TRTS": {"train": "real", "test": "synthetic"},
}

# classes with fewer images are skipped, a handful of images only adds noise to the averages
MIN_TRAIN_PER_CLASS = 20
MIN_TEST_PER_CLASS = 10

LOADING_BATCH_SIZE = 64

# large prime offset, so the per-class generation seeds never overlap
CLASS_SEED_STRIDE = 100_003



def select_indices_per_class(
    dataset,
    class_indices: list[int],
    max_per_class: int,
    rng: np.random.RandomState
) -> dict[int, list[int]]:
    """At most max_per_class random sample indices per class.
    Random because the CSV is sorted by patient, the first N rows would come from few patients."""
    indices_by_class: dict[int, list[int]] = {}
    for class_index in class_indices:
        indices_by_class[class_index] = []
 
    for sample_index, sample in enumerate(dataset.samples):
        if sample["class_index"] in indices_by_class:
            indices_by_class[sample["class_index"]].append(sample_index)
 
    selected: dict[int, list[int]] = {}
    for class_index in class_indices:
        candidates = indices_by_class[class_index]
        shuffled = [candidates[position] for position in rng.permutation(len(candidates))]
        selected[class_index] = sorted(shuffled[:max_per_class])

    return selected


def load_real_images(dataset, indices: list[int], num_workers: int) -> np.ndarray:
    """The given images as a uint8 [N, H, W] grayscale array, same format as the generated ones."""
    loader = DataLoader(Subset(dataset, indices), batch_size=LOADING_BATCH_SIZE,
                        shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    parts = []
    for batch in loader:
        parts.append(to_grayscale(to_uint8_rgb(batch["image"])))
    return np.concatenate(parts, axis=0)



class SyntheticImageGenerator:
    """Loads the diffusion model loosely, so a fully cached run never touches it."""
 
    def __init__(
            self,
            checkpoint_path: Path,
            config: dict,
            device: torch.device,
            batch_size: int,
            num_inference_steps: int,
            guidance_scale: float,
            null_class_index: int
        ):
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.null_class_index = null_class_index
        self.model = None
        self.noise_scheduler = None

    def _load(self) -> None:
        from diffusers import UNet2DModel # type: ignore

        from models.scheduler import build_scheduler_from_config
 
        print("  Loading the diffusion model...")
        self.model = UNet2DModel.from_pretrained(str(self.checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        self.noise_scheduler = build_scheduler_from_config(self.config)

    def generate(
        self,
        class_index: int,
        count: int,
        seed: int
    ) -> np.ndarray:
        """count images of one class as uint8 [count, H, W]."""
        if self.model is None:
            self._load()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        parts = []
        generated_count = 0
        start_time = time.time()

        while generated_count < count:
            current_batch_size = min(self.batch_size, count - generated_count)
            images = generate_batch(
                model=self.model,
                noise_scheduler=self.noise_scheduler,
                batch_size=current_batch_size,
                image_size=self.config["data"]["image_size"],
                in_channels=self.config["model"]["in_channels"],
                num_inference_steps=self.num_inference_steps,
                device=self.device,
                generator=generator,
                class_index=class_index,
                null_class_index=self.null_class_index,
                guidance_scale=self.guidance_scale,
                # silences the per-step progress print, one line per batch is enough here
                progress_callback=lambda step, total: None,
            )
            parts.append(to_grayscale(to_uint8_rgb(images)))
            generated_count += current_batch_size

            percent = 100 * generated_count / count
            print(f"      generating: {generated_count}/{count} images "
                  f"({percent:.0f}%, {time.time() - start_time:.0f}s)")

        return np.concatenate(parts, axis=0)


def check_cache_meta(cache_dir: Path, expected: dict) -> None:
    """Stops if the cache folder was made with different settings, instead of mixing images."""
    meta_path = cache_dir / "cache_meta.json"
    if not meta_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    stored = json.loads(meta_path.read_text(encoding="utf-8"))
    differences = []
    for key, value in expected.items():
        if stored.get(key) != value:
            differences.append(f"{key}: cache {stored.get(key)!r}, now {value!r}")

    if differences:
        print(f"Error: the synthetic image cache was created with different settings:\n"
              f"  {cache_dir}")
        for difference in differences:
            print(f"  - {difference}")
        print("Use another --cache-dir, or --regenerate to overwrite it.")
        sys.exit(1)


def get_synthetic_images(
    generator: SyntheticImageGenerator,
    cache_dir: Path,
    class_index: int,
    class_name: str,
    count: int,
    seed: int,
    regenerate: bool
) -> np.ndarray:
    """count generated images of one class, from the cache, generating (and caching) what is missing."""
    cache_path = cache_dir / f"{file_stem(class_name)}.npz"

    cached = None
    if cache_path.exists() and not regenerate:
        cached = np.load(cache_path)["images"]

    if cached is not None and cached.shape[0] >= count:
        print(f"  {class_name}: {count} images from cache")
        return cached[:count]

    already = 0 if cached is None else cached.shape[0]
    missing = count - already
    print(f"  {class_name}: generating {missing} images"
          f"{f' ({already} already cached)' if already else ''}...")

    # the seed also depends on the cached count, so topped-up images do not repeat earlier ones
    new_images = generator.generate(class_index, missing, seed + CLASS_SEED_STRIDE * class_index + already)

    if cached is None:
        all_images = new_images
    else:
        all_images = np.concatenate([cached, new_images], axis=0)

    np.savez_compressed(cache_path, images=all_images)
    return all_images[:count]



def run_repeat(
    data: dict,
    num_classes: int,
    training_settings: dict,
    device: torch.device,
    seed: int
) -> dict:
    """One repeat: two classifiers trained, three evaluations."""
    results = {}

    print(f"  Training on real images (n={data['real_train_labels'].shape[0]})...")
    real_model = train_classifier(
        data["real_train_images"], data["real_train_labels"], num_classes,
        device=device, seed=seed, log_prefix="    ", **training_settings,
    )
    batch_size = training_settings["batch_size"]
    results["TRTR"] = compute_classification_metrics(
        data["real_test_labels"],
        predict(real_model, data["real_test_images"], batch_size, device),
        num_classes,
    )
    results["TRTS"] = compute_classification_metrics(
        data["synthetic_labels"],
        predict(real_model, data["synthetic_images"], batch_size, device),
        num_classes,
    )
    del real_model

    print(f"  Training on synthetic images (n={data['synthetic_labels'].shape[0]})...")
    synthetic_model = train_classifier(
        data["synthetic_images"], data["synthetic_labels"], num_classes,
        device=device, seed=seed, log_prefix="    ", **training_settings,
    )
    results["TSTR"] = compute_classification_metrics(
        data["real_test_labels"],
        predict(synthetic_model, data["real_test_images"], batch_size, device),
        num_classes,
    )
    del synthetic_model

    return {name: results[name] for name in EXPERIMENTS}


def summarize(runs: list[dict], class_names: list[str]) -> dict:
    """Mean and std over the repeats, per experiment."""
    summary = {}
    for name in EXPERIMENTS:
        experiment_runs = [run["results"][name] for run in runs]
        averaged_recall = mean_per_class([result["per_class_recall"] for result in experiment_runs])
        summary[name] = {
            "accuracy": mean_and_std([result["accuracy"] for result in experiment_runs]),
            "balanced_accuracy": mean_and_std([result["balanced_accuracy"] for result in experiment_runs]),
            "macro_f1": mean_and_std([result["macro_f1"] for result in experiment_runs]),
            "per_class_recall": dict(zip(class_names, averaged_recall)),
        }
    return summary


def format_stat(stat: dict[str, float], with_std: bool) -> str:
    if with_std:
        return f"{stat['mean']:.3f}+-{stat['std']:.3f}"
    return f"{stat['mean']:.3f}"


def format_recall(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_summary(summary: dict, class_names: list[str], counts: dict, repeats: int) -> None:
    with_std = repeats > 1
    width = 13 if with_std else 9

    print("\nResults")
    if with_std:
        print(f"  (mean+-std over {repeats} repeats)")
    header = (f"  {'experiment':<11}{'train':<11}{'test':<11}"
              f"{'accuracy':>{width}}{'bal_acc':>{width}}{'macro_f1':>{width}}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, setup in EXPERIMENTS.items():
        row = summary[name]
        print(f"  {name:<11}{setup['train']:<11}{setup['test']:<11}"
              f"{format_stat(row['accuracy'], with_std):>{width}}"
              f"{format_stat(row['balanced_accuracy'], with_std):>{width}}"
              f"{format_stat(row['macro_f1'], with_std):>{width}}")

    chance = 1.0 / len(class_names)
    trtr = summary["TRTR"]["balanced_accuracy"]["mean"]
    tstr = summary["TSTR"]["balanced_accuracy"]["mean"]
    trts = summary["TRTS"]["balanced_accuracy"]["mean"]
    print(f"\n  chance level (1/{len(class_names)} classes): {chance:.3f}")
    if trtr > chance:
        # the above-chance ratio keeps a TSTR of 0.15 from looking good when chance is 0.1
        print(f"  TSTR / TRTR (balanced accuracy):              {tstr / trtr:.2f}")
        print(f"  TSTR / TRTR above chance ((TSTR-c)/(TRTR-c)):  "
              f"{(tstr - chance) / (trtr - chance):.2f}")
        print(f"  TRTS / TRTR (balanced accuracy):              {trts / trtr:.2f}")
    else:
        print("  Warning: TRTR is not above chance level, the classifier did not learn")
        print("  the real classes, so TSTR and TRTS can not be interpreted.")

    print("\nPer-class recall (TRTS = label fidelity of the generated images):")
    print(f"  {'class':<22}{'TRTR':>8}{'TSTR':>8}{'TRTS':>8}{'n_train':>9}{'n_test':>8}")
    print("  " + "-" * 61)
    for class_name in class_names:
        print(f"  {class_name:<22}"
              f"{format_recall(summary['TRTR']['per_class_recall'][class_name]):>8}"
              f"{format_recall(summary['TSTR']['per_class_recall'][class_name]):>8}"
              f"{format_recall(summary['TRTS']['per_class_recall'][class_name]):>8}"
              f"{counts[class_name]['real_train']:>9d}{counts[class_name]['real_test']:>8d}")



def save_report(path: Path, metadata: dict, runs: list[dict], summary: dict | None,
                complete: bool) -> None:
    report = dict(metadata)
    report["complete"] = complete
    report["runs"] = runs
    report["summary"] = summary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")



def parse_args():
    parser = argparse.ArgumentParser(
        description="TRTR / TSTR / TRTS evaluation of a conditional checkpoint with a classifier")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to evaluate (default: the latest one)")
    parser.add_argument("--classes", default=None,
                        help="Comma-separated class names (default: config trts_tstr.classes, "
                             "empty = all classes of the checkpoint)")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Max. real training images per class; the same number of images "
                             "is generated (config: trts_tstr.samples_per_class)")
    parser.add_argument("--test-samples-per-class", type=int, default=None,
                        help="Max. real test images per class (config: trts_tstr.test_samples_per_class)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Classifier batch size")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=None,
                        help="How many times to train the classifiers (different seeds)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train the ResNet-18 from scratch (no ImageNet weights, no download)")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="config: conditional.guidance_scale")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="config: evaluation.num_inference_steps")
    parser.add_argument("--generation-batch-size", type=int, default=None,
                        help="config: evaluation.batch_size")
    parser.add_argument("--device", default=None, help="'auto', 'cuda' or 'cpu'")
    parser.add_argument("--cache-dir", default=None,
                        help="Folder of the generated image cache (default: in outputs_evaluations)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Ignore the cache and generate all images again")
    parser.add_argument("--output", default=None,
                        help="Path of the result JSON (default: in the outputs_evaluations folder)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    trts_config = config["trts_tstr"]
    evaluation_config = config["evaluation"]

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint(config["paths"]["checkpoints"])

    if not check_checkpoint_compatibility(checkpoint_path, config):
        return

    checkpoint_class_names = load_class_names(checkpoint_path)
    if checkpoint_class_names is None:
        print(f"Error: TRTS/TSTR needs a conditional checkpoint (no class_names.json in "
              f"{checkpoint_path}).")
        sys.exit(1)

    if args.classes is not None:
        requested = [name.strip() for name in args.classes.split(",") if name.strip()]
    else:
        requested = list(trts_config["classes"])

    if requested:
        unet_class_indices = []
        for name in requested:
            try:
                unet_class_indices.append(resolve_label_to_index(name, checkpoint_class_names))
            except ValueError:
                print(f"Error: unknown class '{name}'. Available: {checkpoint_class_names}")
                sys.exit(1)
    else:
        unet_class_indices = list(range(len(checkpoint_class_names)))

    samples_per_class = pick(args.samples_per_class, trts_config["samples_per_class"])
    test_samples_per_class = pick(args.test_samples_per_class, trts_config["test_samples_per_class"])
    repeats = pick(args.repeats, trts_config["repeats"])
    seed = config["seed"]
    training_settings = {
        "epochs": pick(args.epochs, trts_config["epochs"]),
        "batch_size": pick(args.batch_size, trts_config["batch_size"]),
        "learning_rate": pick(args.learning_rate, trts_config["learning_rate"]),
        "weight_decay": trts_config["weight_decay"],
        "pretrained": trts_config["pretrained"] and not args.no_pretrained,
    }
    guidance_scale = pick(args.guidance_scale, config["conditional"]["guidance_scale"])
    num_inference_steps = pick(args.num_inference_steps, evaluation_config["num_inference_steps"])
    generation_batch_size = pick(args.generation_batch_size, evaluation_config["batch_size"])
    device = resolve_device(pick(args.device, config["training"]["device"]))

    run_name = f"{checkpoint_path.parent.name}_{checkpoint_path.name}"
    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = (config["paths"]["outputs_evaluations"] / "trts_tstr_cache"
                     / f"{run_name}_g{guidance_scale:g}_s{num_inference_steps}")

    timestamp = time.strftime("%Y%m%d_%H%M")
    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = config["paths"]["outputs_evaluations"] / f"trts_tstr_{run_name}_{timestamp}.json"

    print("\nTRTR / TSTR / TRTS")
    print(f"  {'checkpoint':<24} {checkpoint_path}")
    print(f"  {'classes':<24} {len(unet_class_indices)}")
    print(f"  {'samples per class':<24} {samples_per_class} train, {test_samples_per_class} test")
    weights_description = "ImageNet-pretrained" if training_settings["pretrained"] else "from scratch"
    print(f"  {'classifier':<24} resnet18 ({weights_description}), {training_settings['epochs']} epochs")
    print(f"  {'repeats':<24} {repeats}")
    print(f"  {'guidance / steps':<24} {guidance_scale} / {num_inference_steps}")
    print(f"  {'device':<24} {device}")
    print(f"  {'cache':<24} {cache_dir}")
    print(f"  {'output':<24} {output_path}")
 
    print("\n--- Real images ---")
    datasets = build_datasets_from_config(config)
 
    # The Datasets's class names must match with the checkpoint's
    if datasets["train"].class_names != checkpoint_class_names:
        print("Error: the class list of the dataset differs from the checkpoint's:")
        print(f"  dataset:    {datasets['train'].class_names}")
        print(f"  checkpoint: {checkpoint_class_names}")
        sys.exit(1)
 
    # the train Dataset augments for the diffusion training; here the classifier augments
    # real and synthetic images the same way, so the plain val transform is used
    datasets["train"].transform = datasets["val"].transform
 
    rng = np.random.RandomState(seed)
    train_indices = select_indices_per_class(datasets["train"], unet_class_indices, samples_per_class, rng)
    test_indices = select_indices_per_class(datasets["test"], unet_class_indices, test_samples_per_class, rng)
 
    selected_classes = []
    excluded_classes = {}
    for unet_index in unet_class_indices:
        name = checkpoint_class_names[unet_index]
        train_count = len(train_indices[unet_index])
        test_count = len(test_indices[unet_index])
        if train_count < MIN_TRAIN_PER_CLASS:
            excluded_classes[name] = f"only {train_count} real training images (< {MIN_TRAIN_PER_CLASS})"
        elif test_count < MIN_TEST_PER_CLASS:
            excluded_classes[name] = f"only {test_count} real test images (< {MIN_TEST_PER_CLASS})"
        else:
            selected_classes.append(unet_index)
 
    for name, reason in excluded_classes.items():
        print(f"  Skipped {name}: {reason}")
    if len(selected_classes) < 2:
        print("Error: at least 2 classes with enough real images are needed.")
        sys.exit(1)
 
    # classifier labels are 0..K-1 in the order of the selected classes, not the UNet indices
    class_names = [checkpoint_class_names[unet_index] for unet_index in selected_classes]
    num_workers = config["data"]["num_workers"]
 
    real_train_parts, real_train_labels = [], []
    real_test_parts, real_test_labels = [], []
    counts = {}
    print(f"  {'class':<22}{'train':>7}{'test':>7}")
    for label, unet_index in enumerate(selected_classes):
        name = checkpoint_class_names[unet_index]
        real_train_parts.append(load_real_images(datasets["train"], train_indices[unet_index], num_workers))
        real_test_parts.append(load_real_images(datasets["test"], test_indices[unet_index], num_workers))
        real_train_labels += [label] * len(train_indices[unet_index])
        real_test_labels += [label] * len(test_indices[unet_index])
        counts[name] = {
            "real_train": len(train_indices[unet_index]),
            "real_test": len(test_indices[unet_index]),
            # as many generated images as real training images
            "synthetic": len(train_indices[unet_index]),
        }
        print(f"  {name:<22}{counts[name]['real_train']:>7d}{counts[name]['real_test']:>7d}")
 
    print("\n--- Synthetic images ---")
    check_cache_meta(cache_dir, {
        # run + epoch name instead of the full path, so the cache stays valid on Colab and locally
        "checkpoint": run_name,
        "image_size": config["data"]["image_size"],
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "class_names": checkpoint_class_names,
    })
    generator = SyntheticImageGenerator(
        checkpoint_path, config, device,
        batch_size=generation_batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        null_class_index=len(checkpoint_class_names),
    )

    synthetic_parts, synthetic_labels = [], []
    for label, unet_index in enumerate(selected_classes):
        name = checkpoint_class_names[unet_index]
        images = get_synthetic_images(generator, cache_dir, unet_index, name,
                                      counts[name]["synthetic"], seed, args.regenerate)
        synthetic_parts.append(images)
        synthetic_labels += [label] * images.shape[0]

    data = {
        "real_train_images": np.concatenate(real_train_parts, axis=0),
        "real_train_labels": np.array(real_train_labels, dtype=np.int64),
        "real_test_images": np.concatenate(real_test_parts, axis=0),
        "real_test_labels": np.array(real_test_labels, dtype=np.int64),
        "synthetic_images": np.concatenate(synthetic_parts, axis=0),
        "synthetic_labels": np.array(synthetic_labels, dtype=np.int64),
    }

    metadata = {
        "checkpoint": str(checkpoint_path),
        "class_names": class_names,
        "excluded_classes": excluded_classes,
        "counts": counts,
        "samples_per_class": samples_per_class,
        "test_samples_per_class": test_samples_per_class,
        "image_size": config["data"]["image_size"],
        "classifier": dict(training_settings, architecture="resnet18"),
        "repeats": repeats,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "synthetic_cache": str(cache_dir),
        "real_train_split": "train",
        "real_test_split": "test",
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    runs = []
    for repeat in range(1, repeats + 1):
        repeat_seed = seed + repeat - 1
        print(f"\n--- Repeat {repeat}/{repeats} (seed {repeat_seed}) ---")
        repeat_start = time.time()
 
        results = run_repeat(data, len(class_names), training_settings, device, repeat_seed)
        runs.append({"repeat": repeat, "seed": repeat_seed, "results": results})
 
        line = "  ".join(f"{name}: {results[name]['balanced_accuracy']:.3f}" for name in EXPERIMENTS)
        print(f"  balanced accuracy  {line}  ({time.time() - repeat_start:.0f}s)")
 
        # saved after every repeat, so an interrupted run keeps its results
        save_report(output_path, metadata, runs, summarize(runs, class_names), complete=False)
 
    summary = summarize(runs, class_names)
    save_report(output_path, metadata, runs, summary, complete=True)
 
    print_summary(summary, class_names, counts, repeats)
    print(f"\nResults saved: {output_path}")
 
 
if __name__ == "__main__":
    main()
