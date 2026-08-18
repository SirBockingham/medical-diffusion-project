import argparse
import csv
import sys
from pathlib import Path

import torch
from dataset import MedicalImageDataset
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config


def collate_batch(batch: list[dict]) -> dict:
    images = torch.stack([sample["image"] for sample in batch])
    labels = torch.stack([sample["labels"] for sample in batch])
 
    label_names = [sample["label_names"] for sample in batch]
    filenames = [sample["filename"] for sample in batch]
    patient_ids = [sample["patient_id"] for sample in batch]


    return {
        "image": images,
        "labels": labels,
        "label_names": label_names,
        "filename": filenames,
        "patient_id": patient_ids,
    }
    
    

def load_filenames_from_split_csv(split_csv_path: Path) -> set[str]:
    filenames = set()
    with open(split_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filenames.add(row["filename"])
    return filenames



def build_datasets(
        csv_path: str,
        img_dir: str,
        splits_dir: str,
        img_size: int = 256
    ) -> dict[str, MedicalImageDataset]:
    
    splits_dir_path = Path(splits_dir)
    
    
    train_filenames = load_filenames_from_split_csv(splits_dir_path / "train_filenames.csv")
    val_filenames = load_filenames_from_split_csv(splits_dir_path / "val_filenames.csv")
    test_filenames = load_filenames_from_split_csv(splits_dir_path / "test_filenames.csv")
    
    
    print("Loading train dataset with label discovery...")
    train_dataset = MedicalImageDataset(
        csv_path=csv_path,
        images_dir=img_dir,
        allowed_filenames=train_filenames,
        image_size=img_size,
        augment=True,
    )
    
    print("\nLoading validation dataset with train label order...")
    val_dataset = MedicalImageDataset(
        csv_path=csv_path,
        images_dir=img_dir,
        allowed_filenames=val_filenames,
        image_size=img_size,
        labels=train_dataset.labels,
        augment=False
    )
    
    print("\nLoading validation dataset with train label order...")
    test_dataset = MedicalImageDataset(
        csv_path=csv_path,
        images_dir=img_dir,
        allowed_filenames=test_filenames,
        image_size=img_size,
        labels=train_dataset.labels,
        augment=False
    )
    
    
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }
    
    
    
def build_datasets_from_config(config: dict | None = None) -> dict[str, MedicalImageDataset]:
    if config is None:
        config = load_config()
        
    return build_datasets(
        csv_path=str(config["paths"]["csv_path"]),
        img_dir=str(config["paths"]["data_raw"]),
        splits_dir=str(config["paths"]["data_splits"]),
        img_size=config["data"]["image_size"]
    )
    
    

def build_dataloaders(
        csv_path: str,
        img_dir: str,
        splits_dir: str,
        img_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> dict[str, DataLoader]:
    
    datasets = build_datasets(
        csv_path=csv_path,
        img_dir=img_dir,
        splits_dir=splits_dir,
        img_size=img_size
    )
    
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    
    
def build_dataloaders_from_config(config: dict | None = None) -> dict[str, DataLoader]:
    if config is None:
        config = load_config()
        
    return build_dataloaders(
        csv_path=str(config["paths"]["csv_path"]),
        img_dir=str(config["paths"]["data_raw"]),
        splits_dir=str(config["paths"]["data_splits"]),
        img_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"]
    )
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--check-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    
    
    config = load_config()
    
    
    def resolve(cli_value, config_value):
        return cli_value if cli_value is not None else config_value
    

    csv_path = resolve(args.csv, str(config["paths"]["csv_path"]))
    images_dir = resolve(args.images_dir, str(config["paths"]["data_raw"]))
    splits_dir = resolve(args.splits_dir, str(config["paths"]["data_splits"]))
    image_size = resolve(args.image_size, config["data"]["image_size"])
    batch_size = resolve(args.batch_size, config["data"]["batch_size"])
    num_workers = resolve(args.num_workers, config["data"]["num_workers"])
    output_dir = resolve(args.output_dir, config["paths"]["outputs_checks"])
    
    
    datasets = build_datasets(
        csv_path=csv_path,
        img_dir=images_dir,
        splits_dir=splits_dir,
        img_size=image_size,
    )
    
    print("\n--- Dataset sizes ---")
    for split_name, dataset in datasets.items():
        print(f"  {split_name}: {len(dataset)} images")
 
    # DataLoaders
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
 
    print("\n--- Check batches ---")
    for split_name, loader in dataloaders.items():
        first_batch = next(iter(loader))
        image_shape = tuple(first_batch["image"].shape)
        labels_shape = tuple(first_batch["labels"].shape)
        print(f"  {split_name}: image batch shape = {image_shape}, label batch shape = {labels_shape}")
 
    # saving check images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    print("\n--- Generating check images ---")
    for split_name, dataset in datasets.items():
        save_path = output_dir / f"check_{split_name}.png"
        dataset.check(n=args.check_samples, save_path=str(save_path))