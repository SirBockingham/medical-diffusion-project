import csv
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config


def build_filename_index(images_dir: Path) -> dict:
    filename_to_path = {}
    allowed_extensions = {".png", ".jpg", ".jpeg"}
    for folder_path, _subfolders, files in os.walk(images_dir):
        for filename in files:
            if Path(filename).suffix.lower() in allowed_extensions:
                filename_to_path.setdefault(filename, os.path.join(folder_path, filename))
    return filename_to_path



def _parse_label_string(raw_value: str, separator: str, no_finding_value: str | None) -> list[str]:
    """
    Converts one cell from the CSV into a Python list. 
    """
    
    raw_value = raw_value.strip()
    
    no_label_present = raw_value == ""
    marked_as_no_finding = no_finding_value is not None and raw_value == no_finding_value
    if no_label_present or marked_as_no_finding:
        return []
    
    label_parts = raw_value.split(separator)
    
    cleaned_labels = []
    for part in label_parts:
        cleaned_part = part.strip()
        if cleaned_part != "":
            cleaned_labels.append(cleaned_part)
    
    return cleaned_labels



def _discover_labels(rows: list[dict], label_col: str, separator: str, no_finding_value: str | None) -> list[str]:
    """Collects all of the appearing unique labels

    Args:
        rows (list[dict]): A list of the key-value pairs for every row
        label_col (str): Name of the column containing the labels
        separator (str): Character separating the labels
        no_finding_value (str | None): Optional value for no finding

    Returns:
        list[str]: A sorted list of the unique labels
    """
    
    unique_labels = set()

    for row in rows:
        raw_value = row.get(label_col, "")
        labels_in_row = _parse_label_string(raw_value, separator, no_finding_value)
        for label in labels_in_row:
            unique_labels.add(label)
    
    return sorted(unique_labels)



def load_filenames(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["filename"] for row in csv.DictReader(f)}



class MedicalImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        filename_col: str = "Image Index",
        label_col: str = "Finding Labels",
        patient_id_col: str = "Patient ID",
        label_separator: str = "|",
        no_finding_value: str | None = "No Finding",
        missing_files_csv: str | None = None,
        corrupted_files_csv: str | None = None,
        image_size: int = 256,
        labels: list[str] | None = None,
        allowed_filenames: set | None = None,
        augment: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        
        filename_to_path = build_filename_index(self.images_dir)
        
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if labels is None:
            self.labels = _discover_labels(rows, label_col, label_separator, no_finding_value)
            if  len(self.labels) == 0:
                print("No labels found")
                
        else:
            self.labels = labels
            
        self.label_to_idx = {}
        for index, label in enumerate(self.labels):
            self.label_to_idx[label] = index
        
        
        self.samples = []
        
        for row in rows:
            filename = row[filename_col].strip()
            
            if allowed_filenames is not None and filename not in allowed_filenames:
                continue
            
            full_path = filename_to_path.get(filename)
            raw_label_value = row.get(label_col, "")
            parsed_labels = _parse_label_string(raw_label_value, label_separator, no_finding_value)

            self.samples.append({
                "filename": filename,
                "path": full_path,
                "labels": parsed_labels,
                "patient_id": row.get(patient_id_col)
            })
        
        print(f"{len(self.samples)} samples loaded, {len(self.labels)} unique labels loaded")
        
        # Image transformations
        transform_steps: list[Callable] = [T.Resize((image_size, image_size))]
 
        if augment:
            transform_steps.append(T.RandomHorizontalFlip(p=0.5))
            transform_steps.append(T.RandomRotation(degrees=10))
            
        transform_steps.append(T.ToTensor())
        transform_steps.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
        self.transform = T.Compose(transform_steps)
        
        
        
    def __len__(self) -> int:
        return len(self.samples)
      
      
      
    def labels_to_multihot(self, label_list: list[str]) -> torch.Tensor:
        multihot_vector = torch.zeros(len(self.labels), dtype=torch.float32)
        for label in label_list:
            label_index = self.label_to_idx.get(label)
            multihot_vector[label_index] = 1.0
        return multihot_vector
    
    
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        image = Image.open(sample["path"]).convert("RGB")
        image_tensor = self.transform(image)
        
        label_vector = self.labels_to_multihot(sample["labels"])
        
        return {
            "image": image_tensor,
            "labels": label_vector,
            "label_names": sample["labels"],
            "filename": sample["filename"],
            "patient_id": sample["patient_id"],
        }
        
    def check(self, n: int = 4, save_path: str = "check.png"):
        import matplotlib.pyplot as plt
 
        sample_count = min(n, len(self))
        random_indices = np.random.choice(len(self), size=sample_count, replace=False)
 
        _fig, axes = plt.subplots(1, sample_count, figsize=(4 * sample_count, 4))
        if sample_count == 1:
            axes = [axes]
 
        for ax, idx in zip(axes, random_indices):
            sample = self[idx]
 
            image_for_display = sample["image"].permute(1, 2, 0).numpy()
            image_for_display = (image_for_display * 0.5) + 0.5
            image_for_display = np.clip(image_for_display, 0, 1)
 
            if sample["label_names"]:
                title = ", ".join(sample["label_names"])
            else:
                title = "No Finding"
 
            ax.imshow(image_for_display)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
 
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"Check saved: {save_path}")
        
        

        
if __name__ == "__main__":
    config = load_config()
    
    dataset = MedicalImageDataset (
        csv_path=str(config["paths"]["csv_path"]),
        images_dir=str(config["paths"]["data_raw"]),
        image_size=config["data"]["image_size"]
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"{len(dataset.labels)} discovered labels: {dataset.labels}")
    
    sample = dataset[0]
    print(f"Image shape: {sample["image"].shape}")
    print(f"Labels: {sample["label_names"]}")
    print(f"Label vector: {sample["labels"]}")
    
    
    dataset.check(n=4)
    