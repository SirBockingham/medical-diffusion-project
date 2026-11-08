import csv
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.common import build_filename_index, parse_label_string, read_filename_csv
from utils.config import load_config


def _discover_labels(rows: list[dict], label_col: str, separator: str, no_finding_value: str | None) -> list[str]:
    """Sorted list of every label that appears in the label column."""
    
    unique_labels = set()

    for row in rows:
        raw_value = row.get(label_col, "")
        labels_in_row = parse_label_string(raw_value, separator, no_finding_value)
        for label in labels_in_row:
            unique_labels.add(label)
    
    return sorted(unique_labels)



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
        single_label_only: bool = False
    ):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.single_label_only = single_label_only
        
        filename_to_path = build_filename_index(self.images_dir)
        
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if labels is None:
            self.labels = _discover_labels(rows, label_col, label_separator, no_finding_value)
            if len(self.labels) == 0:
                print("No labels found")
                
        else:
            self.labels = labels
            
        self.label_to_idx = {}
        for index, label in enumerate(self.labels):
            self.label_to_idx[label] = index
        
        
        no_finding_class_name = no_finding_value if no_finding_value is not None else "No Finding"
        self.class_names = list(self.labels) + [no_finding_class_name]
        self.no_finding_class_index = len(self.labels)
        
        self.null_class_index = len(self.class_names)
        self.num_class_embeds = len(self.class_names) + 1
        
        self.class_name_to_index = {}
        for index, class_name in enumerate(self.class_names):
            self.class_name_to_index[class_name] = index
            
        
        self.samples = []
        
        for row in rows:
            filename = row[filename_col].strip()
            
            if allowed_filenames is not None and filename not in allowed_filenames:
                continue
            
            full_path = filename_to_path.get(filename)
            raw_label_value = row.get(label_col, "")
            parsed_labels = parse_label_string(raw_label_value, label_separator, no_finding_value)
            
            
            known_labels_for_sample = []
            for label in parsed_labels:
                if label in self.label_to_idx:
                    known_labels_for_sample.append(label)
                    
            if len(known_labels_for_sample) == 0:
                class_index = self.no_finding_class_index
            elif len(known_labels_for_sample) == 1:
                class_index = self.label_to_idx[known_labels_for_sample[0]]
            else:
                if single_label_only:
                    continue
                class_index = self.label_to_idx[known_labels_for_sample[0]]

            self.samples.append({
                "filename": filename,
                "path": full_path,
                "labels": known_labels_for_sample,
                "class_index": class_index,
                "patient_id": row.get(patient_id_col)
            })
        
        print(f"{len(self.samples)} samples loaded, {len(self.labels)} unique labels loaded")
        if single_label_only:
            print(f"    -> conditional mode: {len(self.class_names)} classes + 1 null token")
        
        
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
            "class_index": sample["class_index"],
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
    
    dataset = MedicalImageDataset(
        csv_path=str(config["paths"]["csv_path"]),
        images_dir=str(config["paths"]["data_raw"]),
        image_size=config["data"]["image_size"]
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"{len(dataset.labels)} discovered labels: {dataset.labels}")
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Labels: {sample['label_names']}")
    print(f"Label vector: {sample['labels']}")
    
    
    dataset.check(n=4)
    