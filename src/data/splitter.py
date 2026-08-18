import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config


def read_csv_rows(csv_path: str | Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
    
    

def get_dominant_label(patient_labels: list[str]) -> str:
    if len(patient_labels) == 0:
        return "No Finding"
    
    label_counts = Counter(patient_labels)
    most_common_label, _count = label_counts.most_common(1)[0]
    return most_common_label



def parse_label_string(raw_value: str, separator: str, no_finding_value: str | None) -> list[str]:
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



def group_filenames_by_patient(
    rows: list[dict],
    filename_col: str,
    patient_id_col: str,
    label_col: str,
    label_separator: str,
    no_finding_value: str | None
) -> dict[str, list[str]]:
    patient_to_filenames = defaultdict(list)
    for row in rows:
        patient_id = row[patient_id_col].strip()
        filename = row[filename_col].strip()
        patient_to_filenames[patient_id].append(filename)
    return patient_to_filenames



def get_patient_dominant_labels(
    rows: list[dict],
    patient_id_col: str,
    label_col: str,
    label_separator: str,
    no_finding_value: str,
) -> dict[str, str]:
    patient_to_all_labels = defaultdict(list)
 
    for row in rows:
        patient_id = row[patient_id_col].strip()
        raw_label_value = row.get(label_col, "")
        labels_in_row = parse_label_string(raw_label_value, label_separator, no_finding_value)
        patient_to_all_labels[patient_id].extend(labels_in_row)
 
    patient_to_dominant_label = {}
    for patient_id, all_labels in patient_to_all_labels.items():
        patient_to_dominant_label[patient_id] = get_dominant_label(all_labels)
 
    return patient_to_dominant_label



def split_patients_per_label(
    patients_with_this_label: list[str],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> dict[str, list[str]]:
    shuffled_patients = list(patients_with_this_label)
    rng.shuffle(shuffled_patients)
    
    patient_count = len(shuffled_patients)
    train_count = round(patient_count * train_ratio)
    val_count = round(patient_count * val_ratio)
    
    train_patients = shuffled_patients[:train_count]
    val_patients = shuffled_patients[train_count:train_count + val_count]
    test_patients = shuffled_patients[val_count:]


    return {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients
    }
    
    
    
def write_filenames_csv(filenames: list[str], output_path: Path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        for filename in filenames:
            writer.writerow([filename])
            


def print_label_distribution(split_name: str, filenames: list[str], filename_to_labels: dict[str, list[str]]):
    label_counter = Counter()
    for filename in filenames:
        for label in filename_to_labels.get(filename, []):
            label_counter[label] += 1
            
            
    print(f"\n {split_name} label distribution {len(filenames)} images:")
    for label, count in sorted(label_counter.items()):
        percent = 100 * count / len(filenames) if filenames else 0
        print(f"    {label:25s} {count:6d} kep ({percent:.1f}%)")
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--filename-col", default="Image Index")
    parser.add_argument("--label-col", default="Finding Labels")
    parser.add_argument("--patient-id-col", default="Patient ID")
    parser.add_argument("--label-separator", default="|")
    parser.add_argument("--no-finding-value", default="No Finding")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random number generator seed for reproductability")
    args = parser.parse_args()
    
    
    config = load_config()
    
    
    def resolve(cli_value, config_value):
            return cli_value if cli_value is not None else config_value
        
    
    csv_path = resolve(Path(args.csv), config["paths"]["csv_path"])
    train_ratio = resolve(args.train_ratio, config["data"]["train_ratio"])
    val_ratio = resolve(args.val_ratio, config["data"]["val_ratio"])
    test_ratio = resolve(args.test_ratio, config["data"]["test_ratio"])
    seed = resolve(args.seed, config["seed"])
 
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config["paths"]["data_splits"]
 
    
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        print(f"Error: the sum of the train/val/test ratios is not 1.0 (current: {ratio_sum}).")
        return
    
    
    rng = random.Random(seed)
    
    print(f"Loading CSV: {csv_path} ...")
    rows = read_csv_rows(csv_path)
    print(f"    -> {len(rows)} rows loaded.")
    
    
    # Collect the filenames for every patient
    patient_to_filenames = group_filenames_by_patient(
        rows,
        args.filename_col,
        args.patient_id_col,
        args.label_col,
        args.label_separator,
        args.no_finding_value
    )
    
    print(f"    -> {len(patient_to_filenames)} patients found")
    
    
    # Determine dominant label
    patient_to_dominant_label = get_patient_dominant_labels(
        rows,
        args.patient_id_col,
        args.label_col,
        args.label_separator,
        args.no_finding_value
    )
    
    
    # Group patients by dominant label
    label_to_patients = defaultdict(list)
    for patient_id, dominant_label in patient_to_dominant_label.items():
        label_to_patients[dominant_label].append(patient_id)
        
    print("\nDominant label based distribution:")
    for label, patients in sorted(label_to_patients.items()):
        print(f"    {label:25s} {len(patients):5d} patients")
        
    
    train_patients: list[str] = []
    val_patients: list[str] = []
    test_patients: list[str] = []
    
    for label, patients_with_label in label_to_patients.items():
        split_result = split_patients_per_label(patients_with_label, train_ratio, val_ratio, rng)
        train_patients.extend(split_result["train"])
        val_patients.extend(split_result["val"])
        test_patients.extend(split_result["test"])
    
    
    def patients_to_filenames(patient_ids: list[str]) -> list[str]:
        filenames = []
        for patient_id in patient_ids:
            filenames.extend(patient_to_filenames[patient_id])
        return filenames
 
    train_filenames = patients_to_filenames(train_patients)
    val_filenames = patients_to_filenames(val_patients)
    test_filenames = patients_to_filenames(test_patients)
    
    
    # Writing results into file 
    output_dir.mkdir(parents=True, exist_ok=True)
 
    write_filenames_csv(train_filenames, output_dir / "train_filenames.csv")
    write_filenames_csv(val_filenames, output_dir / "val_filenames.csv")
    write_filenames_csv(test_filenames, output_dir / "test_filenames.csv")
 
    print(f"\nSplit files saved to: {output_dir}/")
    print(f"  train: {len(train_patients)} patient(s), {len(train_filenames)} image(s)")
    print(f"  val:   {len(val_patients)} patient(s), {len(val_filenames)} image(s)")
    print(f"  test:  {len(test_patients)} patient(s), {len(test_filenames)} image(s)")
    
    
    
    filename_to_labels: dict[str, list[str]] = {}
    for row in rows:
        filename = row[args.filename_col].strip()
        raw_label_value = row.get(args.label_col, "")
        filename_to_labels[filename] = parse_label_string(
            raw_label_value, args.label_separator, args.no_finding_value,
        )
        
    print("\nLabel distribution check for every split")
    print_label_distribution("train", train_filenames, filename_to_labels)
    print_label_distribution("val", val_filenames, filename_to_labels)
    print_label_distribution("test", test_filenames, filename_to_labels)
    
    
    
if __name__ == "__main__":
    main()