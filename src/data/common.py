import csv
import os
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}



def build_filename_index(images_dir: str | Path, extensions: set[str] = IMAGE_EXTENSIONS) -> dict[str, str]:
    """Maps every image filename under images_dir (recursively) to its full path.
    If the same filename appears twice, the first one found wins."""
    filename_to_path: dict[str, str] = {}
    for folder_path, _subfolders, files in os.walk(images_dir):
        for filename in files:
            if Path(filename).suffix.lower() in extensions:
                filename_to_path.setdefault(filename, os.path.join(folder_path, filename))
    return filename_to_path



def parse_label_string(raw_value: str, separator: str, no_finding_value: str | None) -> list[str]:
    """Turns one label cell of the CSV (e.g. "Effusion|Mass") into a list of labels."""
    raw_value = raw_value.strip()

    no_label_present = raw_value == ""
    marked_as_no_finding = no_finding_value is not None and raw_value == no_finding_value
    if no_label_present or marked_as_no_finding:
        return []

    cleaned_labels = []
    for part in raw_value.split(separator):
        cleaned_part = part.strip()
        if cleaned_part != "":
            cleaned_labels.append(cleaned_part)

    return cleaned_labels



def read_filename_csv(csv_path: str | Path | None) -> set[str]:
    """Reads a one-column CSV with a "filename" header (split files, missing/corrupted lists).
    A missing file gives an empty set."""
    if csv_path is None or not Path(csv_path).exists():
        return set()

    filenames = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filenames.add(row["filename"].strip())
    return filenames



def write_filename_csv(filenames: list[str], output_path: str | Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        for filename in filenames:
            writer.writerow([filename])