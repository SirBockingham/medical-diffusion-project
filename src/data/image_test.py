import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.common import build_filename_index
from utils.config import load_config, resolve_path


def check_image_openable(path: str) -> bool:
    try:
        # verify() catches broken file structure, load() catches broken pixel data
        with Image.open(path) as img:
            img.verify()
        
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--filename-col", default="Image Index")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-decode-check", action="store_true")
    args = parser.parse_args()
    
    
    config = load_config()
    
    
    csv_path = resolve_path(args.csv, config["paths"]["csv_path"])
    images_dir = resolve_path(args.images_dir, config["paths"]["data_raw"])
    output_dir = resolve_path(args.output_dir, config["paths"]["data_metadata"])
    
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Error: image directory not found: {images_dir}")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Indexing image directory: {images_dir} ...")
    filename_index = build_filename_index(images_dir)
    print(f"  -> {len(filename_index)} image files found.\n")
 
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or args.filename_col not in reader.fieldnames:
            print(f"Error: column '{args.filename_col}' not found in CSV.")
            print(f"Available columns: {reader.fieldnames}")
            sys.exit(1)
        rows = list(reader)
 
 
    total = len(rows)
    missing = []
    corrupted = []
    ok_count = 0
    
    
    print(f"Checking {total} entries...")
    for i, row in enumerate(rows, start=1):
        fname = row[args.filename_col].strip()
        full_path = filename_index.get(fname)
        
        
        if full_path is None:
            missing.append(fname)
        
        elif args.skip_decode_check or check_image_openable(full_path):
            ok_count += 1
        else:
           corrupted.append(fname)
        
        if i % 5000 == 0 or i == total:
            print(f"  ... {i}/{total} processed")
 
    print("\n--- Summary ---")
    print(f"Entries in CSV:     {total}")
    print(f"OK (openable):      {ok_count}")
    print(f"Missing files:      {len(missing)}")
    print(f"Corrupted / unreadable: {len(corrupted)}")



if __name__ == "__main__":
    main()