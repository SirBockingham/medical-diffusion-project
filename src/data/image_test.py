import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import load_config

from PIL import Image



def build_filename_index(images_dir: Path) -> dict:
    index = {}
    exts = {".png", ".jpg", ".jpeg", "dcm"}
    for root, _dirs, files in os.walk(images_dir):
        for fname in files:
            if Path(fname).suffix.lower() in exts:
                index.setdefault(fname, os.path.join(root, fname))
    return index


def check_image_openable(path: str) -> bool:
    try:
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
    parser.add_argument("--filename-col", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-decode-check", action="store_true")
    args = parser.parse_args()
    
    
    config = load_config()
    
    
    if args.csv is not None:
        csv_path = Path(args.csv)
    else:
        csv_path = config["paths"]["csv_path"]
        
    if args.images_dir is not None:
        images_dir = Path(args.images_dir)
    else:
        images_dir = config["paths"]["data_raw"]
        
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config["paths"]["data_metadata"] 
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Error: image directory not found: {images_dir}")
        sys.exit(1)
        
    print(f"Reading image directory and building filename-index: {images_dir} ...")
    filename_index = build_filename_index(images_dir)
    print(f"  -> {len(filename_index)} image files found.\n")
 
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if args.filename_col not in reader.fieldnames:
            print(f"Error: the '{args.filename_col}' column not found in CSV.")
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
            continue
        
        if args.skip_decode_check:
            ok_count += 1
        else:
            if check_image_openable(full_path):
                ok_count += 1
            else:
                corrupted.append(fname)
        
        if i % 5000 == 0 or i == total:
            print(f" ... {i}/{total} processed")
 
    print("\n--- Summary ---")
    print(f"Entries in CSV:  {total}")
    print(f"OK (openable):         {ok_count}")
    print(f"Missing files:                  {len(missing)}")
    print(f"Damaged / non-decodable files: {len(corrupted)}")



if __name__ == "__main__":
    main()