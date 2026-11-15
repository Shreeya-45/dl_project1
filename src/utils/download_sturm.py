"""
STURM-FloodDepth Dataset Downloader & Organizer
==============================================
Downloads and organizes the "STURM-FloodDepth Flooded Cars" dataset from Zenodo:
    Record: https://zenodo.org/records/14833532
    DOI   : 10.5281/zenodo.14833532

Expected Structure in `data/sturm/`:
    data/sturm/
    ├── Level0/          (Dry / No Water)
    ├── Level1/          (Water below wheel axle / <15 cm)
    ├── Level2/          (Water covering wheel hub / 15-30 cm)
    ├── Level3/          (Water reaching chassis / windows / 30-60 cm)
    ├── Level4/          (Submerged / >60 cm)
    ├── train.txt        (List of training image paths)
    ├── val.txt          (List of validation image paths)
    └── test.txt         (List of test image paths)
"""

import os
import sys
import argparse
import urllib.request
import json
import zipfile

ZENODO_RECORD_ID = "14833532"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def download_sturm_dataset(output_dir: str = "data/sturm"):
    """
    Queries the Zenodo API for record 14833532 and downloads dataset files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print("   STURM-FLOODDEPTH DATASET DOWNLOADER (ZENODO: 14833532)")
    print("=" * 65)
    print(f"Target Directory: {os.path.abspath(output_dir)}")
    print(f"Querying Zenodo API at {ZENODO_API_URL} ...\n")

    try:
        req = urllib.request.Request(
            ZENODO_API_URL,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))

        files = data.get('files', [])
        print(f"Found {len(files)} files in Zenodo record:")
        for f in files:
            name = f.get('key', f.get('filename', 'unknown'))
            size_mb = f.get('size', 0) / (1024 * 1024)
            print(f"  - {name} ({size_mb:.2f} MB)")

        for f in files:
            name = f.get('key', f.get('filename'))
            download_url = f.get('links', {}).get('self')
            dest_path = os.path.join(output_dir, name)

            if os.path.exists(dest_path):
                print(f"File '{name}' already exists. Skipping download.")
            else:
                print(f"\nDownloading '{name}' from {download_url}...")
                urllib.request.urlretrieve(download_url, dest_path)
                print(f"Saved: {dest_path}")

            # If zip, auto-extract
            if dest_path.endswith('.zip'):
                print(f"Extracting '{name}' into {output_dir}...")
                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                print("Extraction complete.")

        print("\n[SUCCESS] STURM dataset downloaded and organized successfully!")
    except Exception as e:
        print(f"\n[INFO / NOTICE] Direct API query returned: {e}")
        print("If running in an offline environment or without direct Zenodo network access,")
        print("you can download the files manually from:")
        print("    https://zenodo.org/records/14833532")
        print("\nPlace the downloaded folders or zip files into:")
        print(f"    {os.path.abspath(output_dir)}")
        print("\nEnsure the 5 level folders are present:")
        print("    data/sturm/Level0/")
        print("    data/sturm/Level1/")
        print("    data/sturm/Level2/")
        print("    data/sturm/Level3/")
        print("    data/sturm/Level4/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and organize STURM-FloodDepth dataset from Zenodo")
    parser.add_argument('--output-dir', type=str, default='data/sturm', help='Path to target directory')
    args = parser.parse_args()
    download_sturm_dataset(args.output_dir)

