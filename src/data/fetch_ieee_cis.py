"""
Download the IEEE-CIS Fraud Detection competition data from Kaggle into data/raw/.

Setup (one-time):
  1. Join the competition at https://www.kaggle.com/c/ieee-fraud-detection/rules
     (Kaggle requires this before the API will let you download the files.)
  2. Create a Kaggle API token: Kaggle account settings -> "Create New Token".
     This downloads kaggle.json -- place it at C:\\Users\\<you>\\.kaggle\\kaggle.json
  3. pip install -r requirements.txt

Usage:
  python -m src.data.fetch_ieee_cis
"""

import zipfile
from pathlib import Path

COMPETITION = "ieee-fraud-detection"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {COMPETITION} into {RAW_DIR} ...")
    api.competition_download_files(COMPETITION, path=str(RAW_DIR))

    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()

    print("Done. Files in data/raw/:")
    for f in sorted(RAW_DIR.iterdir()):
        print(f" - {f.name}")


if __name__ == "__main__":
    main()
