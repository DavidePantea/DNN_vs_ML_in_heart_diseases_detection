import os
from pathlib import Path
import kagglehub

DATASET = "redwankarimsony/heart-disease-data"

# Put everything under ./datasets (or override with env var)
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", "datasets")).resolve()
DEST_DIR = DATASETS_DIR / "heart-disease-data"

def looks_downloaded(folder: Path) -> bool:
    """
    'Exists' check: folder exists and contains at least one non-hidden file.
    Adjust this if you want to check for a specific filename.
    """
    if not folder.exists() or not folder.is_dir():
        return False
    for p in folder.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            return True
    return False

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if looks_downloaded(DEST_DIR):
        print(f"Dataset already present at: {DEST_DIR}")
        return

    # Force KaggleHub to use a cache inside our project (so we control where it goes)
    os.environ["KAGGLEHUB_CACHE"] = str(DEST_DIR)

    print("Downloading dataset...")
    path = kagglehub.dataset_download(DATASET)
    print("KaggleHub returned path:", path)
    print("Dataset stored under:", DEST_DIR)

if __name__ == "__main__":
    main()
