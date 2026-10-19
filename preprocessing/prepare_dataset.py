from pathlib import Path
import pandas as pd
import shutil


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
VALID_DIR = PROJECT_ROOT / "dataset" / "valid"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"


# Classes
CLASSES = [
    "Over-ripe",
    "Ripe",
    "Under-ripe",
    "Unripe"
]


# Create folders
for split_dir in [TRAIN_DIR, VALID_DIR, TEST_DIR]:

    for class_name in CLASSES:

        (split_dir / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


def find_image(file_path):

    # Try path directly
    candidate = RAW_DIR / file_path

    if candidate.exists():
        return candidate

    # Otherwise search by filename
    filename = Path(file_path).name

    matches = list(
        (RAW_DIR / "Images").rglob(filename)
    )

    if matches:
        return matches[0]

    return None


def process_split(csv_name, destination_dir):

    csv_path = RAW_DIR / csv_name

    if not csv_path.exists():
        print(f"ERROR: {csv_name} not found")
        return

    print()
    print("=" * 60)
    print(f"Processing {csv_name}")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    print(f"Images listed in CSV: {len(df)}")

    copied = 0
    missing = 0

    for _, row in df.iterrows():

        file_path = str(row["file_path"])
        class_name = str(row["class_name"])

        source = find_image(file_path)

        if source is None:

            print(f"WARNING: Image not found: {file_path}")

            missing += 1
            continue

        destination = (
            destination_dir
            / class_name
            / source.name
        )

        shutil.copy2(source, destination)

        copied += 1

    print(f"Copied: {copied}")
    print(f"Missing: {missing}")


if __name__ == "__main__":

    print("=" * 60)
    print("PALM RIPENESS PREDICTION")
    print("DATASET PREPARATION")
    print("=" * 60)

    print()
    print("Classes:")

    for index, class_name in enumerate(CLASSES):
        print(f"{index} -> {class_name}")

    process_split(
        "train.csv",
        TRAIN_DIR
    )

    process_split(
        "valid.csv",
        VALID_DIR
    )

    process_split(
        "test.csv",
        TEST_DIR
    )

    print()
    print("=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)