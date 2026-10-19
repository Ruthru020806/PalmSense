from pathlib import Path
from PIL import Image


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
VALID_DIR = PROJECT_ROOT / "dataset" / "valid"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"

PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"


# --------------------------------------------------
# IMAGE SIZE
# --------------------------------------------------

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# SUPPORTED IMAGE TYPES
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# --------------------------------------------------
# RESIZE FUNCTION
# --------------------------------------------------

def resize_images(source_dir, destination_dir):

    total = 0

    for image_path in source_dir.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Preserve class folder structure
        relative_path = image_path.relative_to(source_dir)

        output_path = destination_dir / relative_path

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            image = Image.open(image_path)

            # Convert to RGB
            image = image.convert("RGB")

            # Resize
            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            # Save
            image.save(
                output_path,
                quality=95
            )

            total += 1

        except Exception as error:

            print(
                f"ERROR processing {image_path}: {error}"
            )

    return total


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("PALM RIPENESS PREDICTION")
    print("IMAGE RESIZING")
    print("=" * 60)

    splits = {
        "train": TRAIN_DIR,
        "valid": VALID_DIR,
        "test": TEST_DIR
    }

    total_images = 0

    for split_name, source_dir in splits.items():

        print()
        print(f"Processing: {split_name}")

        destination_dir = (
            PROCESSED_DIR / "resized" / split_name
        )

        count = resize_images(
            source_dir,
            destination_dir
        )

        print(f"Resized images: {count}")

        total_images += count

    print()
    print("=" * 60)
    print(f"TOTAL IMAGES RESIZED: {total_images}")
    print("=" * 60)