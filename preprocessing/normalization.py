from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLAHE_DIR = PROJECT_ROOT / "dataset" / "processed" / "clahe"
NORMALIZED_DIR = PROJECT_ROOT / "dataset" / "processed" / "normalized"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

def normalize_images(source_dir, destination_dir):
    total = 0

    for image_path in source_dir.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(source_dir)

        output_path = destination_dir / relative_path
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        try:
            image = Image.open(image_path).convert("RGB")

            # Convert image to NumPy array
            image_array = np.array(image).astype(np.float32)

            # Normalize pixel values from [0, 255] to [0, 1]
            normalized_array = image_array / 255.0

            # Convert back to [0, 255] for image storage
            normalized_image = (
                normalized_array * 255.0
            ).astype(np.uint8)

            normalized_image = Image.fromarray(
                normalized_image
            )

            normalized_image.save(
                output_path,
                quality=95
            )

            total += 1

        except Exception as error:
            print(
                f"ERROR processing {image_path}: {error}"
            )

    return total


if __name__ == "__main__":

    print("=" * 60)
    print("PALM RIPENESS PREDICTION")
    print("IMAGE NORMALIZATION")
    print("=" * 60)

    splits = {
        "train": CLAHE_DIR / "train",
        "valid": CLAHE_DIR / "valid",
        "test": CLAHE_DIR / "test"
    }

    total_images = 0

    for split_name, source_dir in splits.items():

        print()
        print(f"Processing: {split_name}")

        destination_dir = (
            NORMALIZED_DIR / split_name
        )

        count = normalize_images(
            source_dir,
            destination_dir
        )

        print(
            f"Normalized images: {count}"
        )

        total_images += count

    print()
    print("=" * 60)
    print(
        f"TOTAL IMAGES PROCESSED: {total_images}"
    )
    print("=" * 60)