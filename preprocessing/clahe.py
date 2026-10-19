from pathlib import Path
import cv2


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESIZED_DIR = PROJECT_ROOT / "dataset" / "processed" / "resized"

CLAHE_DIR = PROJECT_ROOT / "dataset" / "processed" / "clahe"


# --------------------------------------------------
# IMAGE SETTINGS
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# --------------------------------------------------
# CLAHE SETTINGS
# --------------------------------------------------

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# --------------------------------------------------
# PROCESS IMAGES
# --------------------------------------------------

def apply_clahe(source_dir, destination_dir):

    total = 0

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

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

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                print(
                    f"WARNING: Could not read {image_path}"
                )
                continue

            # Convert BGR → LAB
            lab_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2LAB
            )

            # Split LAB channels
            l_channel, a_channel, b_channel = cv2.split(
                lab_image
            )

            # Apply CLAHE only to luminance
            enhanced_l = clahe.apply(
                l_channel
            )

            # Merge channels
            enhanced_lab = cv2.merge(
                [
                    enhanced_l,
                    a_channel,
                    b_channel
                ]
            )

            # Convert LAB → BGR
            enhanced_image = cv2.cvtColor(
                enhanced_lab,
                cv2.COLOR_LAB2BGR
            )

            # Save image
            cv2.imwrite(
                str(output_path),
                enhanced_image
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
    print("CLAHE CONTRAST ENHANCEMENT")
    print("=" * 60)

    splits = {
        "train": RESIZED_DIR / "train",
        "valid": RESIZED_DIR / "valid",
        "test": RESIZED_DIR / "test"
    }

    total_images = 0

    for split_name, source_dir in splits.items():

        print()
        print(f"Processing: {split_name}")

        destination_dir = (
            CLAHE_DIR / split_name
        )

        count = apply_clahe(
            source_dir,
            destination_dir
        )

        print(
            f"CLAHE images: {count}"
        )

        total_images += count

    print()
    print("=" * 60)
    print(
        f"TOTAL IMAGES PROCESSED: {total_images}"
    )
    print("=" * 60)