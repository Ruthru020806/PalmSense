# ============================================================
# PALM RIPENESS PREDICTION
# MODEL EVALUATION
# ConvNeXt-FPN + Transformer
# ============================================================

import sys
import os

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from models.convnext_fpn import ConvNeXtFPNTransformer


# ============================================================
# PATHS
# ============================================================

TEST_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "processed",
    "augmented",
    "test"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "training",
    "best_model.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 4

NUM_CLASSES = 4


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("PALM RIPENESS PREDICTION")
print("MODEL EVALUATION")
print("=" * 60)

print()
print("Device:", device)


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

print()
print("Loading test dataset...")


test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


print()
print("Test images:", len(test_dataset))

print()
print("Classes:")
print(test_dataset.class_to_idx)


# ============================================================
# TEST DATA LOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("Creating model...")


model = ConvNeXtFPNTransformer(
    num_classes=NUM_CLASSES
)


model = model.to(device)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print()
print("Loading trained model...")


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# ------------------------------------------------------------
# LOAD MODEL WEIGHTS
# ------------------------------------------------------------

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ------------------------------------------------------------
# CHECKPOINT INFORMATION
# ------------------------------------------------------------

best_epoch = checkpoint.get(
    "epoch",
    "Unknown"
)

best_validation_accuracy = checkpoint.get(
    "validation_accuracy",
    "Unknown"
)


print()
print("Model loaded successfully.")

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation accuracy:",
    best_validation_accuracy
)


# ============================================================
# EVALUATION
# ============================================================

model.eval()


all_predictions = []

all_labels = []


correct = 0

total = 0


print()
print("Running evaluation...")


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        outputs = model(
            images
        )


        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        _, predictions = torch.max(
            outputs,
            1
        )


        # ----------------------------------------------------
        # ACCURACY
        # ----------------------------------------------------

        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# TEST ACCURACY
# ============================================================

test_accuracy = (
    correct / total
) * 100


print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print()
print(
    f"Correct predictions: {correct}/{total}"
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Over-ripe",
    "Ripe",
    "Under-ripe",
    "Unripe"
]


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    digits=4,
    zero_division=0
)


print()
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)


cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)


print()

print(
    "                 "
    "Over-ripe  Ripe  Under-ripe  Unripe"
)


for index, row in enumerate(cm):

    print(
        f"{class_names[index]:12s}",
        end=""
    )

    for value in row:

        print(
            f"{value:10d}",
            end=""
        )

    print()


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print()
print("=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

print()


for index, class_name in enumerate(
    class_names
):

    class_total = cm[index].sum()

    class_correct = cm[index, index]


    if class_total > 0:

        class_accuracy = (
            class_correct /
            class_total
        ) * 100

    else:

        class_accuracy = 0.0


    print(
        f"{class_name:12s}: "
        f"{class_accuracy:.2f}% "
        f"({class_correct}/{class_total})"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print()
print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy}%"
)

print(
    f"Best Epoch: "
    f"{best_epoch}"
)

print()
print(
    "Model:"
)

print(
    MODEL_PATH
)

print()
print("=" * 60)