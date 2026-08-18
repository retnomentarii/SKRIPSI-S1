# import
import os
import copy
import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import (
    compute_class_weight
)
import matplotlib.pyplot as plt
from dataset import FootDataset
# m4 device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print("Device:", DEVICE)

# path
SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)
TRAIN_CSV = os.path.join(
    SCRIPT_DIR,
    "csv",
    "train-bismillah_augmented.csv"
)
VAL_CSV = os.path.join(
    SCRIPT_DIR,
    "csv",
    "val-bismillah_split.csv"
)
TEST_CSV = os.path.join(
    SCRIPT_DIR,
    "csv",
    "test-bismillah_split.csv"
)

IMAGE_DIR = os.path.join(
    SCRIPT_DIR,
    "images"
)

OUTPUT_MODEL = os.path.join(
    SCRIPT_DIR,
    "outputs",
    "models",
    "bismillah-best_resnet50-adamW8.pth"
)

# hyperparameter
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
PATIENCE = 8
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# load csv
train_df = pd.read_csv(
    TRAIN_CSV
)
val_df = pd.read_csv(
    VAL_CSV
)
test_df = pd.read_csv(
    TEST_CSV
)
print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

# transform
train_transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

test_transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# Dataset
train_dataset = FootDataset(
    train_df,
    IMAGE_DIR,
    train_transform
)

val_dataset = FootDataset(
    val_df,
    IMAGE_DIR,
    test_transform
)

test_dataset = FootDataset(
    test_df,
    IMAGE_DIR,
    test_transform
)

from collections import Counter

print("\n========== DATASET AFTER AUGMENTATION ==========")
print(f"Training Images : {len(train_dataset)}")

train_labels = train_df["Label"].tolist()
class_count = Counter(train_labels)

print(f"Class 0 (Non-Flatfoot) : {class_count[0]}")
print(f"Class 1 (Flatfoot)     : {class_count[1]}")

# Dataloader
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=g
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# class weight
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["Label"]),
    y=train_df["Label"]
)

print("\n========== CLASS WEIGHT ==========")
for i, w in enumerate(weights):
    class_name = "Non-Flatfoot" if i == 0 else "Flatfoot"
    print(f"{class_name:<15}: {w:.4f}")

weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(DEVICE)

print("\nTensor Class Weight:")
print(weights)

# model resnet50
model = resnet50(
    weights=ResNet50_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(
    DEVICE
)

# loss and optimizer
criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# training loop
best_loss = float("inf")
best_epoch = 0

counter = 0
best_model = None

train_losses = []
val_losses = []

train_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    model.eval()
    running_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
    print(
    f"Epoch [{epoch+1}/{EPOCHS}] "
    f"Train Loss : {epoch_loss:.4f} | "
    f"Validation Loss : {val_loss:.4f}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch + 1
        counter = 0
        best_model = copy.deepcopy(
            model.state_dict()
        )
        torch.save(
            best_model,
            OUTPUT_MODEL
        )
    else:
        counter += 1
        if counter >= PATIENCE:
            print(
                "\nEarly stopping!"
            )
            break
train_time = time.time() - train_start
print("\n==============================")
print("TRAINING SUMMARY")
print("==============================")

print(f"Best Epoch           : {best_epoch}")
print(f"Best Validation Loss : {best_loss:.4f}")
print(f"Training Time        : {train_time:.2f} sec")

# load best model
# ==================================================
# LOAD BEST MODEL
# ==================================================
model.load_state_dict(
    torch.load(
        OUTPUT_MODEL,
        map_location=DEVICE
    )
)

# ==================================================
# TESTING
# ==================================================
model.eval()

y_true = []
y_pred = []
y_score = []

test_start = time.time()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            preds.cpu().numpy()
        )

        y_score.extend(
            probs[:,1].cpu().numpy()
        )

test_time = time.time() - test_start

# ==================================================
# METRICS
# ==================================================
accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

auc = roc_auc_score(
    y_true,
    y_score
)

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "Non-Flatfoot",
        "Flatfoot"
    ]
)

cm = confusion_matrix(
    y_true,
    y_pred
)

MODE_NAME = "ResNet50"

# ==================================================
# PRINT RESULT
# ==================================================
print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)

print("\n")
print("=" * 50)
print("===== TEST RESULTS =====")
print("=" * 50)

print(
    f"Mode          : {MODE_NAME}"
)

print(
    f"Accuracy      : {accuracy:.4f}"
)

print(
    f"Precision     : {precision:.4f}"
)

print(
    f"Recall        : {recall:.4f}"
)

print(
    f"F1-Score      : {f1:.4f}"
)

print(
    f"AUC ROC       : {auc:.4f}"
)

print(
    f"Best Epoch    : {best_epoch}"
)

print(
    f"Best Val Loss : {best_loss:.4f}"
)

print(
    f"Training Time : {train_time:.2f} sec"
)

print(
    f"Testing Time  : {test_time:.2f} sec"
)

print("=" * 50)

# Simpan Hasil
os.makedirs(
    os.path.join(
        SCRIPT_DIR,
        "outputs",
        "reports"
    ),
    exist_ok=True
)

os.makedirs(
    os.path.join(
        SCRIPT_DIR,
        "outputs",
        "figures"
    ),
    exist_ok=True
)

# Save Report txt
with open(
    os.path.join(
        SCRIPT_DIR,
        "outputs",
        "reports",
        "resnet",
        "bismillah_resnet8.txt"
    ),
    "w"
) as f:

    f.write(report)
    f.write("\n")
    f.write("===== TEST RESULTS =====\n")
    f.write(f"Model          : {MODE_NAME}\n")
    f.write(f"Accuracy      : {accuracy:.4f}\n")
    f.write(f"Precision     : {precision:.4f}\n")
    f.write(f"Recall        : {recall:.4f}\n")
    f.write(f"F1-Score      : {f1:.4f}\n")
    f.write(f"AUC ROC       : {auc:.4f}\n")
    f.write(f"Training Time : {train_time:.2f} sec\n")
    f.write(f"Testing Time  : {test_time:.2f} sec\n")
    f.write(f"Best Epoch           : {best_epoch}\n")
    f.write(f"Best Validation Loss : {best_loss:.4f}\n")

# Save confusion matrix png
import seaborn as sns
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    os.path.join(
        SCRIPT_DIR,
        "outputs",
        "figures",
        "resnet",
        "conmat",
        "bismillah_conmat-resnet8.png"
    )
)

plt.close()

# Save ROC curve
fpr, tpr, _ = roc_curve(
    y_true,
    y_score
)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC={auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.savefig(
    os.path.join(
        SCRIPT_DIR,
        "outputs",
        "figures",
        "resnet",
        "rocurve",
        "bismillah_roc_curve-resnet8.png"
    )
)

plt.close()

# ==========================================
# TRAINING & VALIDATION LOSS
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(train_losses)+1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    range(1, len(val_losses)+1),
    val_losses,
    marker="s",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.savefig(
    os.path.join(
        SCRIPT_DIR,
        "outputs",
        "figures",
        "resnet",
        "grafik",
        "bismillah_training_validation_loss-resnet8.png"
    )
)

plt.close()

