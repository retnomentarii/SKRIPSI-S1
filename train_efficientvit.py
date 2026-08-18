# import
from collections import Counter
import os
from typing import Counter
import time
from sklearn.metrics import accuracy_score
import copy
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.utils.class_weight import (
    compute_class_weight
)

import matplotlib.pyplot as plt

from dataset import FootDataset


# m4 device
if torch.backends.mps.is_available():

    DEVICE = torch.device("cpu")

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
    "bismillah_best_efficientvit_m0-2.pth"
)

# hyperparameter
BATCH_SIZE = 32

LR = 1e-4

EPOCHS = 50

PATIENCE = 10

SEED = 42
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
val_df = pd.read_csv(VAL_CSV)
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

# model efficientvit_m0
model = timm.create_model(
    "efficientvit_m0",
    pretrained=True,
    num_classes=2
)

model = model.to(
    DEVICE
)

# loss and optimizer
criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

# training loop
start_time = time.time()
best_loss = float("inf")
best_epoch = 0
counter = 0
best_model = None
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    epoch_start = time.time()
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

    epoch_time = time.time() - epoch_start

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
    f"Validation Loss : {val_loss:.4f} | "
    f"Time : {epoch_time:.2f}s"
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
training_time = time.time() - start_time

print("\n==============================")
print("TRAINING SUMMARY")
print("==============================")

print(f"Best Epoch           : {best_epoch}")
print(f"Best Validation Loss : {best_loss:.4f}")
print(f"Training Time        : {training_time:.2f} sec")

# load best model
model.load_state_dict(
    torch.load(
        OUTPUT_MODEL,
        map_location=DEVICE
    )
)

# evaluasi test set
test_start = time.time()
model.eval()

y_true = []
y_pred = []
y_score = []

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

# classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "Normal",
        "Flatfoot"
    ]
)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
# confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred
)
acc = accuracy_score(
    y_true,
    y_pred
)
test_time = time.time() - test_start
# ROC AUC
auc = roc_auc_score(
    y_true,
    y_score
)

print(report)

print("="*50)
print("\nConfusion Matrix:")
print(cm)

print("\n" + "="*50)
print("===== TEST RESULTS =====")
print("="*50)

print(f"Model          : EfficientViT-M0")
print(f"Accuracy       : {acc:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")
print(f"AUC ROC        : {auc:.4f}")
print(f"Best Epoch     : {best_epoch}")
print(f"Best Val Loss  : {best_loss:.4f}")
print(f"Training Time  : {training_time:.2f} sec")
print(f"Testing Time   : {test_time:.2f} sec")

print("="*50)

# Full report string (include accuracy, AUC, and test time)
report_full = (
    report
    + f"\n\nAccuracy             = {acc:.4f}"
    + f"\nPrecision            = {precision:.4f}"
    + f"\nRecall               = {recall:.4f}"
    + f"\nF1-Score             = {f1:.4f}"
    + f"\nAUC                  = {auc:.4f}"
    + f"\nTraining Time        = {training_time:.2f} sec"
    + f"\nTesting Time         = {test_time:.2f} sec"
    + f"\nBest Epoch           = {best_epoch}"
    + f"\nBest Validation Loss = {best_loss:.4f}"
)

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
        "efficientvit",
        "bismillah_efficientvit_cr3.txt"
    ),
    "w"
) as f:

    f.write(report_full)

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
        "efficientvit",
        "conmat",
        "bismillah_efficientvit_conmat3.png"
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
        "efficientvit",
        "rocurve",
        "bismillah_efficientvit_roc3.png"
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
        "efficientvit",
        "grafik",
        "bismillah_training_validation_loss-efficientvit3.png"
    )
)

plt.close()