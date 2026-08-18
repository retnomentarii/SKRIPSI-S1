# ==================================================
# IMPORT
# ==================================================
import os
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

from sklearn.model_selection import (
    StratifiedKFold,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

from sklearn.utils.class_weight import (
    compute_class_weight
)

from dataset import FootDataset

# ==================================================
# DEVICE
# ==================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)

# ==================================================
# PATH
# ==================================================
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

MODEL_DIR = os.path.join(
    SCRIPT_DIR,
    "outputs",
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)
# ==================================================
# OUTPUT DIRECTORY
# ==================================================
os.makedirs(
    os.path.join(
        SCRIPT_DIR,
        "outputs",
        "reports",
        "resnet"
    ),
    exist_ok=True
)

os.makedirs(
    os.path.join(
        SCRIPT_DIR,
        "outputs",
        "figures",
        "resnet",
        "conmat"
    ),
    exist_ok=True
)

os.makedirs(
    os.path.join(
        SCRIPT_DIR,
        "outputs",
        "figures",
        "resnet",
        "rocurve"
    ),
    exist_ok=True
)

os.makedirs(
    os.path.join(
        SCRIPT_DIR,
        "outputs",
        "figures",
        "resnet",
        "grafik"
    ),
    exist_ok=True
)

# ==================================================
# CONFIG
# ==================================================
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
PATIENCE = 8
N_SPLITS = 5
EXPERIMENT_NAME = "5fold_bismillah_resnet50"
MODEL_NAME = "ResNet50"
# ==================================================

import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# ==================================================
# LOAD DATA
# ==================================================
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

train80_df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Train+Val  : {len(train80_df)}")
print(f"Test       : {len(test_df)}")

test_df = test_df.reset_index(drop=True)

print("\nTrain+Validation")
print(train80_df["Label"].value_counts())
print("\nTest")
print(test_df["Label"].value_counts())

# ==================================================
# TRANSFORM
# ==================================================
train_transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def validate_one_epoch(
    model,
    loader,
    criterion
):

    model.eval()
    running_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(
                outputs,
                labels
            )
            running_loss += loss.item()
    return running_loss / len(loader)

# ==================================================
# OOF RESNET50
# ==================================================
def generate_oof_resnet(
    train80_df
):

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []
    fold_results = []

    best_fold_loss = float("inf")

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(
        skf.split(
            train80_df,
            train80_df["Label"]
        )
    ):

        print("\n" + "=" * 60)
        print(f"FOLD {fold+1}")
        print("=" * 60)

        fold_train = train80_df.iloc[
            train_idx
        ].reset_index(drop=True)

        fold_valid = train80_df.iloc[
            valid_idx
        ].reset_index(drop=True)

        print("\nTrain")
        print(fold_train["Label"].value_counts())

        print("\nValidation")
        print(fold_valid["Label"].value_counts())

        train_dataset = FootDataset(
            fold_train,
            IMAGE_DIR,
            train_transform
        )
        valid_dataset = FootDataset(
            fold_valid,
            IMAGE_DIR,
            val_transform
        )
        g = torch.Generator()
        g.manual_seed(SEED)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=g
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        # ==========================================
        # CLASS WEIGHT
        # ==========================================
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(
                fold_train["Label"]
            ),
            y=fold_train["Label"]
        )
        weights = torch.tensor(
            weights,
            dtype=torch.float32
        ).to(DEVICE)
        # ==========================================
        # MODEL
        # ==========================================
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
        criterion = nn.CrossEntropyLoss(
            weight=weights
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=1e-4
        )
        # ==========================================
        # TRAINING
        # ==========================================
        best_model = None
        best_loss = float("inf")
        counter = 0

        for epoch in range(EPOCHS):

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )

            val_loss = validate_one_epoch(
                model,
                valid_loader,
                criterion
            )

            print(
                f"Fold {fold+1} | "
                f"Epoch [{epoch+1}/{EPOCHS}] | "
                f"Train Loss : {train_loss:.4f} | "
                f"Val Loss : {val_loss:.4f}"
            )

            if val_loss < best_loss:

                best_loss = val_loss

                best_model = copy.deepcopy(
                    model.state_dict()
                )

                counter = 0

            else:

                counter += 1

                if counter >= PATIENCE:

                    print("Early stopping!")

                    break
        # ==========================================
        # LOAD BEST MODEL
        # ==========================================
        model.load_state_dict(
            best_model
        )

        # ==========================================
        # VALIDATION
        # ==========================================
        model.eval()

        y_true = []
        y_pred = []
        y_score = []

        with torch.no_grad():

            for images, labels in valid_loader:

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
                    probs[:, 1].cpu().numpy()
                )

        acc = accuracy_score(
            y_true,
            y_pred
        )

        prec = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        rec = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        auc = roc_auc_score(
            y_true,
            y_score
        )

        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)
        auc_scores.append(auc)
        fold_results.append({
            "fold": fold + 1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc
        })

        if best_loss < best_fold_loss:
            best_fold_loss = best_loss

        print("\nRESULT")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"AUC      : {auc:.4f}")
        print(
            f"\nFold {fold+1} Summary | "
            f"Acc={acc:.4f} | "
            f"Prec={prec:.4f} | "
            f"Rec={rec:.4f} | "
            f"F1={f1:.4f} | "
            f"AUC={auc:.4f}"
        )
    
    return (
        accuracy_scores,
        precision_scores,
        recall_scores,
        f1_scores,
        auc_scores,
        fold_results,
        best_fold_loss
    )

# ==================================================
# FINAL RESNET50 TRAINING
# ==================================================
def train_final_resnet(
    train80_df
):
    dataset = FootDataset(
        train80_df,
        IMAGE_DIR,
        train_transform
    )
    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(
            train80_df["Label"]
        ),
        y=train80_df["Label"]
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    ).to(DEVICE)

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

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )
    best_model = None
    best_loss = np.inf
    train_losses = []

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(
            model,
            loader,
            criterion,
            optimizer
        )

        train_losses.append(
            train_loss
        )
        if train_loss < best_loss:
            best_loss = train_loss

            best_model = copy.deepcopy(
                model.state_dict()
            )
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss : {train_loss:.4f}"
        )
    model.load_state_dict(
        best_model
    )
    torch.save(
        model.state_dict(),
        os.path.join(
            MODEL_DIR,
            f"{EXPERIMENT_NAME}.pth"
        )
    )

    return (
        model,
        train_losses
    )

# ==================================================
# TEST RESNET50 PREDICTION
# ==================================================
def predict_test_resnet(
    model,
    test_df
):

    dataset = FootDataset(
        test_df,
        IMAGE_DIR,
        val_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model.eval()

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():

        for images, labels in loader:

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
                probs[:, 1].cpu().numpy()
            )

    return (
        y_true,
        y_pred,
        y_score
    )

# ==================================================
# SUMMARY
# ==================================================
def summarize(
    metric_name,
    values
):

    values = np.array(
        values
    )

    print("\n" + "=" * 60)
    print(metric_name)
    print("=" * 60)

    print(
        f"Mean    : {values.mean():.4f}"
    )

    print(
        f"Std Dev : {values.std():.4f}"
    )

    print(
        f"Min     : {values.min():.4f}"
    )

    print(
        f"Median  : {np.median(values):.4f}"
    )

    print(
        f"Max     : {values.max():.4f}"
    )

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":

    train_start = time.time()

    (
        accuracy_scores,
        precision_scores,
        recall_scores,
        f1_scores,
        auc_scores,
        fold_results,
        best_fold_loss
    ) = generate_oof_resnet(train80_df)

    print("\n")
    print("=" * 60)
    print("CROSS VALIDATION SUMMARY")
    print("=" * 60)

    summarize(
        "Accuracy",
        accuracy_scores
    )

    summarize(
        "Precision",
        precision_scores
    )

    summarize(
        "Recall",
        recall_scores
    )

    summarize(
        "F1 Score",
        f1_scores
    )

    summarize(
        "AUC",
        auc_scores
    )

    print("\nTraining Final ResNet50...")

    final_model, train_losses = train_final_resnet(
        train80_df
    )

    train_time = time.time() - train_start
    print("\nGenerating Test Prediction...")
    test_start = time.time()
    y_true, y_pred, y_score = predict_test_resnet(
        final_model,
        test_df
    )
    test_time = time.time() - test_start
    # ==========================================
    # METRICS
    # ==========================================
    accuracy = accuracy_score(
        y_true,
        y_pred
    )
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
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
        ],
        zero_division=0
    )
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n===== FINAL TEST RESULT =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("\n")
    print(report)

    print("\n")
    print("=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)

    print(f"Model          : {MODEL_NAME}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"AUC ROC        : {auc:.4f}")
    print(f"Training Time  : {train_time:.2f} sec")
    print(f"Testing Time   : {test_time:.2f} sec")
    print(f"Epochs         : {EPOCHS}")
    print(f"Final Loss     : {train_losses[-1]:.4f}")

    report_path = os.path.join(
        SCRIPT_DIR,
        "outputs",
        "reports",
        "resnet",
        f"{EXPERIMENT_NAME}.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(report)

        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("CROSS VALIDATION SUMMARY\n")
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("FOLD RESULTS\n")
        f.write("=" * 60 + "\n")

        for result in fold_results:
            f.write(
                f"Fold {result['fold']} | "
                f"Acc={result['accuracy']:.4f} | "
                f"Prec={result['precision']:.4f} | "
                f"Rec={result['recall']:.4f} | "
                f"F1={result['f1']:.4f} | "
                f"AUC={result['auc']:.4f}\n"
            )

        f.write("\n")
        f.write("=" * 60 + "\n")

        f.write(f"Accuracy Mean  : {np.mean(accuracy_scores):.4f}\n")
        f.write(f"Accuracy Std   : {np.std(accuracy_scores):.4f}\n")

        f.write(f"Precision Mean : {np.mean(precision_scores):.4f}\n")
        f.write(f"Precision Std  : {np.std(precision_scores):.4f}\n")

        f.write(f"Recall Mean    : {np.mean(recall_scores):.4f}\n")
        f.write(f"Recall Std     : {np.std(recall_scores):.4f}\n")

        f.write(f"F1 Mean        : {np.mean(f1_scores):.4f}\n")
        f.write(f"F1 Std         : {np.std(f1_scores):.4f}\n")

        f.write(f"AUC Mean       : {np.mean(auc_scores):.4f}\n")
        f.write(f"AUC Std        : {np.std(auc_scores):.4f}\n")

        f.write("\n")

        f.write("="*60 + "\n")

        f.write("FINAL TEST SUMMARY\n")

        f.write("="*60 + "\n")

        f.write(f"Model          : {MODEL_NAME}\n")
        f.write(f"Accuracy       : {accuracy:.4f}\n")
        f.write(f"Precision      : {precision:.4f}\n")
        f.write(f"Recall         : {recall:.4f}\n")
        f.write(f"F1-Score       : {f1:.4f}\n")
        f.write(f"AUC ROC        : {auc:.4f}\n")
        f.write(f"Training Time  : {train_time:.2f} sec\n")
        f.write(f"Testing Time   : {test_time:.2f} sec\n")
        f.write(f"Epochs         : {EPOCHS}\n")
        f.write(f"Final Loss     : {train_losses[-1]:.4f}\n")

    # Save confusion matrix png
    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Flatfoot", "Flatfoot"],
        yticklabels=["Non-Flatfoot", "Flatfoot"]
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
            f"{EXPERIMENT_NAME}_conmat.png"
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
    plt.title("ROC Curve")
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
            f"{EXPERIMENT_NAME}_roc_curve.png"
        )
    )

    plt.close()

    # ==========================================
    # TRAINING LOSS
    # ==========================================
    plt.figure(figsize=(8,5))

    plt.plot(
        range(1, len(train_losses)+1),
        train_losses,
        marker="o",
        label="Training Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.savefig(
        os.path.join(
            SCRIPT_DIR,
            "outputs",
            "figures",
            "resnet",
            "grafik",
            f"{EXPERIMENT_NAME}_training_loss.png"
        )
    )

    plt.close()
