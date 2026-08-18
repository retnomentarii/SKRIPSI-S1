# import libraries
# ==================================================
# IMPORT
# ==================================================
import os
import time
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader
)
from torchvision import transforms
from torchvision.models import (
    vgg16,
    VGG16_Weights
)
from sklearn.preprocessing import (
    StandardScaler
)
from sklearn.neighbors import (
    KNeighborsClassifier
)
from sklearn.linear_model import (
    LogisticRegression
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# DEVICE
# ==================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device(
        "mps"
    )
elif torch.cuda.is_available():
    DEVICE = torch.device(
        "cuda"
    )
else:
    DEVICE = torch.device(
        "cpu"
    )
print(
    "Device:",
    DEVICE
)

# ==================================================
# CONFIG
# ==================================================
MODE_NAME = "Stacking_VGG16_kNN"
BATCH_SIZE = 16
FINAL_EPOCHS = 50
LR = 1e-4
PATIENCE = 10
N_SPLITS = 5
KNN_NEIGHBORS = 5
IMG_SIZE = 224

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
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "outputs"
)
REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)
FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "figures"
)
MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

# ==================================================
# OUTPUT FOLDER
# ==================================================
os.makedirs(
    REPORT_DIR,
    exist_ok=True
)
os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)
os.makedirs(
    MODEL_DIR,
    exist_ok=True
)
os.makedirs(
    os.path.join(
        REPORT_DIR,
        "stacking"
    ),
    exist_ok=True
)
os.makedirs(
    os.path.join(
        FIGURE_DIR,
        "stacking",
        "conmat"
    ),
    exist_ok=True
)
os.makedirs(
    os.path.join(
        FIGURE_DIR,
        "stacking",
        "rocurve"
    ),
    exist_ok=True
)
os.makedirs(
    os.path.join(
        FIGURE_DIR,
        "stacking",
        "grafik"
    ),
    exist_ok=True
)

# ==================================================
# LOAD CSV dan TRAIN TEST SPLIT
# ==================================================
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)


print(f"\nTrain      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")
print(
    "\nMissing Values Train:"
)
print(
    train_df[["L","Q","R"]].isna().sum()
)
print(
    "\nMissing Values Validation:"
)
print(
    val_df[["L","Q","R"]].isna().sum()
)
print(
    "\nMissing Values Test:"
)
print(
    test_df[["L","Q","R"]].isna().sum()
)
print("\nTrain")
print(train_df["Label"].value_counts())

print("\nValidation")
print(val_df["Label"].value_counts())

print("\nTest")
print(test_df["Label"].value_counts())
# ==================================================
# TRANSFORM
# ==================================================
transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

# ==================================================
# DATASET
# ==================================================
class FootDataset(
    Dataset
):
    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None
    ):
        self.df = dataframe.reset_index(
            drop=True
        )
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(
            self.df
        )
    def __getitem__(
        self,
        idx
    ):
        row = self.df.iloc[idx]
        img_path = os.path.join(
            self.image_dir,
            row["filename"]
        )

        image = Image.open(
            img_path
        ).convert(
            "RGB"
        )

        label = int(
            row["Label"]
        )

        if self.transform:
            image = self.transform(
                image
            )
        return image, label
    

# ==================================================
# VGG16 MODEL
# ==================================================
class VGG16Binary(
    nn.Module
):
    def __init__(
        self
    ):
        super().__init__()
        self.backbone = vgg16(
            weights=VGG16_Weights.DEFAULT
        )
        in_features = self.backbone.classifier[
            6
        ].in_features

        self.backbone.classifier[
            6
        ] = nn.Linear(

            in_features,
            1
        )

    def forward(
        self,
        x
    ):
        return self.backbone(
            x
        )
    
# ==================================================
# EARLY STOPPING
# ==================================================
class EarlyStopping:
    def __init__(
        self,
        patience=5
    ):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.stop = False
    def __call__(
        self,
        val_loss
    ):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True

# ==================================================
# LOSS
# ==================================================
criterion = nn.BCEWithLogitsLoss()

# ==================================================
# TRAIN ONE EPOCH
# ==================================================
def train_one_epoch(
    model,
    loader,
    optimizer
):
    model.train()
    running_loss = 0
    for images, labels in loader:
        images = images.to(
            DEVICE
        )

        labels = labels.float().view(
            -1,
            1
        ).to(
            DEVICE
        )
        optimizer.zero_grad()
        outputs = model(
            images
        )
        loss = criterion(
            outputs,
            labels
        )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# ==================================================
# VALIDATE
# ==================================================
def validate_one_epoch(
    model,
    loader
):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(
                DEVICE
            )
            labels = labels.float().view(
                -1,
                1
            ).to(
                DEVICE
            )
            outputs = model(
                images
            )
            loss = criterion(
                outputs,
                labels
            )
            running_loss += loss.item()
    return running_loss / len(loader)

# ==================================================
# VGG16 PROBABILITY
# ==================================================
def predict_proba_vgg(
    model,
    loader
):
    model.eval()
    probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(
                DEVICE
            )
            outputs = model(
                images
            )
            outputs = torch.sigmoid(
                outputs
            )
            probs.extend(
                outputs.cpu()
                .numpy()
                .flatten()
            )
    return np.array(probs)
# ==================================================
# FINAL VGG16 TRAINING
# ==================================================
def train_final_vgg(
    train_df,
    val_df
):
    dataset = FootDataset(
        train_df,
        IMAGE_DIR,
        transform
    )
    val_dataset = FootDataset(
        val_df,
        IMAGE_DIR,
        transform
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    model = VGG16Binary().to(
        DEVICE
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )
    best_loss = np.inf
    best_epoch = 0

    train_losses = []
    val_losses = []

    best_model = copy.deepcopy(
        model.state_dict()
    )
    patience_counter = 0
    for epoch in range(
        FINAL_EPOCHS
    ):
        train_loss = train_one_epoch(
            model,
            loader,
            optimizer
        )
        val_loss = validate_one_epoch(
            model,
            val_loader
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{FINAL_EPOCHS}] "
            f"Train Loss : {train_loss:.4f} | "
            f"Validation Loss : {val_loss:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1

            best_model = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                "Final VGG Early Stopping"
            )
            break

    model.load_state_dict(
        best_model
    )

    torch.save(
        model.state_dict(),
        os.path.join(
            MODEL_DIR,
            "best_vgg16.pth"
        )
    )
    return (
        model,
        best_loss,
        best_epoch,
        train_losses,
        val_losses
    )

# ==================================================
# FINAL KNN
# ==================================================
def train_final_knn(
    train_df
):
    X_train = train_df[
        [
            "L",
            "Q",
            "R"
        ]
    ].values

    y_train = train_df[
        "Label"
    ].values
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )
    knn = KNeighborsClassifier(
        n_neighbors=KNN_NEIGHBORS
    )
    knn.fit(
        X_train,
        y_train
    )
    with open(
        os.path.join(
            MODEL_DIR,
            "knn.pkl"
        ),
        "wb"
    ) as f:
        pickle.dump(
            knn,
            f
        )
    with open(
        os.path.join(
            MODEL_DIR,
            "scaler.pkl"
        ),
        "wb"
    ) as f:
        pickle.dump(
            scaler,
            f
        )
    return (
            knn,
            scaler
        )

# ==================================================
# TEST VGG PROBABILITY
# ==================================================
def predict_test_vgg(
    model,
    test_df
):
    dataset = FootDataset(
        test_df,
        IMAGE_DIR,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    probs = predict_proba_vgg(
        model,
        loader
    )
    return probs

# ==================================================
# TEST KNN PROBABILITY
# ==================================================
def predict_test_knn(
    knn,
    scaler,
    test_df
):

    X_test = test_df[
        [
            "L",
            "Q",
            "R"
        ]

    ].values
    X_test = scaler.transform(
        X_test
    )
    probs = knn.predict_proba(
        X_test
    )[:,1]
    return probs

# ==================================================
# VGG16
# ==================================================
if __name__ == "__main__":
    train_start = time.time()
    print("\nTraining Final VGG16...")
    final_vgg, final_loss, best_epoch, train_losses, val_losses = train_final_vgg(
        train_df,
        val_df
    )
    print("\nTraining Final kNN...")
    final_knn, scaler = train_final_knn(
        train_df
    )
    val_vgg_prob = predict_test_vgg(
        final_vgg,
        val_df
    )
    val_knn_prob = predict_test_knn(
        final_knn,
        scaler,
        val_df
    )
    X_meta_train = np.column_stack([
        val_vgg_prob,
        val_knn_prob
    ])
    y_meta_train = val_df["Label"].values
    meta_model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    meta_model.fit(
        X_meta_train,
        y_meta_train
    )
    # -------------------------
    # Save Meta Learner
    # -------------------------

    with open(
        os.path.join(
            MODEL_DIR,
            "meta_lr.pkl"
        ),
        "wb"
    ) as f:
        pickle.dump(
            meta_model,
            f
        )
    print("\n==============================")
    print("TRAINING SUMMARY")
    print("==============================")

    print(f"Best Epoch           : {best_epoch}")
    print(f"Best Validation Loss : {final_loss:.4f}")
    print("==============================")
    
    print(
    "\nGenerating Test Probabilities..."
    )
    train_time = (
        time.time() - train_start
    )

    test_start = time.time()
    test_vgg_prob = predict_test_vgg(
        final_vgg,
        test_df
    )
    test_knn_prob = predict_test_knn(
        final_knn,
        scaler,
        test_df
    )
    X_meta_test = np.column_stack([
        test_vgg_prob,
        test_knn_prob
    ])
    print(
        "\nMeta Test Shape:",
        X_meta_test.shape
    )
    y_pred = meta_model.predict(
        X_meta_test
    )
    y_score = meta_model.predict_proba(
        X_meta_test
    )[:,1]
    y_test = test_df[
        "Label"
    ].values
    print(
        "\nPrediction Complete"
    )
    print(
        "Total Test:",
        len(y_test)
    )
    test_time = (
        time.time() - test_start
    )

    # ==================================================
    # METRICS
    # ==================================================
    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    auc = roc_auc_score(
        y_test,
        y_score
    )
    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Flatfoot"
        ]
    )
    cm = confusion_matrix(
        y_test,
        y_pred
    )
    print("\n===== TEST RESULTS =====")
    print(
        "Accuracy:",
        round(accuracy,4)
    )
    print(
        "Precision:",
        round(precision,4)
    )
    print(
        "Recall:",
        round(recall,4)
    )
    print(
        "F1:",
        round(f1,4)
    )
    print(
        "AUC:",
        round(auc,4)
    )
    print("\n")
    print(report)
    # ==================================================
    # SAVE REPORT
    # ==================================================
    report_path = os.path.join(
        REPORT_DIR,
        "stacking",
        "bismillah_stacking_report2.txt"
    )
    os.makedirs(
        os.path.dirname(report_path),
        exist_ok=True
    )
    with open(
        report_path,
        "w"
    ) as f:
        f.write(report)
        f.write("\n")
        f.write(
            "===== TEST RESULTS =====\n"
        )
        f.write(
            f"Mode                 : {MODE_NAME}\n"
        )
        f.write(
            f"Accuracy             : {accuracy:.4f}\n"
        )
        f.write(
            f"Precision            : {precision:.4f}\n"
        )
        f.write(
            f"Recall               : {recall:.4f}\n"
        )
        f.write(
            f"F1-Score             : {f1:.4f}\n"
        )
        f.write(
            f"AUC ROC              : {auc:.4f}\n"
        )
        f.write(
            f"Training Time        : {train_time:.2f} sec\n"
        )
        f.write(
            f"Testing Time         : {test_time:.2f} sec\n"
        )
        f.write(
            f"Best Epoch           : {best_epoch}\n"
        )
        f.write(
            f"Best Validation Loss : {final_loss:.4f}\n"
        )
    print(f"Training Time        : {train_time:.2f} sec")   
    # ==================================================
    # CONFUSION MATRIX
    # ==================================================
    plt.figure(
        figsize=(6,5)
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        MODE_NAME
    )

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "stacking",
            "conmat",
            "bismillah_stacking_cm2.png"
        )
    )

    plt.close()
    
    # ==================================================
    # ROC CURVE
    # ==================================================
    fpr, tpr, _ = roc_curve(
        y_test,
        y_score
    )
    plt.figure(
        figsize=(6,5)
    )
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
    plt.xlabel(
        "False Positive Rate"
    )
    plt.ylabel(
        "True Positive Rate"
    )
    plt.legend()
    plt.title(
        MODE_NAME
    )
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "stacking",
            "rocurve",
            "bismillah_stacking_roc2.png"
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
        linewidth=2,
        label="Training Loss"
    )

    plt.plot(
        range(1, len(val_losses)+1),
        val_losses,
        marker="s",
        linewidth=2,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "stacking",
            "grafik",
            "bismillah_stacking_loss.png"
        )
    )

    plt.close()

    print(
    "\nFinished Successfully"
    )

    print(
        f"\nReport Saved: {report_path}"
    )

    print(
        "\nTest VGG Shape:",
        test_vgg_prob.shape
    )
    print(
        "Test KNN Shape:",
        test_knn_prob.shape
    )
    print(
        "\nMeta Learner Trained"
    )

