import os
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CSV_PATH = os.path.join(
    SCRIPT_DIR,
    "csv",
    "clean_label-bismillah.csv"
)

df = pd.read_csv(CSV_PATH)

# =====================================
# Split 80% Train+Validation
# 20% Test
# =====================================
train_val_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Label"],
    random_state=7
)

# =====================================
# Split Train menjadi
# 80% Train
# 20% Validation
# =====================================
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.2,
    stratify=train_val_df["Label"],
    random_state=7
)

TRAIN_CSV = os.path.join(
    SCRIPT_DIR,
    "csv",
    "train-bismillah_split.csv"
)

VAL_CSV = os.path.join(
    SCRIPT_DIR,
    "csv",
    "val-bismillah_split.csv"
)

TEST_CSV = os.path.join(
    SCRIPT_DIR,
    "csv",
    "test-bismillah_split1.csv"
)

train_df.to_csv(
    TRAIN_CSV,
    index=False
)

val_df.to_csv(
    VAL_CSV,
    index=False
)

test_df.to_csv(
    TEST_CSV,
    index=False
)

print("Train      :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))

print("\nTrain Distribution")
print(train_df["Klasifikasi"].value_counts())

print("\nValidation Distribution")
print(val_df["Klasifikasi"].value_counts())

print("\nTest Distribution")
print(test_df["Klasifikasi"].value_counts())