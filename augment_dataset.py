import os
import pandas as pd
from PIL import Image

# =====================================================
# PATH
# =====================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_CSV = os.path.join(
    SCRIPT_DIR,
    "csv",
    "train-bismillah_split.csv"
)

OUTPUT_CSV = os.path.join(
    SCRIPT_DIR,
    "csv",
    "train-bismillah_augmented.csv"
)

# =====================================================
# LOAD DATASET
# =====================================================

train_df = pd.read_csv(TRAIN_CSV)

print("=" * 60)
print("DATASET BEFORE AUGMENTATION")
print("=" * 60)

print(train_df["Klasifikasi"].value_counts())
print(f"\nTotal Images : {len(train_df)}")

# =====================================================
# AUGMENT ONLY FLATFOOT
# =====================================================

augmented_rows = []

flatfoot_df = train_df[
    train_df["Label"] == 1
]

print(f"\nFlatfoot Images : {len(flatfoot_df)}")

for _, row in flatfoot_df.iterrows():

    image_path = row["image_path"]

    if not os.path.exists(image_path):
        print(f"Image not found : {image_path}")
        continue

    image = Image.open(image_path).convert("RGB")

    # Horizontal Flip
    flip_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Nama file baru
    filename = row["filename"]
    name, ext = os.path.splitext(filename)

    new_filename = f"{name}_flip{ext}"

    # Lokasi penyimpanan gambar baru
    new_image_path = os.path.join(
        os.path.dirname(image_path),
        new_filename
    )

    # Simpan gambar hasil flip
    flip_image.save(new_image_path)

    # Copy seluruh metadata
    new_row = row.copy()

    new_row["filename"] = new_filename
    new_row["image_path"] = new_image_path

    augmented_rows.append(new_row)

# =====================================================
# COMBINE DATA
# =====================================================

augmented_df = pd.DataFrame(augmented_rows)

new_train_df = pd.concat(
    [train_df, augmented_df],
    ignore_index=True
)

# =====================================================
# SAVE CSV
# =====================================================

new_train_df.to_csv(
    OUTPUT_CSV,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("=" * 60)
print("DATASET AFTER AUGMENTATION")
print("=" * 60)

print(new_train_df["Klasifikasi"].value_counts())

print(f"\nTotal Images      : {len(new_train_df)}")
print(f"Augmented Images  : {len(augmented_rows)}")

print("\nAugmented CSV saved at:")
print(OUTPUT_CSV)