import os
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# =====================================
# PARAMETER
# =====================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDER_PATH = os.path.join(
    SCRIPT_DIR,
    "images"
)
OUTPUT_CSV = os.path.join(
    SCRIPT_DIR,
    "label.csv"
)

MIN_AREA = 1500
MARGIN = 20
TOE_REMOVAL_PERCENT = 0.15

# =====================================
# FUNGSI PROCESS IMAGE
# =====================================
def process_image(image_path):

    try:

        # =========================
        # LOAD IMAGE
        # =========================
        img = cv2.imread(image_path)

        if img is None:
            print(f"Gagal membaca: {image_path}")
            return None

        # =========================
        # THRESHOLD GREEN CHANNEL
        # =========================
        green = img[:, :, 1]

        blur = cv2.GaussianBlur(
            green,
            (5, 5),
            0
        )

        _, binary = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Pastikan footprint putih
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        # =========================
        # MORPHOLOGY
        # =========================
        kernel = np.ones((5, 5), np.uint8)

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        # =========================
        # REMOVE SMALL OBJECTS
        # =========================
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        binary_clean = np.zeros_like(binary)

        for i in range(1, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area > MIN_AREA:
                binary_clean[labels == i] = 255

        binary = binary_clean

        # =========================
        # FILL HOLE
        # =========================
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        mask = np.zeros_like(binary)

        cv2.drawContours(
            mask,
            contours,
            -1,
            255,
            thickness=cv2.FILLED
        )

        binary = mask

        # =========================
        # CROP FOOTPRINT
        # =========================
        coords = np.column_stack(
            np.where(binary > 0)
        )

        if len(coords) == 0:
           return {
             "status": "Failed",
             "error": "No footprint detected"
            }

        y_top, x_left = coords.min(axis=0)
        y_bottom, x_right = coords.max(axis=0)

        y_top = max(0, y_top - MARGIN)
        x_left = max(0, x_left - MARGIN)

        y_bottom = min(
            binary.shape[0] - 1,
            y_bottom + MARGIN
        )

        x_right = min(
            binary.shape[1] - 1,
            x_right + MARGIN
        )

        binary = binary[
            y_top:y_bottom + 1,
            x_left:x_right + 1
        ]

        # =========================
        # PCA ALIGNMENT
        # =========================
        coords = np.column_stack(
            np.where(binary > 0)
        )

        if len(coords) < 10:
            return {
                "status": "Failed",
                "error": "Not enough points for PCA"
            }

        pca = PCA(n_components=2)
        pca.fit(coords)

        angle = np.arctan2(
            pca.components_[0, 1],
            pca.components_[0, 0]
        )

        angle_deg = np.degrees(angle)

        h, w = binary.shape

        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(
            center,
            angle_deg,
            1.0
        )

        rotated = cv2.warpAffine(
            binary,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST
        )

        # =========================
        # CROP SETELAH ROTASI
        # =========================
        coords = np.column_stack(
            np.where(rotated > 0)
        )

        if len(coords) == 0:
            return {
                "status": "Failed",
                "error": "No footprint detected after rotation" 
            }

        y_top, x_left = coords.min(axis=0)
        y_bottom, x_right = coords.max(axis=0)

        foot = rotated[
            y_top:y_bottom + 1,
            x_left:x_right + 1
        ]

        # =========================
        # REMOVE TOE
        # =========================
        cut = int(
            TOE_REMOVAL_PERCENT *
            foot.shape[0]
        )

        foot = foot[cut:, :]

        # =========================
        # HITUNG L
        # =========================
        coords = np.column_stack(
            np.where(foot > 0)
        )

        if len(coords) == 0:
            return {
                "status": "Failed",
                "error": "No footprint detected after toe removal"
            }

        y_top = coords[:, 0].min()
        y_bottom = coords[:, 0].max()

        L = y_bottom - y_top

        # =========================
        # HITUNG Q DAN R
        # =========================
        y_midfoot = int(
            y_top + (L / 2)
        )

        y_heel = int(
            y_top + (5 * L / 6)
        )

        WINDOW = 5

        mid_region = foot[
            max(0, y_midfoot-WINDOW):
            min(foot.shape[0], y_midfoot+WINDOW+1),
            :
        ]

        heel_region = foot[
            max(0, y_heel-WINDOW):
            min(foot.shape[0], y_heel+WINDOW+1),
            :
        ]

        Q = np.max(
            np.sum(mid_region > 0, axis=1)
        )

        R = np.max(
            np.sum(heel_region > 0, axis=1)
        )

        if R == 0:
            return {
                "status": "Failed",
                "error": "R is zero, cannot compute Staheli Index"
            }

        SPAI = Q / R

        # =========================
        # KLASIFIKASI
        # =========================
        if SPAI >= 0.9:
            label = "Flatfoot"
        else:
            label = "Normal"

        return {
            "status": "Success",
            "filename": os.path.basename(image_path),
            "image_path": image_path,
            "L": int(L),
            "Q": int(Q),
            "R": int(R),
            "Staheli_Index": round(SPAI, 4),
            "Klasifikasi": label
        }

    except Exception as e:
        return {
            "status": "Failed",
            "error": str(e)
        }


# =====================================
# LOOP SEMUA GAMBAR
# =====================================
results = []
failed_results = []

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

files = sorted(os.listdir(FOLDER_PATH))

total = len(files)
counter = 0

for filename in files:

    if filename.lower().endswith(
        image_extensions
    ):

        counter += 1

        image_path = os.path.join(
            FOLDER_PATH,
            filename
        )

        result = process_image(
            image_path
        )

        if result["status"] == "Success":

            results.append(result)

            print(
                f"[{counter}] OK : {filename} | "
                f"SPAI={result['Staheli_Index']} | "
                f"{result['Klasifikasi']}"
            )

        else:

            failed_results.append({
                "filename": filename,
                "image_path": image_path,
                "error": result["error"]
            })

            print(
                f"[{counter}] GAGAL : {filename} | "
                f"{result['error']}"
            )

# =====================================
# SIMPAN CSV
# =====================================
df_success = pd.DataFrame(results)

df_success.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

failed_csv = os.path.join(
    SCRIPT_DIR,
    "failed_images-2.csv"
)

df_failed = pd.DataFrame(failed_results)

df_failed.to_csv(
    failed_csv,
    index=False,
    encoding="utf-8-sig"
)

# =====================================
# RINGKASAN
# =====================================
print("\n" + "="*50)
print("SELESAI")
print("="*50)
print(f"Total file diproses : {counter}")
print(f"Berhasil            : {len(df_success)}")
print(f"Gagal               : {len(df_failed)}")
print(f"CSV label           : {OUTPUT_CSV}")
print(f"CSV gagal           : {failed_csv}")
print("="*50)

print("\nDistribusi Label:")

if len(df_success) > 0:
    print(
        df_success["Klasifikasi"]
        .value_counts()
    )

print("\nContoh hasil:")
print(df_success.head())