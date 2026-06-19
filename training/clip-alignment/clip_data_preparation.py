# import pandas as pd
# import os


# BASE_DIR = "C:/MTI Research/Preprocessing_CPU/ValidationNpy"
# def build_volume_path(volume_name):
#     # print(BASE_DIR + "/" + volume_name + ".npy")
#     return str(BASE_DIR + "/" + volume_name + ".npy")


# # Read the CSV file
# df = pd.read_csv(r"C:\MTI Research\dataset\radiology_text_reports\validation_reports.csv")

# # Concatenate Impression_EN and Findings_EN
# df['report_text'] = df['Findings_EN'].fillna('') + ' ' + df['Impressions_EN'].fillna('')
# df['report_text'] = df['report_text'].str.strip()

# # Assuming there's an identifier column (e.g., 'patient_id' or 'volume_id') to match with volume paths
# # Adjust the column name as needed

# # Assuming volume names are filenames (without extensions) in the directory
# volume_names = [os.path.splitext(f)[0] for f in os.listdir(BASE_DIR) if os.path.isfile(os.path.join(BASE_DIR, f))]


# df["VolumeName"] = df["VolumeName"].str.replace(".nii.gz", "", regex=False)

# # Add volume path column (adjust 'identifier_column' to your actual column name)
# filtered_df = df[df['VolumeName'].isin(volume_names)]
# filtered_df['volume_path'] = filtered_df['VolumeName'].apply(build_volume_path)

# # print(filtered_df[['report_text', 'volume_path']].head())
# # print(f"Total matched records: {len(filtered_df)}")



# # # Keep only report_text and volume_path
# result_df = filtered_df[['report_text', 'volume_path']]
# #
# print(result_df.head())
# result_df.to_csv("C:/MTI Research/dataset/Language_Image_Data/test_processed_reports.csv", index=False)


from numpy import rint
import pandas as pd
import os


# =========================
# Paths
# =========================
TRAIN_VOL_CSV = "C:/MTI Research/MoCo3D-MedicalNet-ResNet50/moco3d_medicalnet/Evaluation/Train_Test_Labels/train.csv"
VAL_VOL_CSV   = "C:/MTI Research/MoCo3D-MedicalNet-ResNet50/moco3d_medicalnet/Evaluation/Train_Test_Labels/val.csv"

REPORTS_CSV  = "C:/MTI Research/dataset/Language_Image_Data/train_processed_reports.csv"

OUT_TRAIN_CSV = "C:/MTI Research/dataset/Language_Image_Data/train_volumes_with_reports.csv"
OUT_VAL_CSV   = "C:/MTI Research/dataset/Language_Image_Data/val_volumes_with_reports.csv"


# =========================
# Column names (EDIT IF NEEDED)
# =========================
VOLUME_PATH_COL = "path"   # column in train.csv / val.csv
REPORT_PATH_COL = "path"   # column in reports csv used for matching
REPORT_TEXT_COL = "report_text"        # text report column in reports csv


# =========================
# Helper: extract ID from volume path
# =========================


# =========================
# Load data
# =========================
train_vols = pd.read_csv(TRAIN_VOL_CSV)
val_vols   = pd.read_csv(VAL_VOL_CSV)
reports    = pd.read_csv(REPORTS_CSV)


def normalize_path(p):
    return p.replace("\\", "/")

train_vols["path"] = train_vols["path"].apply(normalize_path)
val_vols["path"]   = val_vols["path"].apply(normalize_path)
reports["path"]    = reports["path"].apply(normalize_path)


# =========================
# Merge
# =========================
# Merge train volumes with reports based on exact path match
train_merged = train_vols.merge(
    reports[["path", "report_text"]],
    on="path",
    how="inner"
)[["report_text", "path"]]

val_merged = val_vols.merge(
    reports[["path", "report_text"]],
    on="path",
    how="inner"
)[["report_text", "path"]]

print(f"Train merged records: {train_merged.head()}")
print(f"Train merged records: {len(train_merged)}")


print(f"val merged records: {val_merged.head()}")
print(f"Val merged records: {len(val_merged)}")

# # =========================
# # Save
# # =========================
train_merged.to_csv(OUT_TRAIN_CSV, index=False)
val_merged.to_csv(OUT_VAL_CSV, index=False)

# print("✅ CSV files created successfully:")
# print(OUT_TRAIN_CSV)
# print(OUT_VAL_CSV)
