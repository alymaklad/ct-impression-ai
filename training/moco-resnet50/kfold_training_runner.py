# src/kfold_runner.py
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold
import subprocess
import sys

def extract_patient_id(vol):
    # Example: "train_53_a_1"
    return vol.split("_")[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to train_metadata.csv")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to train_predicted_labels.csv")
    parser.add_argument("--npy_dir", type=str, required=True,
                        help="Path to dataset/train_fixed directory")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=str, default="folds/")
    parser.add_argument("--run_training", action="store_true",
                        help="If set, will call train_moco.py automatically")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load metadata & labels -------- #
    try:
        meta = pd.read_csv(args.metadata)
        labels = pd.read_csv(args.labels)
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return

    # Merge on VolumeName
    if "VolumeName" not in meta.columns or "VolumeName" not in labels.columns:
        print("Error: 'VolumeName' column missing in one of the CSVs.")
        return

    df_csv = meta.merge(labels, on="VolumeName", how="inner")

    # -------- Scan Directory for .npy files -------- #
    npy_path = Path(args.npy_dir)
    if not npy_path.exists():
        print(f"Error: npy_dir {args.npy_dir} does not exist.")
        return

    # List all .npy files
    npy_files = list(npy_path.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files in {args.npy_dir}")

    # Create DataFrame from files
    # Filename is VolumeName.npy -> VolumeName = stem
    # However, CSV VolumeName might have .nii.gz suffix or not.
    # Based on previous errors, CSV has .nii.gz suffix (e.g. train_1813_a_2.nii.gz)
    # But file on disk is train_1813_a_2.npy (or train_1813_a_2.nii.gz.npy?)
    # The user previously said: "remove the .nii.gz fromm the name".
    # And the file list shows: train_10014_a_1.npy.
    # So VolumeName should be train_10014_a_1.
    
    file_data = []
    for f in npy_files:
        vol_name = f.stem # removes .npy
        # If the file was named train_10014_a_1.nii.gz.npy, stem would be train_10014_a_1.nii.gz
        # If it is train_10014_a_1.npy, stem is train_10014_a_1
        # We need to match this with CSV VolumeName.
        # If CSV has train_10014_a_1.nii.gz, we need to strip .nii.gz from CSV to match file stem.
        file_data.append({"volume_name_clean": vol_name, "path": str(f)})
    
    df_files = pd.DataFrame(file_data)

    # Prepare CSV dataframe for merge
    # Create a clean volume name column to match with files
    df_csv["volume_name_clean"] = df_csv["VolumeName"].apply(lambda x: x.replace(".nii.gz", ""))

    # Merge files with CSV info
    # Inner join: only keep files that have metadata AND exist on disk
    df = df_files.merge(df_csv, on="volume_name_clean", how="inner")
    
    # Rename VolumeName back or keep clean one? dataset.py uses 'volume_name'
    df["volume_name"] = df["volume_name_clean"]
    
    # Extract patient IDs
    df["patientID"] = df["volume_name"].apply(extract_patient_id)

    # Unique patients for splitting
    patients = df["patientID"].unique()
    patients.sort()

    print(f"Total Matched Volumes (Files + Metadata): {len(df)}")
    print(f"Unique Training Patients: {len(patients)}")

    # -------- KFold on patient IDs -------- #
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    for fold_idx, (train_p_idx, val_p_idx) in enumerate(kf.split(patients)):

        train_pat = set(patients[train_p_idx])
        val_pat   = set(patients[val_p_idx])

        train_df = df[df["patientID"].isin(train_pat)].reset_index(drop=True)
        val_df   = df[df["patientID"].isin(val_pat)].reset_index(drop=True)

        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        train_csv = fold_dir / "train.csv"
        val_csv   = fold_dir / "val.csv"

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        print(f"\n===== FOLD {fold_idx} ===== ")
        print(f"Train patients: {len(train_pat)} -> samples: {len(train_df)}")
        print(f"Val patients:   {len(val_pat)} -> samples: {len(val_df)}")

        if args.run_training:
            cmd = [
                sys.executable, "src/train_moco.py",
                "--train_csv", str(train_csv),
                "--val_csv", str(val_csv),
            ]
            print("Executing:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
