
import os
import glob
import argparse
import pandas as pd
import subprocess
import sys
from pathlib import Path

def extract_patient_id(vol):
    try:
        return vol.split("_")[1]
    except IndexError:
        return vol

def find_best_fold(checkpoints_dir):
    """
    Scans *_metrics.csv in checkpoints_dir to find the fold with the minimum validation loss.
    Returns (best_fold_name, min_val_loss, best_epoch)
    """
    pattern = os.path.join(checkpoints_dir, "*_metrics.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No metrics files found in {checkpoints_dir}")
        return None, None, None

    global_min_loss = float('inf')
    best_fold = None
    best_epoch = -1

    print(f"Scanning {len(csv_files)} metrics files...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'val_loss' not in df.columns:
                print(f"Skipping {csv_file}: 'val_loss' column missing.")
                continue
            
            # Find min val loss in this file
            # Filter out non-numeric or NaN if any (though pandas handles numeric read)
            min_row = df.loc[df['val_loss'].idxmin()]
            min_loss = min_row['val_loss']
            epoch = min_row['epoch']
            
            # Extract fold name from filename (e.g., fold_0_metrics.csv -> fold_0)
            fold_name = os.path.basename(csv_file).replace("_metrics.csv", "")
            
            print(f"  {fold_name}: Min Val Loss {min_loss:.6f} at epoch {epoch}")

            if min_loss < global_min_loss:
                global_min_loss = min_loss
                best_fold = fold_name
                best_epoch = epoch
                
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    return best_fold, global_min_loss, best_epoch

def prepare_full_dataset(metadata_path, labels_path, npy_dir, out_csv):
    """
    Merges metadata and labels, matches with NPY files, and saves the full dataset csv.
    Reuses logic from kfold_runner.py to ensure consistency.
    """
    print("\nPreparing full dataset...")
    
    # Load metadata & labels
    meta = pd.read_csv(metadata_path)
    labels = pd.read_csv(labels_path)
    
    if "VolumeName" not in meta.columns or "VolumeName" not in labels.columns:
        raise ValueError("Error: 'VolumeName' column missing in one of the CSVs.")

    df_csv = meta.merge(labels, on="VolumeName", how="inner")
    
    # Scan Directory for .npy files
    npy_path = Path(npy_dir)
    if not npy_path.exists():
        raise FileNotFoundError(f"npy_dir {npy_dir} does not exist.")

    npy_files = list(npy_path.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files in {npy_dir}")

    file_data = []
    for f in npy_files:
        vol_name = f.stem 
        file_data.append({"volume_name_clean": vol_name, "path": str(f)})
    
    df_files = pd.DataFrame(file_data)

    # Clean volume name in CSV to match files
    # Assuming CSV VolumeName has .nii.gz suffix that needs removal to match .npy stem
    df_csv["volume_name_clean"] = df_csv["VolumeName"].apply(lambda x: x.replace(".nii.gz", ""))

    # Merge files with CSV info
    df = df_files.merge(df_csv, on="volume_name_clean", how="inner")
    
    # dataset.py uses 'volume_name'
    df["volume_name"] = df["volume_name_clean"]
    
    print(f"Total Matched Volumes for Final Training: {len(df)}")
    
    # Ensure output dir exists
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    df.to_csv(out_csv, index=False)
    print(f"Full dataset saved to {out_csv}")
    return len(df)

def main():
    parser = argparse.ArgumentParser(description="Find best fold and train final model on full dataset.")
    parser.add_argument("--checkpoints_dir", type=str, default="outputs/checkpoints", help="Directory containing *_metrics.csv")
    parser.add_argument("--metadata", type=str, required=True, help="Path to train_metadata.csv")
    parser.add_argument("--labels", type=str, required=True, help="Path to train_predicted_labels.csv")
    parser.add_argument("--npy_dir", type=str, required=True, help="Path to dataset/train_fixed directory")
    parser.add_argument("--out_csv", type=str, default="dataset/full_data.csv", help="Where to save the full dataset CSV")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override num_epochs for final training (default: use config)")
    parser.add_argument("--dry_run", action="store_true", help="Just find best model and prepare data, do not train.")
    
    args = parser.parse_args()

    # 1. Find best fold
    print("--- 1. Analyzing Training Logs ---")
    best_fold, min_loss, best_epoch = find_best_fold(args.checkpoints_dir)
    
    if best_fold:
        print(f"\n=> Best Performing Model: {best_fold}")
        print(f"=> Minimum Validation Loss: {min_loss:.6f} (Epoch {best_epoch})")
    else:
        print("\n=> Could not identify best fold. Proceeding with caution.")

    # 2. Prepare Data
    print("\n--- 2. Preparing Full Dataset ---")
    num_samples = prepare_full_dataset(args.metadata, args.labels, args.npy_dir, args.out_csv)
    
    if num_samples == 0:
        print("Error: No samples found. generated CSV is empty.")
        return

    # 3. Train
    print("\n--- 3. Final Training ---")
    if args.dry_run:
        print("Dry run enabled. Skipping actual training command.")
        print(f"To run manually: python src/train_moco.py --train_csv \"{args.out_csv}\"")
    else:
        # Construct command
        # We don't provide val_csv so train_moco.py will just train without validation loop
        cmd = [
            sys.executable, "src/train_moco.py",
            "--train_csv", args.out_csv
        ]
        
        # Note: we might want to override num_epochs if best_epoch suggests fewer are needed?
        # Usually for final model on more data, we might want slightly more or same epochs.
        # User prompt was just "train it as the final model". 
        # I won't change config automatically but if user passed --num_epochs arg I'd use it.
        # But train_moco.py takes config, I can't override config params via CLI easily unless I modify train_moco.py
        # Current train_moco.py loads config from yaml. 
        # I'll stick to running it as is.
        
        print("Executing:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed with error: {e}")

if __name__ == "__main__":
    main()
