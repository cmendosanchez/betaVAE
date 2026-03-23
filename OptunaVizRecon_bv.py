import os
import argparse
import pandas as pd


def load_best_trials(csv_path):
    """
    Load CSV containing best trials per region.
    Expected columns: ['Region', 'Trial Number']
    """
    df = pd.read_csv(csv_path)
    return df


def get_trial_paths(root_dir, region, trial_number):
    """
    Build paths to input/output files for a given region and trial
    """
    trial_folder = os.path.join(root_dir, region, f"trial_{trial_number}")

    input_path = os.path.join(trial_folder, "input.nii.gz")
    output_path = os.path.join(trial_folder, "output.nii.gz")

    return input_path, output_path


def process_region(region, df, root_dir):
    """
    For a given region:
    - find best trial
    - get file paths
    """

    row = df[df["Region"] == region]

    if row.empty:
        print(f"⚠️ No entry found for region: {region}")
        return

    trial_number = int(row["Trial id"].iloc[0])

    input_path, output_path = get_trial_paths(root_dir, region, trial_number)

    # Check existence
    if not os.path.exists(input_path):
        print(f"❌ Missing input: {input_path}")
        return

    if not os.path.exists(output_path):
        print(f"❌ Missing output: {output_path}")
        return

    print(f"\n✅ Region: {region}")
    print(f"   Trial: {trial_number}")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")

    # ---- place your processing here ----
    # e.g., nibabel.load(input_path)
    # -----------------------------------


def main():
    parser = argparse.ArgumentParser(description="Process best trials per region")

    parser.add_argument("--regions", nargs="+", required=True,
                        help="List of regions")

    parser.add_argument("--criteria", nargs="+", required=True,
                        help="List of Seg. Criteria")

    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory with experiments")

    parser.add_argument("--csv", type=str, required=True,
                        help="CSV file with best trials")

    args = parser.parse_args()

    # Load dataframe
    df = load_best_trials(args.csv)

    # Loop over regions
    for region in args.regions:
        process_region(region, df, args.root_dir)


if __name__ == "__main__":
    main()