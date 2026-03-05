import os
import argparse
import yaml
import nibabel as nib
from tqdm import tqdm
from colors import bcolors

# -----------------------------
# YAML Dumper (inline lists)
# -----------------------------
class MyDumper(yaml.SafeDumper):
    pass

def represent_inline_list(dumper, data):
    return dumper.represent_sequence(
        'tag:yaml.org,2002:seq', data, flow_style=True
    )

MyDumper.add_representer(list, represent_inline_list)

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create YAML dataset configs for betaVAE sulci crops.",
    epilog="""
    Example:
    python3 create_yaml_datasets.py \
        --regions S.C.-sylv._left S.C.-sylv._right \
                S.T.s._left S.T.s._right \
                S.F.int.-F.C.M.ant._left S.F.int.-F.C.M.ant._right \
        --output /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE/configs/dataset/UKB_Train_6Regions
    """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--regions",
        nargs="+",
        required=True,
        help="List of region names (e.g. S.C.-sylv._left S.T.s._right)"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for YAML files"
    )

    parser.add_argument(
        "--mask_path",
        default="/neurospin/dico/cmendoza/Runs/17_PhD_2026/Output/mask_skeleton",
        help="Base path to sulci mask skeletons"
    )

    parser.add_argument(
        "--databases",
        nargs="+",
        default=["UKB"],
        help="Databases to use (default: UKB)"
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=["SWM", "DWM", "Comm"],
        help="Modes to generate (default: SWM DWM Comm)"
    )

    return parser.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    region_list = args.regions
    yaml_folder = args.output
    Path_sulci_masks = args.mask_path
    databases = args.databases
    modes = args.modes

    os.makedirs(yaml_folder, exist_ok=True)
    for anom in ['Underconnectivity','Overconnectivity']:
        for database in databases:
            for region in region_list:

                if "left" in region:
                    hemi = "L"
                elif "right" in region:
                    hemi = "R"
                else:
                    raise ValueError(f"Cannot infer hemisphere from region: {region}")

                region_base = region.replace("_left", "").replace("_right", "")
                mask_path = (
                    f"{Path_sulci_masks}/{region_base}/"
                    f"{hemi}mask_skeleton_1mm_crop.nii.gz"
                )

                dims = nib.load(mask_path).get_fdata().shape

                for mode in modes:
                    if mode == "SWM":
                        minl, maxl = 0, 80
                    elif mode == "DWM":
                        minl, maxl = 80, 250
                    elif mode == "Comm":
                        minl, maxl = 0, 250
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    print(
                        f"{bcolors.CYAN}Creating Dataset for "
                        f"{database} {region} {mode}{bcolors.RESET}"
                    )

                    dataset_name = f"{database}_{region}_{mode}_{anom}"

                    config = {
                        "dataset_name"   : dataset_name,
                        "in_shape"       : [1, dims[0], dims[1], dims[2]],
                        "Train_list"     : "${dataset_folder}/DataSplit_DL/train.tsv",
                        "Rcon_val_list"  : "${dataset_folder}/DataSplit_DL/val_rconerror.tsv",
                        "Class_val_list" : "${dataset_folder}/DataSplit_DL/val_anorm.tsv",
                        "Anomaly"    : anom,
                        "Criteria"   : mode,
                        "Region"     : region,
                        "Database"   : database,
                        "path_crops" : f"${{dataset_folder}}/crops/{region}/{mode}",
                        "path_anom"  : f"${{dataset_folder}}/FakeAnomaly_crops/{database}/{region}/{mode}/{anom}",
                        "path_stats" : f"${{dataset_folder}}/Stats_Anomaly/{database}/{database}_{region}_{anom}_{mode}.pkl",
                        "minl": minl,
                        "maxl": maxl,
                        "referential": "icbm09c",
                    }

                    with open(f"{yaml_folder}/{dataset_name}.yaml", "w") as f:
                        f.write("# @package _global_\n")
                        yaml.dump(config, f, sort_keys=False, Dumper=MyDumper)

if __name__ == "__main__":
    main()