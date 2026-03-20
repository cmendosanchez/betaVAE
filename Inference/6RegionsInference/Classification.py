import os
import sys
import argparse
import yaml
import torch 
import torch.nn as nn
from types import SimpleNamespace



def setup_paths():
    """
    Add project root (2 levels up) to PYTHONPATH
    """
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if ROOT not in sys.path:
        sys.path.append(ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Run VAE script")

    # List of strings argument
    parser.add_argument(
        "--regions",
        nargs="+",                # accepts 1 or more values
        type=str,
        required=True,
        help="List of region names"
    )

    parser.add_argument(
        "--models_folder",
        type=str,
        required=True,
        help="Path to models folder"
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="Path to test dataset"
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=["SWM", "DWM", "Comm"],
        help="Modes to use (default: SWM DWM Comm)"
    )


    return parser.parse_args()


def main():
    args = parse_args()

    # Now you can safely import
    from beta_vae import VAE
    #from General_utils import read_one_column_tsv
    #from load_data import create_subset_from_list
    from colors import bcolors
    from GetAUC import get_AUC_testing

    print("Running main...")
    print("Regions:", args.regions)

    regions       = args.regions
    modes         = args.modes
    test_dataset  = args.test_dataset
    models_folder = args.models_folder

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Example usage
    for region in regions:
        print(f"Processing {region}")
        #subjects = read_one_column_tsv(test_dataset)
        #print(f'{subjects[0:10]}...{bcolors.GREEN}\nNsubjects: {len(subjects)}{bcolors.RESET}')
        for mode in modes:
            model_dir = f'{models_folder}/UKB_{region}_{mode}/model.pt'
            config_file    = f'{models_folder}/UKB_{region}_{mode}/config.yaml'
            config_file    = f'/neurospin/dico/cmendoza/FullModels/UKB_S.C.-sylv._left_SWM/2026-03-20/UKB_S.C.-sylv._left_SWM_dim_64_beta_5.2442628875334885_13-14-41/config.yaml'
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)
            print(config)
            print(model_dir)
            model = VAE(config['in_shape'], config['n'], depth=config['depth'], loss_selected = config['loss'])
            #print('torch load',torch.load(model_dir,weights_only=False))
            checkpoint = torch.load(model_dir,weights_only=False)
            #print('torch load',torch.load(model_dir)[1])
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            config['path_crops']     = f'/neurospin/dico/cmendoza/Runs/17_PhD_2026/Output/crops/{region}/{mode}'
            config['path_anom']      = f'/neurospin/dico/cmendoza/Runs/17_PhD_2026/Output//FakeAnomaly_crops/UKB/{region}/{mode}/'
            config['path_stats']     = f'/neurospin/dico/cmendoza/Runs/17_PhD_2026/Output/Stats_Anomaly/UKB/UKB_{region}_'
            config['Class_val_list'] = test_dataset
            #subjects_set = create_subset_from_list(config,subjects)
            #dataloader = torch.utils.data.DataLoader(subjects_set,batch_size=32,num_workers=12, shuffle=False)
            config = SimpleNamespace(**config)
            print(config)
            resulting_aucs, individual_aucs = get_AUC_testing(config,model,device,nn.MSELoss(reduction='sum'))



    print("Result:")


if __name__ == "__main__":
    setup_paths()
    main()