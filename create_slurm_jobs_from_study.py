import os
import argparse
import textwrap
from colors import bcolors
from datetime import datetime
import pandas as pd
#EXAMPLE
#python3 create_slurm_jobs_from_study.py --regions S.C.-sylv._left S.C.-sylv._right S.T.s._left S.T.s._right S.F.int.-F.C.M.ant._right S.F.int.-F.C.M.ant._left --output /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE/configs/slurm_files/Train_6Regions_FullModel --train_tag 6Regions --dataset_folder /lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions --epochs 50 --path_params ../../../../OptunaResults/summary.csv --delta 150 --patience 5


def format_range(values):
    """
    If one value -> return single value as string
    If multiple values -> return [min,max] without spaces
    """
    if len(values) == 1:
        return str(values[0])
    else:
        return f"[{min(values)},{max(values)}]"

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
    description="Generate SLURM files for Regional Optuna tuning",
    epilog="""
    Example:
        python create_slurm_jobs.py \
    --regions S.C.-sylv._left S.C.-sylv._right S.T.s._left \
    --output /neurospin/.../UKB_Train_3Regions_slurm_files \
    --train_tag 3Regions \
    --dataset_folder PhD_2026/Crops_3Regions
        """,
    formatter_class=argparse.RawTextHelpFormatter
    )
    

    parser.add_argument(
        "--regions",
        nargs="+",
        required=True,
        help="List of region names"
    )


    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for slurm files"
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
        help="Modes to use (default: SWM DWM Comm)"
    )

    parser.add_argument(
        "--dataset_folder",
        required=True,
        help="Dataset folder passed to the training script"
    )

    parser.add_argument(
        "--train_tag",
        required=True,
        help="Training tag (e.g. 6Regions, 3Regions...)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for Early Stopping (default: 50)"
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=150,
        help="Delta for Early Stopping"
    )

    parser.add_argument(
        "--path_params",
        type=str,
        required=True,
        help="Path to .csv containing the parameters"
    )

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():

    args = parse_args()

    region_list    = args.regions
    databases      = args.databases
    modes          = args.modes
    output         = args.output
    dataset_folder = args.dataset_folder
    train_tag      = args.train_tag
    patience       = args.patience
    delta          = args.delta
    epochs         = args.epochs
    path_params    = args.path_params

    os.makedirs(output, exist_ok=True)
    for database in databases:
        for region in region_list:
            for mode in modes:
                config_name = f"{database}_{region}_{mode}"
                
                df = pd.read_csv(path_params)
                params  = df[
                (df["Region"] == region) & 
                (df["Seg. Criteria"] == mode)
                ]

                ndims = params['Dimensions'].iloc[0]
                beta  = params['Beta'].iloc[0]
                lr    = params['Learning Rate'].iloc[0]
                batch_size = params['Batch size'].iloc[0]
                weight_decay = params['Weight decay'].iloc[0]

                #job_name = f'{config_name}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
                job_name = f'{config_name}'
                print(f'{bcolors.GREEN}Writing {config_name}{bcolors.RESET}')
                python_call = (
                    f"python3 Train_full_model.py "
                    f"+save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/{job_name} "
                    f"+dataset=UKB_Train_{train_tag}/{config_name} "
                    f"+patience={patience} "
                    f"+delta={delta} "
                    f"+dataset_folder={dataset_folder} "
                    f"+path_model=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/{job_name}/model.pt "
                    f"n={ndims} "
                    f"kl={beta} "
                    f"lr={lr} "
                    f"batch_size={batch_size} "
                    f"weight_decay={weight_decay} "
                    f"nb_epoch={epochs}")

                script = textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={job_name}
                #SBATCH -C v100-32g
                ##SBATCH -C h100
                #SBATCH --nodes=1
                #SBATCH --ntasks-per-node=1
                #SBATCH --gres=gpu:1
                #SBATCH --cpus-per-task=40
                #SBATCH --hint=nomultithread
                #SBATCH --time=20:00:00
                #SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/{job_name}/{job_name}%j.out
                #SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/{job_name}/{job_name}%j.out
                ##SBATCH -A miu@v100
                ##SBATCH -A miu@h100
                #SBATCH -A miu@v100

                module purge
                ##module load arch/h100
                module load pytorch-gpu/py3/2.8.0

                nvidia-smi
                lscpu
                free -h

                echo $SLURM_MEM_PER_CPU
                echo $SLURM_CPUS_PER_TASK

                set -x

                cd $WORK
                cd PhD_2026/betaVAE

                {python_call}
                """)

                with open(f"{output}/{job_name}.slurm", "w") as f:
                    f.write(script)


if __name__ == "__main__":
    main()