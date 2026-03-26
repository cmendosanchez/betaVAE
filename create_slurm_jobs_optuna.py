import os
import argparse
import textwrap
from colors import bcolors
from datetime import datetime

#EXAMPLE
#python3 create_slurm_jobs_optuna.py --regions S.C.-sylv._left S.C.-sylv._right S.T.s._left S.T.s._right S.F.int.-F.C.M.ant._right S.F.int.-F.C.M.ant._left --output /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE/configs/slurm_files/Train_6Regions_Optuna --train_tag 6Regions --dataset_folder /lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions --epochs 20 --ndims 16 256 --beta 1 20 --sub_perc 0.25 --ntrials 6 --nworkers 5 --early_stop 0 --database UKB --modes SWM DWM Comm


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
        type=str,
        required=True,
        help="Output folder for slurm files"
    )

    parser.add_argument(
        "--database",
        type=str,
        default="UKB",
        required=True,
        help="Database (default: UKB)"
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=["SWM", "DWM", "Comm"],
        required=True,
        help="Modes to use (default: SWM DWM Comm)"
    )

    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Dataset folder passed to the training script"
    )

    parser.add_argument(
        "--train_tag",
        type=str,
        required=True,
        help="Training tag (e.g. 6Regions, 3Regions...)"
    )

    parser.add_argument(
        "--lr",
        nargs="+",
        default=[1e-5,1e-2],
        help="Learning rate"
    )

    parser.add_argument(
        "--weight_decay",
        nargs="+",
        default=[1e-7, 1e-2],
        help="weight_decay"
    )

    parser.add_argument(
        "--batch_size",
        nargs="+",
        default=[8,32],
        help="Batch size"
    )

    parser.add_argument(
        "--epochs",
        default=30,
        required=True,
        type=int,
        help="Number of epochs"
    )

    parser.add_argument(
        "--ndims",
        nargs="+",
        default=[32,256],
        help="Number of latent dimensions"
    )

    parser.add_argument(
        "--beta",
        nargs="+",
        default=[1,20],
        help="Beta"
    )

    parser.add_argument(
    "--sub_perc",
    type=float,
    default=0.25,
    required=True,
    help="Subjects percentage (default: 0.25)"
    )

    parser.add_argument(
    "--ntrials",
    type=int,
    default=10,
    required=True,
    help="Number of optuna trials (default: 10)"
    )

    parser.add_argument(
    "--nworkers",
    type=int,
    default=5,
    required=True,
    help="Number of optuna workers (default: 5)"
    )

    parser.add_argument(
    "--early_stop",
    type=int,
    default=1,
    required=True,
    help="Activate early stopping")

    parser.add_argument(
    "--patience",
    type=int,
    default=5,
    help="Patience for Early Stopping (default: 5)"
    )

    parser.add_argument(
    "--delta",
    type=float,
    default=0,
    help="Delta for Early Stopping (default: 0)"
    )


    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():

    args = parse_args()

    region_list    = args.regions
    database       = args.database
    modes          = args.modes
    output         = args.output
    dataset_folder = args.dataset_folder
    train_tag      = args.train_tag
    weight_decay   = args.weight_decay
    lr             = args.lr
    batch_size     = args.batch_size
    epochs         = args.epochs
    ndims          = args.ndims
    beta           = args.beta
    ntrials        = args.ntrials
    sub_perc       = args.sub_perc
    nworkers       = args.nworkers
    patience       = args.patience
    delta          = args.delta
    earlystop      = args.early_stop

    os.makedirs(output, exist_ok=True)

    for region in region_list:
        for mode in modes:

            config_name = f"{database}_{region}_{mode}"
            job_name = f'{config_name}'
            print(f'{bcolors.GREEN}Writing {config_name}{bcolors.RESET}')

            if earlystop == 1:
                python_call = (
                    f"python3 Regional_Optuna_tuning.py "
                    f"+save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name} "
                    f"+dataset=UKB_Train_{train_tag}/{config_name} "
                    f"+optuna_folder=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name} "
                    f"+optuna_lr={format_range(lr)} "
                    f"+optuna_batch_size={format_range(batch_size)} "
                    f"+optuna_epoch={epochs} "
                    f"+optuna_ndim={format_range(ndims)} "
                    f"+optuna_beta={format_range(beta)} "
                    f"+optuna_sub_perc={sub_perc} "
                    f"+optuna_ntrials={ntrials} "
                    f"+optuna_nworkers={nworkers} "
                    f"+optuna_weight_decay={format_range(weight_decay)} "
                    f"+early_stopping={earlystop} "
                    f"+patience={patience} "
                    f"+delta={delta} "
                    f"+dataset_folder={dataset_folder}")
            else:
                python_call = (
                    f"python3 Regional_Optuna_tuning.py "
                    f"+save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name} "
                    f"+dataset=UKB_Train_{train_tag}/{config_name} "
                    f"+optuna_folder=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name} "
                    f"+optuna_lr={format_range(lr)} "
                    f"+optuna_batch_size={format_range(batch_size)} "
                    f"+optuna_epoch={epochs} "
                    f"+optuna_ndim={format_range(ndims)} "
                    f"+optuna_beta={format_range(beta)} "
                    f"+optuna_sub_perc={sub_perc} "
                    f"+optuna_ntrials={ntrials} "
                    f"+optuna_nworkers={nworkers} "
                    f"+optuna_weight_decay={format_range(weight_decay)} "
                    f"+early_stopping={earlystop} "
                    f"+dataset_folder={dataset_folder}")

            script = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name={job_name}
            #SBATCH -C h100
            #SBATCH --nodes=1
            #SBATCH --ntasks-per-node=1
            #SBATCH --gres=gpu:1
            #SBATCH --cpus-per-task=96
            #SBATCH --hint=nomultithread
            #SBATCH --time=20:00:00
            #SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name}/{job_name}%j.out
            #SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name}/{job_name}%j.out
            #SBATCH -A miu@h100

            module purge
            module load arch/h100
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