import os
import argparse
import textwrap
from colors import bcolors
from datetime import datetime

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

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():

    args = parse_args()

    region_list = args.regions
    databases = args.databases
    modes = args.modes
    output = args.output
    dataset_folder = args.dataset_folder
    train_tag = args.train_tag

    os.makedirs(output, exist_ok=True)

    for database in databases:
        for region in region_list:
            for mode in modes:

                config_name = f"{database}_{region}_{mode}"
                job_name = f'{config_name}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
                print(f'{bcolors.GREEN}Writing {config_name}{bcolors.RESET}')
                python_call = (
                    f"python3 Regional_Optuna_tuning.py "
                    f"+save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name} "
                    f"+dataset=UKB_Train_{train_tag}/{config_name} "
                    f"+optuna_folder=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name} "
                    f"+Train_with_anomaly=False "
                    f"+optuna_lr=[1e-5,1e-2] "
                    f"+optuna_batch_size=[8,32] "
                    f"+optuna_epoch=[5,30] "
                    f"+optuna_ndim=256 "
                    f"+optuna_beta=1 "
                    f"+optuna_sub_perc=0.10 "
                    f"+optuna_ntrials=20 "
                    f"+optuna_enqueue_trial=False "
                    f"+dataset_folder={dataset_folder}"
                )

                script = textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={job_name}
                ##SBATCH -C v100-32g
                #SBATCH -C h100
                #SBATCH --nodes=1
                #SBATCH --ntasks-per-node=1
                #SBATCH --gres=gpu:1
                #SBATCH --cpus-per-task=96
                #SBATCH --hint=nomultithread
                #SBATCH --time=20:00:00
                #SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name}/{job_name}%j.out
                #SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name}/{job_name}%j.out
                ##SBATCH -A miu@v100
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