import os
import argparse
import textwrap
from colors import bcolors
from datetime import datetime
import pandas as pd
#EXAMPLE
#python3 create_slurm_jobs_from_study.py --regions S.C.-sylv._left S.C.-sylv._right S.T.s._left S.T.s._right S.F.int.-F.C.M.ant._right S.F.int.-F.C.M.ant._left --output /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE/configs/slurm_files/Train_6Regions_FullModel --train_tag 6Regions --dataset_folder /lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions --epochs 50 --path_params ../../../../OptunaResults/summary.csv --delta 150 --patience 5

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
        "--models",
        required=True,
        help="Output folder for slurm files"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for slurm files"
    )

    parser.add_argument(
        "--database",
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
        "--subjects",
        required=True,
        help="Output folder for slurm files"
    )

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():

    args = parse_args()

    region_list    = args.regions
    databases      = args.databases
    models         = args.models
    modes          = args.modes
    output         = args.output
    subjects       = args.subjects

    os.makedirs(output, exist_ok=True)
    for database in databases:
        for region in region_list:
            for mode in modes:
                config_name = f"{database}_{region}_{mode}"

                if database == 'UKB':
                    data= f'/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/crops/{region}/{mode}'
                elif database == 'HCP':
                    data = f'/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/crops_HCP/{region}/{mode}'

                job_name = f'{config_name}'
                print(f'{bcolors.GREEN}Writing {config_name}{bcolors.RESET}')
                python_call = (
                    f"python3 Inference.py "
                    f"--model_dir={models} "
                    f"--region={region} "
                    f"--criteria={mode} "
                    f"--subjects={subjects} "
                    f"--data={data}"
                    )

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
                #SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/Inference/{job_name}/{job_name}%j.out
                #SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/Inference/{job_name}/{job_name}%j.out
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