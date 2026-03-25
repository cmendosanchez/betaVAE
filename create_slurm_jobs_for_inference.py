import os
import argparse
import textwrap
from colors import bcolors
from datetime import datetime
import pandas as pd
#EXAMPLE
#python3 create_slurm_jobs_for_inference.py --regions S.C.-sylv._left S.C.-sylv._right S.T.s._left S.T.s._right S.F.int.-F.C.M.ant._right S.F.int.-F.C.M.ant._left --output_slurm /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE/configs/slurm_files/Inference_6Regions --databases UKB HCP --models /lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels --modes SWM DWM Comm --output_inference /lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/Inference


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
        help="Path to models"
    )

    parser.add_argument(
        "--output_slurm",
        required=True,
        help="Output folder for slurm files"
    )

    parser.add_argument(
        "--output_inference",
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
    output_slurm   = args.output_slurm
    output_inference = args.output_inference

    os.makedirs(output_slurm, exist_ok=True)
    for database in databases:
        for region in region_list:
            for mode in modes:
                config_name = f"{database}_{region}_{mode}"

                if database == 'UKB':
                    data     = f'/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/crops/{region}/{mode}'
                    subjects = f'/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/UKB37090.tsv'
                elif database == 'HCP':
                    data = f'/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/crops_HCP/crops_HCP/{region}/{mode}'
                    subjects = f'/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/HCP1030.tsv'

                job_name = f'{config_name}'
                print(f'{bcolors.GREEN}Writing {config_name}{bcolors.RESET}')
                python_call = (
                    f"python3 Inference.py "
                    f"--model_dir={models}/UKB_{region}_{mode} "
                    f"--region={region} "
                    f"--criteria={mode} "
                    f"--subjects={subjects} "
                    f"--data={data} "
                    f"--outdir={output_inference}/{database}_{region}_{mode} "
                    f"--database={database} "
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
                #SBATCH --time=10:00:00
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

                with open(f"{output_slurm}/{job_name}.slurm", "w") as f:
                    f.write(script)


if __name__ == "__main__":
    main()