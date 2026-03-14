import os
import argparse
import textwrap
from colors import bcolors
from datetime import datetime
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import optuna
#EXAMPLE
#python create_slurm_jobs_from_study.py --regions S.C.-sylv._left S.C.-sylv._right S.T.s._left S.T.s._right S.F.int.-F.C.M.ant._right S.F.int.-F.C.M.ant._left --optuna_study ../../../../OptunaResults --output /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE/configs/slurm_files/Train_6Regions_2 --train_tag 6Regions_with_anom --dataset_folder /lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions --epochs 50 --beta 0.01 10 --sub_perc 1.0 --ntrials 12 --nworkers 3 --anom Underconnectivity Overconnectivity


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
    "--optuna_study",
    type=str,
    default=10,
    help="Path_to_optuna_study"
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
        nargs="+",
        default=[50],
        type=int,
        help="Number of epochs"
    )

    parser.add_argument(
        "--beta",
        nargs="+",
        default=[0.01,10],
        type=float,
        help="Beta"
    )

    parser.add_argument(
    "--sub_perc",
    type=float,
    default=0.05,
    help="Subject percentage (default: 0.05)"
    )

    parser.add_argument(
    "--ntrials",
    type=int,
    default=10,
    help="Number of optuna trials (default: 10)"
    )

    parser.add_argument(
    "--nworkers",
    type=int,
    default=5,
    help="Number of optuna workers(default: 5)"
    )

    parser.add_argument(
        "--anom",
        nargs="+",
        default=['None'],
        help="Anom to to use (Underconnectivity/Overconnectivity)"
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
    optuna_study   = args.optuna_study
    epochs      = args.epochs
    beta        = args.beta
    anoms       = args.anom
    ntrials     = args.ntrials
    sub_perc    = args.sub_perc
    nworkers    = args.nworkers


    

    os.makedirs(output, exist_ok=True)

    for anom in anoms:
        for database in databases:
            for region in region_list:

                for mode in modes:

                    if anom != 'None':
                        config_name = f"{database}_{region}_{mode}_{anom}"
                    else:
                        config_name = f"{database}_{region}_{mode}"

                     # Load study
                    journal_path = os.path.join(f'{args.optuna_study}/{database}_{region}_{mode}', "journal.log")
                    storage = JournalStorage(JournalFileBackend(journal_path))
                    study_name="journal_storage_multiprocess"
                    study = optuna.load_study(
                        study_name=study_name,
                        storage=storage)

                    best_trial = study.best_trial
                    Params = best_trial.params
                    print(Params)
                    #job_name = f'{config_name}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
                    job_name = f'{config_name}'
                    print(f'{bcolors.GREEN}Writing {config_name}{bcolors.RESET}')
                    python_call = (
                        f"python3 Regional_Optuna_tuning.py "
                        f"+save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name} "
                        f"+dataset=UKB_Train_{train_tag}/{config_name} "
                        f"+optuna_folder=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/{job_name} "
                        f"+optuna_lr={Params['Learning Rate']} "
                        f"+optuna_batch_size={Params['Batch size']} "
                        f"+optuna_epoch={format_range(epochs)} "
                        f"+optuna_ndim={Params['Dimensions']} "
                        f"+optuna_beta={format_range(beta)} "
                        f"+optuna_sub_perc={sub_perc} "
                        f"+optuna_ntrials={ntrials} "
                        f"+optuna_nworkers={nworkers} "
                        f"+optuna_weight_decay={Params['Weight decay']} "
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
                    #SBATCH --time=10:00:00
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