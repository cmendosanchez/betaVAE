#!/bin/bash
#SBATCH --job-name=UKB_S.C.-sylv._right_SWM_Underconnectivity
##SBATCH -C v100-32g
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults2/UKB_S.C.-sylv._right_SWM_Underconnectivity/UKB_S.C.-sylv._right_SWM_Underconnectivity%j.out
#SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults2/UKB_S.C.-sylv._right_SWM_Underconnectivity/UKB_S.C.-sylv._right_SWM_Underconnectivity%j.out
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

python3 Regional_Optuna_tuning.py +save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults2/UKB_S.C.-sylv._right_SWM_Underconnectivity +dataset=UKB_Train_6Regions_with_anom/UKB_S.C.-sylv._right_SWM_Underconnectivity +optuna_folder=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults2/UKB_S.C.-sylv._right_SWM_Underconnectivity +optuna_lr=0.00011969516658498936 +optuna_batch_size=11 +optuna_epoch=50 +optuna_ndim=166 +optuna_beta=[0.01,10.0] +optuna_sub_perc=1.0 +optuna_ntrials=12 +optuna_nworkers=3 +optuna_weight_decay=4.3879789776687354e-07 +dataset_folder=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions
