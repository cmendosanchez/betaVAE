#!/bin/bash
#SBATCH --job-name=UKB_S.T.s._right_SWM
##SBATCH -C v100-32g
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/UKB_S.T.s._right_SWM/UKB_S.T.s._right_SWM%j.out
#SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/UKB_S.T.s._right_SWM/UKB_S.T.s._right_SWM%j.out
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

python3 Regional_Optuna_tuning.py +save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/UKB_S.T.s._right_SWM +dataset=UKB_Train_6Regions/UKB_S.T.s._right_SWM +optuna_folder=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/UKB_S.T.s._right_SWM +optuna_lr=[1e-05,0.01] +optuna_batch_size=[8,32] +optuna_epoch=2 +optuna_ndim=[32,256] +optuna_beta=[1.0,30.0] +optuna_sub_perc=0.1 +optuna_ntrials=6 +optuna_nworkers=5 +optuna_weight_decay=[1e-07,0.01] +patience=4 +delta=150.0 +dataset_folder=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions
