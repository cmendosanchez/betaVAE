#!/bin/bash
#SBATCH --job-name=UKB_S.T.s._right_Comm
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/configs/logs/UKB_S.T.s._right_Comm%j.out
#SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/configs/logs/UKB_S.T.s._right_Comm%j.out
#SBATCH -A miu@a100

module purge
module load arch/a100
module purge
module load pytorch-gpu/py3/2.4.0

set -x

cd $WORK
cd PhD_2026/betaVAE

python3 Regional_Optuna_tuning.py +save_dir=../Output +dataset=UKB_Train_6Regions/UKB_S.T.s._right_Comm +optuna_folder=../OptunaResults/UKB_S.T.s._right_Comm +Train_with_anomaly=False +optuna_lr=[1e-5,1e-2] +optuna_batch_size=[16,64] +optuna_epoch=[5,30] +optuna_ndim=64 +optuna_beta=1 +optuna_sub_perc=[0.10,0.20] +optuna_ntrials=1 +dataset_folder=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/PhD_2026/Crops_6Regions
