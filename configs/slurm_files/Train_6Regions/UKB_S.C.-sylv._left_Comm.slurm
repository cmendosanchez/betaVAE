#!/bin/bash
#SBATCH --job-name=UKB_S.C.-sylv._left_Comm
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/tgu/ugf68us/PhD_UKB/betaVAE/configs/logs/UKB_S.C.-sylv._left_Comm%j.out
#SBATCH --error=/lustre/fswork/projects/rech/tgu/ugf68us/PhD_UKB/betaVAE/configs/logs/UKB_S.C.-sylv._left_Comm%j.out
#SBATCH -A tgu@a100

module purge
module load arch/a100
module purge
module load pytorch-gpu/py3/2.4.0

set -x

cd $WORK
cd PhD_UKB/betaVAE

python3 Regional_Optuna_tuning.py +save_dir=../Output +dataset=UKB_Train_6Regions/UKB_S.C.-sylv._left_Comm +optuna_folder=../OptunaResults/UKB_S.C.-sylv._left_Comm +Train_with_anomaly=False +optuna_lr=[1e-5,1e-2] +optuna_batch_size=[16,64] +optuna_epoch=[5,30] +optuna_ndim=64 +optuna_beta=1 +optuna_sub_perc=[0.10,0.20] +dataset_folder=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/PhD_2026/Crops_6Regions
