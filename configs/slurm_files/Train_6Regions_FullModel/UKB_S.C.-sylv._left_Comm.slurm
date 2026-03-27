#!/bin/bash
#SBATCH --job-name=UKB_S.C.-sylv._left_Comm
#SBATCH -C v100-32g
##SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.C.-sylv._left_Comm/UKB_S.C.-sylv._left_Comm%j.out
#SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.C.-sylv._left_Comm/UKB_S.C.-sylv._left_Comm%j.out
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

python3 Train_full_model.py +save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.C.-sylv._left_Comm/ +dataset=UKB_Train_6Regions/UKB_S.C.-sylv._left_Comm +dataset_folder=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions +path_model=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.C.-sylv._left_Comm/model.pt +early_stopping=0 n=64 kl=3.743033228835967 lr=0.00178 batch_size=8 weight_decay=1.13e-06 nb_epoch=20
