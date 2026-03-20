#!/bin/bash
#SBATCH --job-name=UKB_S.T.s._right_SWM
#SBATCH -C v100-32g
##SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.T.s._right_SWM/UKB_S.T.s._right_SWM%j.out
#SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.T.s._right_SWM/UKB_S.T.s._right_SWM%j.out
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

python3 Train_full_model.py +save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.T.s._right_SWM +dataset=UKB_Train_6Regions/UKB_S.T.s._right_SWM +patience=5 +delta=150.0 +dataset_folder=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions +path_model=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.T.s._right_SWM/model.pt n=128 kl=29.679374541131946 lr=1.36e-05 batch_size=16 weight_decay=5.04e-05 nb_epoch=50
