#!/bin/bash
#SBATCH --job-name=UKB_S.F.int.-F.C.M.ant._right_DWM
#SBATCH -C v100-32g
##SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.F.int.-F.C.M.ant._right_DWM/UKB_S.F.int.-F.C.M.ant._right_DWM%j.out
#SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.F.int.-F.C.M.ant._right_DWM/UKB_S.F.int.-F.C.M.ant._right_DWM%j.out
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

python3 Train_full_model.py +save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.F.int.-F.C.M.ant._right_DWM +dataset=UKB_Train_6Regions/UKB_S.F.int.-F.C.M.ant._right_DWM +patience=5 +delta=150.0 +dataset_folder=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions +path_model=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.F.int.-F.C.M.ant._right_DWM/model.pt n=128 kl=19.41483919337259 lr=0.00244 batch_size=16 weight_decay=0.00184 nb_epoch=50
