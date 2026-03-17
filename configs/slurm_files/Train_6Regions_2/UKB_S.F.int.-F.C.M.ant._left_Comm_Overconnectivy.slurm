#!/bin/bash
#SBATCH --job-name=UKB_S.F.int.-F.C.M.ant._left_Comm_Overconnectivy
##SBATCH -C v100-32g
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults2/UKB_S.F.int.-F.C.M.ant._left_Comm_Overconnectivy/UKB_S.F.int.-F.C.M.ant._left_Comm_Overconnectivy%j.out
#SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults2/UKB_S.F.int.-F.C.M.ant._left_Comm_Overconnectivy/UKB_S.F.int.-F.C.M.ant._left_Comm_Overconnectivy%j.out
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

python3 Regional_Optuna_tuning.py +save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults2/UKB_S.F.int.-F.C.M.ant._left_Comm_Overconnectivy +dataset=UKB_Train_6Regions_with_anom/UKB_S.F.int.-F.C.M.ant._left_Comm_Overconnectivy +optuna_folder=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults2/UKB_S.F.int.-F.C.M.ant._left_Comm_Overconnectivy +optuna_lr=0.0008353816318944936 +optuna_batch_size=20 +optuna_epoch=50 +optuna_ndim=[10,500] +optuna_beta=[0.01,100.0] +optuna_sub_perc=0.5 +optuna_ntrials=6 +optuna_nworkers=5 +optuna_weight_decay=4.7856315650343575e-05 +patience=4 +delta=0.1 +dataset_folder=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions
