#!/bin/bash
#SBATCH --job-name=UKB_S.F.int.-F.C.M.ant._left_SWM_2026_03_05_06_20_25
##SBATCH -C v100-32g
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/UKB_S.F.int.-F.C.M.ant._left_SWM_2026_03_05_06_20_25/UKB_S.F.int.-F.C.M.ant._left_SWM_2026_03_05_06_20_25%j.out
#SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/UKB_S.F.int.-F.C.M.ant._left_SWM_2026_03_05_06_20_25/UKB_S.F.int.-F.C.M.ant._left_SWM_2026_03_05_06_20_25%j.out
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

python3 Regional_Optuna_tuning.py +save_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/UKB_S.F.int.-F.C.M.ant._left_SWM_2026_03_05_06_20_25 +dataset=UKB_Train_6Regions_with_anom/UKB_S.F.int.-F.C.M.ant._left_SWM +optuna_folder=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/OptunaResults/UKB_S.F.int.-F.C.M.ant._left_SWM_2026_03_05_06_20_25 +Train_with_anomaly=False +optuna_lr=[1e-5,1e-2] +optuna_batch_size=[8,32] +optuna_epoch=[5,30] +optuna_ndim=256 +optuna_beta=1 +optuna_sub_perc=0.05 +optuna_ntrials=20 +optuna_enqueue_trial=False +dataset_folder=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions
