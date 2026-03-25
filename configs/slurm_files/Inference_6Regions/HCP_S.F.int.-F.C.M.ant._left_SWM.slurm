#!/bin/bash
#SBATCH --job-name=HCP_S.F.int.-F.C.M.ant._left_SWM
#SBATCH -C v100-32g
##SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/Inference/HCP_S.F.int.-F.C.M.ant._left_SWM/HCP_S.F.int.-F.C.M.ant._left_SWM%j.out
#SBATCH --error=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/Inference/HCP_S.F.int.-F.C.M.ant._left_SWM/HCP_S.F.int.-F.C.M.ant._left_SWM%j.out
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

python3 Inference.py --model_dir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/FullModels/UKB_S.F.int.-F.C.M.ant._left_SWM --region=S.F.int.-F.C.M.ant._left --criteria=SWM --subjects=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/HCP1030.tsv --data=/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/crops_HCP/crops_HCP/S.F.int.-F.C.M.ant._left/SWM --outdir=/lustre/fswork/projects/rech/miu/ugf68us/PhD_2026/betaVAE/Inference/HCP_S.F.int.-F.C.M.ant._left_SWM --database=HCP 
