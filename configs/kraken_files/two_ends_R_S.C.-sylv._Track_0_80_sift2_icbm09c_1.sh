#!/bin/bash
#PBS -q kraken
#PBS -l walltime=100:00:00
#PBS -N OptunaFibers
#PBS -l select=1:ncpus=24:ngpus=1:mem=125g:host=node-177
#PBS -o /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/OptunaResults/two_ends_R_S.C.-sylv._Track_0_80_sift2_icbm09c_trial_example2
#PBS -e /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/OptunaResults/two_ends_R_S.C.-sylv._Track_0_80_sift2_icbm09c_trial_example2
echo "$(whoami)@$(hostname)"
nvidia-smi
. /home_local/cm283129/env_torch/bin/activate
cd /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE
python3 Optuna_tuning.py +save_dir=/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output +dataset=PhD_UKB/two_ends_R_S.C.-sylv._Track_0_80_sift2_icbm09c +path_crops=/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/crops/R.S.C.-sylv/two_ends_R_S.C.-sylv._Track_0_80_sift2_icbm09c +optuna_folder=/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/OptunaResults/two_ends_R_S.C.-sylv._Track_0_80_sift2_icbm09c_trial_example2 +optuna_workers=2 +optuna_trials_per_worker=3
