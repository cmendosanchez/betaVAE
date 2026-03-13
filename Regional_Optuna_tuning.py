# /usr/bin/env python3
# coding: utf-8
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
#
# https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate/


import os
import hydra
from omegaconf import OmegaConf
from omegaconf import ListConfig
import time
import torch
from datetime import datetime
from train import train_vae_optuna
from utils.config import process_config
import optuna
from optuna.pruners import PatientPruner, MedianPruner
from colors import bcolors
from copy import deepcopy
import threading
from General_utils import adjust_in_shape
from concurrent.futures import ProcessPoolExecutor as Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

def is_range(x):
    return isinstance(x, (list, ListConfig))

def validate_range(x, name):
    if len(x) != 2:
        raise ValueError(f"{name} must contain exactly 2 values (low, high)")
    return x[0], x[1]

now = datetime.now()
optuna.logging.set_verbosity(optuna.logging.INFO)

# This will be the objective function for Optuna optimization
def objective(trial, config):
    try:
        print(f"{bcolors.BG_BLUE}{bcolors.YELLOW}Running trial {trial.number=} in {threading.current_thread().name}{bcolors.RESET}")
        config = deepcopy(config)
        # Suggest hyperparameters with Optuna
        # Optuna will suggest a learning rate and batch size for each trial
        # ---- LEARNING RATE (float, log scale) ----
        if is_range(config.optuna_lr):
            low, high = validate_range(config.optuna_lr, "optuna_lr")
            LEARNING_RATE = trial.suggest_float("Learning Rate", low, high, log=True)
        elif isinstance(config.optuna_lr, (float, int)):
            LEARNING_RATE = float(config.optuna_lr)
        else:
            raise TypeError("optuna_lr must be float or [low, high]")

        # ---- WEIGHT DECAY (float, log scale) ----
        if is_range(config.optuna_weight_decay):
            low, high = validate_range(config.optuna_weight_decay, "optuna_weight_decay")
            WEIGHT_DECAY = trial.suggest_float("Weight decay", low, high, log=True)
        elif isinstance(config.optuna_weight_decay, (float, int)):
            WEIGHT_DECAY = float(config.optuna_weight_decay)
        else:
            raise TypeError("optuna_weight_decay must be float or [low, high]")

        # ---- BATCH SIZE (int) ----
        if is_range(config.optuna_batch_size):
            low, high = validate_range(config.optuna_batch_size, "optuna_batch_size")
            BATCH_SIZE = trial.suggest_int("Batch size", int(low), int(high))
        elif isinstance(config.optuna_batch_size, int):
            BATCH_SIZE = config.optuna_batch_size
        else:
            raise TypeError("optuna_batch_size must be int or [low, high]")

        # ---- LATENT DIMENSIONS (int) ----
        if is_range(config.optuna_ndim):
            low, high = validate_range(config.optuna_ndim, "optuna_ndim")
            LATENT_DIMENSIONS = trial.suggest_int(
                "Dimensions", int(low), int(high)
            )
        elif isinstance(config.optuna_ndim, int):
            LATENT_DIMENSIONS = config.optuna_ndim
        else:
            raise TypeError("optuna_ndim must be int or [low, high]")

        # ---- BETA (float) ----
        if is_range(config.optuna_beta):
            low, high = validate_range(config.optuna_beta, "optuna_beta")
            BETA = trial.suggest_float("Beta", float(low), float(high))
        elif isinstance(config.optuna_beta, (float, int)):
            BETA = float(config.optuna_beta)
        else:
            raise TypeError("optuna_beta must be float or [low, high]")

        
        # ---- SUB_PERC (float) ----
        if is_range(config.optuna_sub_perc):
            low, high = validate_range(config.optuna_sub_perc, "optuna_sub_perc")
            SUB_PERC = trial.suggest_float("Percentage of subjects", float(low), float(high))
        elif isinstance(config.optuna_sub_perc, (float, int)):
            SUB_PERC = float(config.optuna_sub_perc)
        else:
            raise TypeError("optuna_sub_perc must be float or [low, high]")

        # ---- Assign back to config ----
        config.lr         = LEARNING_RATE
        config.batch_size = BATCH_SIZE
        config.n          = LATENT_DIMENSIONS
        config.kl         = BETA
        config.sub_perc   = SUB_PERC
        config.nb_epoch   = int(config.optuna_epoch)

        config.weight_decay = WEIGHT_DECAY

        # Configuration step
        config = process_config(config)
        torch.manual_seed(3)

        config.save_dir = config.save_dir + f"/{config.dataset_name}_dim_{config.n}_beta_{config.kl}_{now:%H-%M-%S}_trial_{trial.number}/"

        # Create the save directory
        try:
            os.makedirs(config.save_dir)
        except FileExistsError:
            print(f"Directory {config.save_dir} already exists")
            pass

        # Save config as a yaml file
        with open(config.save_dir + "/config.yaml", "w") as f:
            OmegaConf.save(config, f)
        
        print(""" Train model for given configuration """)
        final_loss_val = train_vae_optuna(config, trial,root_dir=config.save_dir)

        torch.cuda.empty_cache()
        return final_loss_val
    
    except optuna.TrialPruned:
        raise
    except RuntimeError as e:
        # Optional: prune on CUDA OOM
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory")
            raise optuna.TrialPruned()
        else:
            print(e)
            raise
    

def Run_optuna_optimization(config):
    # Here, we access the dataset directly in shared memory
    try:
        print('Run optimization')
        #storage_name = f"sqlite:///{config.optuna_folder}/study.db"
        study_name="journal_storage_multiprocess"
        storage_name = JournalStorage(JournalFileBackend(file_path=f"{config.optuna_folder}/journal.log"))

        if config.Anomaly == None:
            print(f'{bcolors.YELLOW}Minimizing Reconstruction Error{bcolors.RESET}')
            #pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
            #pruner = PatientPruner(MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1), patience=1, min_delta  = 100)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)

            sampler = optuna.samplers.TPESampler()
            study = optuna.create_study(study_name=study_name,directions=['minimize'],
                                        storage=storage_name,sampler=sampler,
                                        pruner=pruner,load_if_exists=True)
        else:
            print(f'{bcolors.YELLOW}Maximizing AUC{bcolors.RESET}')
            sampler = optuna.samplers.TPESampler()
            pruner = PatientPruner(wrapped_pruner=None, patience=3, min_delta  = 0.02)
            study = optuna.create_study(study_name=study_name,directions=['maximize'],
                                        storage=storage_name,sampler=sampler,
                                        pruner=pruner,load_if_exists=True)
        
        study.optimize(lambda trial: objective(trial,config), n_trials=config.optuna_ntrials)

    except optuna.TrialPruned:
        pass

    except Exception as e:
        print(f"Run optimization failed with error: {e}")

@hydra.main(config_name='config', version_base="1.1", config_path="configs")
def train(config):
    start_time = time.time()
    print(f'{bcolors.GREEN}{bcolors.UNDERLINE}Launching Optuna_tuning.py{bcolors.RESET}')
    print(f'{bcolors.YELLOW}Config:{config}{bcolors.RESET}')
    
    if not os.path.exists(config.optuna_folder):
        os.makedirs(config.optuna_folder,exist_ok=True)

    config.in_shape = adjust_in_shape(config)

    print(f'{bcolors.CYAN}~~~~~~ @ Running Optuna Framework @ ~~~~~~{bcolors.RESET}')
    #Run_optuna_optimization(config)
    nworkers = config.optuna_nworkers
    with Pool(max_workers=nworkers) as pool:
        pool.map(Run_optuna_optimization, [config]*nworkers)

    print("--- Optuna optimization finish in %s seconds ---" % (time.time() - start_time))
     
if __name__ == '__main__':
    train()


