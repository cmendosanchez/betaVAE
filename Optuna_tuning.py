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
import sys
import hydra
import omegaconf
from omegaconf import OmegaConf
import time
import numpy as np
import pandas as pd
import json
import yaml
import itertools
import torch
import gc
from datetime import datetime
from train import train_vae_optuna
from utils.config import process_config
from torch.utils.data import Subset, Dataset
from tqdm import tqdm
from subprocess import call
from hydra.utils import get_original_cwd
import optuna
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import Trial
from optuna.samplers import RandomSampler
#from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial
from optuna.samplers import TPESampler

def adjust_in_shape(config):
    dims=[]
    for idx in range(1, 4):
        dim = config.in_shape[idx]
        r = dim%(2**config.depth)
        if r!=0:
            dim+=(2**config.depth-r)
        dims.append(dim)
    return((1, dims[0]+4, dims[1], dims[2]))


now = datetime.now()
optuna.logging.set_verbosity(optuna.logging.INFO)

# This will be the objective function for Optuna optimization
def objective(trial, config, dataset):
    try:
        print(f"Running trial {trial.number=} in process {os.getpid()}")
        # Suggest hyperparameters with Optuna
        # Optuna will suggest a learning rate and batch size for each trial
        #LEARNING_RATE      = trial.suggest_float('LEARNING_RATE', 1e-6,1e-4,log=True)
        LEARNING_RATE      = trial.suggest_categorical('LEARNING_RATE', [1e-5,1e-4])
        BATCH_SIZE         = trial.suggest_categorical('BATCH_SIZE', [16,32])
        N_EPOCH            = trial.suggest_categorical('N_EPOCH', [6,12,18,24,30]) 
        #N_EPOCH            = trial.suggest_categorical('N_EPOCH', [3])
        LATENT_DIMENSIONS  = trial.suggest_categorical('LATENT_DIMENSIONS', [32,64,128,256,512,1024])
        #LATENT_DIMENSIONS  = trial.suggest_categorical('LATENT_DIMENSIONS', [512])
        BETA               = trial.suggest_categorical('BETA', [1,2,4,8,16,32])
        #BETA               = trial.suggest_categorical('BETA', [87])
        N_SUBJECTS         = trial.suggest_categorical('N_SUBJECTS', [15000,25000,35000]) 
        #N_SUBJECTS         = trial.suggest_categorical('N_SUBJECTS', [5000]) 
        # Update config dynamically
        config.lr         = LEARNING_RATE
        config.batch_size = BATCH_SIZE
        config.nb_epoch   = N_EPOCH 
        config.n          = LATENT_DIMENSIONS
        config.kl         = BETA
        config.nsamples   = N_SUBJECTS

        # Configuration step
        config = process_config(config)
        torch.manual_seed(3)

        config.save_dir = config.save_dir + f"/{now:%Y-%m-%d}/{config.dataset_name}_dim_{config.n}_beta_{config.kl}_{now:%H-%M-%S}_trial_{trial.number}/"

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
        final_loss_val = train_vae_optuna(config, dataset, trial,root_dir=config.save_dir)

        torch.cuda.empty_cache()
        return final_loss_val
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned() 
    

def Run_optuna_optimization(config,path_crops,trials_per_worker):
    # Here, we access the dataset directly in shared memory
    try:
        print('Run optimization')
        #print(f'~~~ Arguments \nconfig:{config} \npath_crops:{path_crops}\ntrials_per_worker:{trials_per_worker}')
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
        #study = optuna.create_study(direction='minimize',sampler=TPESampler(),study_name="journal_storage_multiprocess",pruner=pruner,
        #                            storage=JournalStorage(JournalFileBackend(file_path=f"{config.optuna_folder}/journal_gpu_prio_12.log")),load_if_exists=True)
        study = optuna.create_study(direction='minimize',sampler=TPESampler(),study_name="example2_study",pruner=pruner,storage="mysql://gaia:Optima1Pass!@rosette:3306/example2",load_if_exists=True)

 
        study.optimize(lambda trial: objective(trial,config,path_crops), n_trials=trials_per_worker)
    except Exception as e:
        print(f"Run optimization failed with error: {e}")

@hydra.main(config_name='config', version_base="1.1", config_path="configs")
def train(config):
    start_time = time.time()

    PATH_CROPS_               = config.path_crops
    STUDY_FOLDER_             = config.optuna_folder
    OPTUNA_WORKERS_            = int(config.optuna_workers)
    OPTUNA_TRIALS_PER_WORKER_ = int(config.optuna_trials_per_worker)
    
    if not os.path.exists(STUDY_FOLDER_):
        os.makedirs(STUDY_FOLDER_,exist_ok=True)

    print(""" Load data and generate torch datasets within train """)
    config.in_shape = adjust_in_shape(config)

    print('~~~~~~ @ Running Optuna Framework @ ~~~~~~')
    with Pool(max_workers=OPTUNA_WORKERS_) as pool:
        pool.map(Run_optuna_optimization, [config]*OPTUNA_WORKERS_,[PATH_CROPS_]*OPTUNA_WORKERS_,[OPTUNA_TRIALS_PER_WORKER_]*OPTUNA_WORKERS_)

    #Run_optuna_optimization(config,PATH_CROPS_,OPTUNA_TRIALS_PER_WORKER_)
    print("--- Optuna optimization finish in %s seconds ---" % (time.time() - start_time))
    

if __name__ == '__main__':
    train()


