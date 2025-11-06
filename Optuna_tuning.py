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
from train import train_vae, train_vae_optuna
from load_data import create_subset
from utils.config import process_config
from torch.utils.data import Subset, Dataset
from tqdm import tqdm
from subprocess import call
from hydra.utils import get_original_cwd
import optuna
import optuna.visualization as vis
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import threading
from optuna.trial import Trial
import joblib

now = datetime.now()
optuna.logging.set_verbosity(optuna.logging.INFO)
'''
FilteredDataset for CustomDatasets
'''

def split_by_ids(dataset, train_ids, val_ids):
    train_ids = set(train_ids)
    val_ids = set(val_ids)

    train_indices = [i for i in tqdm(range(len(dataset)), desc="Building train split") if dataset[i][1] in train_ids]
    val_indices   = [i for i in tqdm(range(len(dataset)), desc="Building validation split") if dataset[i][1] in val_ids]

    train_set = Subset(dataset, train_indices)
    val_set   = Subset(dataset, val_indices)
    return train_set, val_set

class FilteredDataset(Dataset):
    def __init__(self, data, keep_ids):
        """
        data: list of (numpy_array, id) or a dataset yielding (numpy_array, id)
        keep_ids: set/list of IDs to keep
        """
        self.samples = []
        self.keep_ids = set(keep_ids)

        # iterate through the data and filter
        for x, sid in tqdm(data,'Loading data'):
            if sid in self.keep_ids:
                self.samples.append((x, sid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def adjust_in_shape(config):
    dims=[]
    for idx in range(1, 4):
        dim = config.in_shape[idx]
        r = dim%(2**config.depth)
        if r!=0:
            dim+=(2**config.depth-r)
        dims.append(dim)
    return((1, dims[0]+4, dims[1], dims[2]))

# This will be the objective function for Optuna optimization
def objective(trial: Trial, config, dataset):
    try:
        print(f"Running trial {trial.number=} in {threading.current_thread().name}")
        # Suggest hyperparameters with Optuna
        # Optuna will suggest a learning rate and batch size for each trial
        LEARNING_RATE = trial.suggest_categorical('LEARNING_RATE', [1e-5,1e-4,1e-3,1e-2])
        BATCH_SIZE    = trial.suggest_categorical('BATCH_SIZE', [16,32,64])
        #EPOCH_OPTUNA         = trial.suggest_int('EPOCH_OPTUNA', 10, 60, step=10)
        N_EPOCH         = trial.suggest_int('N_EPOCH', 3, 6, step=1)
        LATENT_DIMENSIONS             = trial.suggest_categorical('LATENT_DIMENSIONS', [32,64,128,256,512])
        #N_OPTUNA             = trial.suggest_categorical('N_OPTUNA', [32,64])
        BETA          = trial.suggest_categorical('BETA', [1,2,4,8,16,32,64])
        N_SUBJECTS    = trial.suggest_int('N_SUBJECTS', 1000, 3000, step=100)

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

        # Create a subset dataset
        subset1 = Subset(dataset, list(range(0, config.nsamples)))
        print('Objective fun - subset size',len(subset1),subset1[0][0].shape)

        if config.train_list is not None:
            filename_train = os.path.basename(config.train_list)
            config.save_dir = config.save_dir + f"/{now:%Y-%m-%d}/{config.dataset_name}_dim_{config.n}_beta_{config.kl}_{os.path.splitext(filename_train)[0]}_{now:%H-%M-%S}/"
        else:
            config.save_dir = config.save_dir + f"/{now:%Y-%m-%d}/{config.dataset_name}_dim_{config.n}_beta_{config.kl}_{now:%H-%M-%S}/"

        # Create the save directory
        try:
            os.makedirs(config.save_dir)
        except FileExistsError:
            print(f"Directory {config.save_dir} already exists")
            pass

        # Save config as a yaml file
        with open(config.save_dir + "/config.yaml", "w") as f:
            OmegaConf.save(config, f)
        
        print(""" Load data and generate torch datasets """)
        #subset1 = create_subset(config)
        proportion_test = 0.8
        proportion_validation = 0.2

        if config.split == 'RandomSplit':
            print('Random Split')
            train_set, val_set = torch.utils.data.random_split(subset1,
                                [round(proportion_test * len(subset1)), round(proportion_validation * len(subset1))])
        elif config.split == 'CustomSplit':
            print('Custom Split')
            train_ids = pd.read_csv(config.train_list)['Subject'].values.tolist()
            train_ids_cleaned = [s.removeprefix("sub-") for s in train_ids]
            validation_ids = pd.read_csv(config.validation_list)['Subject'].values.tolist()
            validation_ids_cleaned = [s.removeprefix("sub-") for s in validation_ids]
            train_set, val_set = split_by_ids(subset1, train_ids_cleaned, validation_ids_cleaned)
        
        print(f'Nsubjects Train: {len(train_set)}, Nsubjects Validation: {len(val_set)}')
        
        # DataLoader for training and validation
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, num_workers=8, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=8, shuffle=False)

        print(""" Train model for given configuration """)
        final_loss_val = train_vae_optuna(config, trainloader, valloader, trial,root_dir=config.save_dir)

        # Clean up
        del trainloader, valloader, subset1
        torch.cuda.empty_cache()
        return final_loss_val
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned() 


@hydra.main(config_name='config', version_base="1.1", config_path="configs")
def train(config):
    start_time = time.time()
    out_plots = f'/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/OptunaResults/Optuna_{now:%Y-%m-%d}_{now:%H-%M-%S}/'
    if not os.path.exists(out_plots):
        os.makedirs(out_plots)
    print(""" Load data and generate torch datasets within train """)
    subset1 = create_subset(config)
    print('Set',subset1,len(subset1))
    config.in_shape = adjust_in_shape(config)
    #Create Optuna pruner
    print('~~~~~~ @ Running Optuna Framework @ ~~~~~~')
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
    study = optuna.create_study(direction='minimize',study_name=f"betaVAE_{now:%Y-%m-%d}_{now:%H-%M-%S}",pruner=pruner)
    # Objective function is a wrapped version of the training function
    study.optimize(lambda trial: objective(trial,config,subset1), n_trials=4, n_jobs=2)  # 10 trials

    print('~~~~ Plotting Results ~~~~')
    # Plot optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.write_image(f"{out_plots}/optimization_history.png",scale=3)

    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_image(f"{out_plots}/param_importances.png",scale=3)

    fig3 = optuna.visualization.plot_intermediate_values(study)
    fig3.write_image(f"{out_plots}/intermediate_values.png",scale=3)

    fig4 = optuna.visualization.plot_timeline(study)
    fig4.write_image(f"{out_plots}/timelines.png",scale=3)

    fig5 = optuna.visualization.plot_contour(study, params=["N_DIMENSIONS","BETA"])
    fig5.write_image(f"{out_plots}/contour_plot_NDIM_BETA.png",scale=3)

    # Print the best trial details
    print(f"Best trial: {study.best_trial}")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    print(f'Saving study to {out_plots}')
    joblib.dump(study, f"{out_plots}/optuna_study_batch.pkl")   # save study

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    train()