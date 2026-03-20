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
from train import train_vae_model
from utils.config import process_config
from torch.utils.data import Subset, Dataset
from tqdm import tqdm
from subprocess import call
from hydra.utils import get_original_cwd
import matplotlib.pyplot as plt
#from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor as Pool
from functools import partial
from colors import bcolors

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

@hydra.main(config_name='config', version_base="1.1", config_path="configs")
def train(config):
    start_time = time.time()

    print(f'{bcolors.GREEN}{bcolors.UNDERLINE}Launching Train_full_model.py{bcolors.RESET}')
    print(f'{bcolors.YELLOW}Config:{config}{bcolors.RESET}')
    
    print("Load data and generate torch datasets within train")
    config.in_shape = adjust_in_shape(config)
    # Configuration step
    config = process_config(config)
    torch.manual_seed(3)
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
    
    print(""" Train model for given configuration """)
    train_vae_model(config, root_dir=config.save_dir)
    torch.cuda.empty_cache()

    print("--- Train finished in %s seconds ---" % (time.time() - start_time))
     

if __name__ == '__main__':
    train()


