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
from datetime import datetime
now = datetime.now()
from train import train_vae
from load_data import create_subset
from utils.config import process_config
from torch.utils.data import Subset, Dataset
from tqdm import tqdm

'''
FilteredDataset for CustomDatasets
'''
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

@hydra.main(config_name='config', version_base="1.1", config_path="configs")
def train_model(config):

    #Configuration step
    config=process_config(config)
    torch.manual_seed(3)
    config.save_dir = config.save_dir + f"/{now:%Y-%m-%d}/{config.dataset_name}_dim_{config.n}_beta_{config.kl}_{now:%H-%M-%S}/"
    config.in_shape = adjust_in_shape(config)
    print(config)

    #Create the save dir
    try:
        os.makedirs(config.save_dir)
    except FileExistsError:
        print("Directory " , config.save_dir ,  " already exists")
        pass

    #Save config as a yaml file
    with open(config.save_dir+"/config.yaml", "w") as f:
        OmegaConf.save(config, f)

    print(""" Load data and generate torch datasets """)
    subset1 = create_subset(config)
    proportion_test = 0.8
    proportion_validation = 0.2
    if config.split == 'RandomSplit':
        print('""" Random Split """')
        train_set, val_set = torch.utils.data.random_split(subset1,
                                [round(proportion_test*len(subset1)), round(proportion_validation*len(subset1))])

    elif config.split == 'CustomSplit':
        print('""" CustomSplit Split """')
        train_ids = pd.read_csv(config.train_list)['Subject'].values.tolist()
        train_ids_cleaned = [s.removeprefix("sub-") for s in train_ids]
        validation_ids = pd.read_csv(config.validation_list)['Subject'].values.tolist()
        validation_ids_cleaned = [s.removeprefix("sub-") for s in validation_ids]
        #print(train_ids)
        train_set = FilteredDataset(subset1, train_ids_cleaned)
        val_set = FilteredDataset(subset1, validation_ids_cleaned)
    print('Nsubjects Train:',len(train_set),'Nsubjects Validation:',len(val_set),'Nsubjects total (Train + Validation):',len(train_set)+len(val_set))

    #Train and Validation split
    trainloader = torch.utils.data.DataLoader(
                  train_set,
                  batch_size=config.batch_size,
                  num_workers=8,
                  shuffle=True)
    valloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=1,
                num_workers=8,
                shuffle=False)

    #Save subjects ID to csv
    print('Generating Train and Validation .csv files',)
    train_label = []
    for _, path in trainloader:
        for sub_for_train in path:
            train_label.append(sub_for_train)
    np.savetxt(f"{config.save_dir}/train_label.csv", np.array(train_label), delimiter =", ", fmt ='% s')

    val_label = []
    for _, path in valloader:
        val_label.append(path[0])
    np.savetxt(f"{config.save_dir}/val_label.csv", np.array(val_label), delimiter =", ", fmt ='% s')
    print('Done')

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    print(""" Train model for given configuration """)
    vae, final_loss_val = train_vae(config, trainloader, valloader,
                                    root_dir=config.save_dir)


if __name__ == '__main__':
    start_time = time.time()
    train_model()
    print("--- %s seconds ---" % (time.time() - start_time))

