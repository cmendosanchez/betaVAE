# -*- coding: utf-8 -*-
# /usr/bin/env python3
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
import pandas as pd
import torch
import torch.nn as nn
from beta_vae import VAE, ModelReconstruct
from load_data import create_subset
import hydra
from utils.config import process_config
from tqdm import tqdm
from torch.utils.data import Subset, Dataset


def adjust_in_shape(config):

    dims=[]
    for idx in range(1, 4):
        dim = config.in_shape[idx]
        r = dim%(2**config.depth)
        if r!=0:
            dim+=(2**config.depth-r)
        dims.append(dim)
    return((1, dims[0]+4, dims[1], dims[2]))

def split_set(dataset, ids):
    ids = set(ids)
    indices = [i for i in tqdm(range(len(dataset)), desc="Building decoding split") if dataset[i][1] in ids]
    print('split_dataset',dataset[0][1])
    train_set = Subset(dataset, indices)
    return train_set

@hydra.main(config_name='config', config_path="configs")
def main(config):
    """
    Infer a trained model on test data and saves the embeddings as csv
    """
    print('Infer a trained model on test data and saves the embeddings as csv')
    config=process_config(config)
    print(config)
    torch.manual_seed(0)
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)

    config.in_shape = adjust_in_shape(config)
    print('in_shape:',adjust_in_shape)
    print('""" Model',os.path.join(config.test_model_dir, 'checkpoint.pt'),'"""')
    model_dir = os.path.join(config.test_model_dir, 'checkpoint.pt') 
    model = VAE(config.in_shape, config.n, depth=config.depth, Use_MSE = config.MSE_loss)
    model.load_state_dict(torch.load(model_dir))
    model = model.to(device)
    print(config)

    subset_test = create_subset(config)
    decode_ids = pd.read_csv(config.decode_subs_list, dtype=str)['ID'].values.tolist()
    print(decode_ids)
    if decode_ids[0][:4]=='sub-':
        decode_ids_cleaned = [s.removeprefix("sub-") for s in decode_ids]
    else:
        decode_ids_cleaned = decode_ids
    print(decode_ids_cleaned)
    decode_set= split_set(subset_test, decode_ids_cleaned)
    print('Nsubjects to decode:',len(decode_set))

    testloader = torch.utils.data.DataLoader(
              decode_set,
              batch_size=1,
              num_workers=8,
              shuffle=False) #Shuffle was set to True?
    dico_set_loaders = {'test': testloader}

    reconstruct = ModelReconstruct(model=model, dico_set_loaders=dico_set_loaders,kl_weight=config.kl, n_latent=config.n, depth=config.depth,save_dir=config.test_model_dir,dataset_name = config.dataset_name,selected_loss=config.loss)
    reconstruct.decode()

if __name__ == '__main__':
    main()
