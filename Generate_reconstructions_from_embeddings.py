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
from beta_vae import VAE
from tqdm import tqdm
import argparse
from torch.autograd import Variable
import numpy as np
# Initialize the parser
def create_parser():
    parser = argparse.ArgumentParser(description="Generate reconstructions from a list of embeddings")
    # Add arguments
    parser.add_argument('-i', '--input_embeddings', type=str, help="Input folder"    , required=True)
    parser.add_argument('-m', '--model_path'      , type=str, help="Model path"      , required=True)
    parser.add_argument('-n', '--n_dimensions', type=int, help="Model dimensions", required=True)
    parser.add_argument('-o', '--out_folder'      , type=str, help="Output folder"   , required=True)
    parser.add_argument('-s','--shape', type=int, nargs='+', help='Input shape')

    return parser

def main():
    # Create the parser and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Access the arguments
    in_embeddings = args.input_embeddings
    model_path    = args.model_path
    n             = args.n_dimensions
    out_folder    = args.out_folder
    in_shape      = args.shape
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    torch.manual_seed(0)
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"

    model = VAE(in_shape, n, 3,True)
    model.load_state_dict(torch.load(f'{model_path}/checkpoint.pt'))
    model = model.to(device)
    device = torch.device("cuda", index=0)
    model.eval()

    embeddings = torch.from_numpy(np.load(in_embeddings).astype(np.float32)).reshape(1, n)
    print(embeddings,embeddings.shape)
    with torch.no_grad():
        for row,z in enumerate(embeddings):
            z = z.reshape(1,n)
            z = Variable(z).to(device, dtype=torch.float32)
            print(z.shape)
            outputs = model.decode(z)
            np.save(f'{out_folder}/{row}.npy',outputs.cpu().numpy()[0,0,:,:,:])
            
if __name__ == '__main__':
    main()
