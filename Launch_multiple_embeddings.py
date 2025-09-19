import numpy as np
from subprocess import call
from subprocess import os
import yaml
import torch
import gc

gc.collect()

torch.cuda.empty_cache()

nifti_files =   [
                'two_ends_R_S.C.-sylv._Track_0_40_icbm09c.nii.gz'          ,
                'two_ends_R_S.C.-sylv._Track_0_40_sift2_icbm09c.nii.gz'    ,
                'two_ends_R_S.C.-sylv._Track_40_80_icbm09c.nii.gz'         ,
                'two_ends_R_S.C.-sylv._Track_40_80_sift2_icbm09c.nii.gz'   ,
                'two_ends_R_S.C.-sylv._Track_0_80_icbm09c.nii.gz'          ,
                'two_ends_R_S.C.-sylv._Track_0_80_sift2_icbm09c.nii.gz'    ]

nifti_files =   [
                'two_ends_R_S.C.-sylv._Track_0_80_icbm09c.nii.gz'          ,
                'two_ends_R_S.C.-sylv._Track_0_80_sift2_icbm09c.nii.gz'    ]

dims    = [75,128,256]
dims    = [75]
preprocs = ['Nothing','MinMax','LogMinMax']

model_folders = os.listdir('/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/2025-08-31')
for model in model_folders:
    #base = nifti.removesuffix('.nii.gz')
    #left = model.rsplit("_", 1)[0].split('/')[-1]
    _, dataset = model.split("_crops_", 1)   # split only once
    left, right = dataset.rsplit("_", 1)
    print('Dataset',left)
    #for dim in dims:
    #for preproc in preprocs:
    # Load YAML file
    with open(f'/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/2025-08-31/{model}/config.yaml', "r") as f:
        config = yaml.safe_load(f)
    print(config)
    cmd = f"python3 generate_embeddings.py n={config['n']} +dataset_localization=neurospin_CorticalConnectivity +dataset=CorticalConnectivity/{left} +MSE_loss=True +binary=False +split=Random +preproc={config['preproc']} +test_model_dir=/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/2025-08-31/{model}"
    call([cmd],shell=True) 
    