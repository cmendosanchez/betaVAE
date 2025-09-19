import yaml
import os
from tqdm import tqdm

from yaml.representer import SafeRepresenter

class MyDumper(yaml.SafeDumper):
    pass

# Force lists to be inline only for in_shape
def represent_inline_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# Attach the custom representer only to lists of length 4 (or you can check for your specific key)
MyDumper.add_representer(list, represent_inline_list)

Hemi   = f'R'
Region = f'S.C.-sylv.'
nifti_files =   []
for conn in ['','_sift2']:
    for mode in ['two_ends','one_end','full_brain','CC']:
        if   mode =='two_ends':
            l_ranges = [('0','40'),('40','80'),('0','80')]
        elif mode =='one_end':
            l_ranges = [('0','40'),('40','80'),('0','80'),('80','250'),('0','250')]
        else:
            l_ranges = [('0','250')]

        for l_range in l_ranges:
            nifti_files.append(f'{mode}_{Hemi}_{Region}_Track_{l_range[0]}_{l_range[1]}{conn}_icbm09c.nii.gz')

yaml_folder = '/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE/configs/dataset/PhD_UKB'

for nifti in tqdm(nifti_files, desc="Processing config files"):
    base = nifti.removesuffix('.nii.gz')
    dataset_name = base
    config = {
        "dataset_name"      : f'{base}',
        "data_dir"          : f'${{dataset_folder}}/{Hemi}_{Region}_crops_{base}.npy',
        "subject_dir"       : f'${{dataset_folder}}/{Hemi}_{Region}_subjects.csv',
        "in_shape"          : [1, 84, 68, 98],
        "subjects_to_remove": None,
        "remove_subjects"   : False,
        "train_list"        : None, 
        "validation_list"   : None}
    with open(f"{yaml_folder}/{base}.yaml", "w") as f:
        f.write("# @package _global_\n")
        yaml.dump(config, f, sort_keys=False,  Dumper=MyDumper)