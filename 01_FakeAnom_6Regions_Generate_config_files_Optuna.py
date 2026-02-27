import yaml
import os
from tqdm import tqdm
from colors import bcolors
from yaml.representer import SafeRepresenter
import nibabel as nib
class MyDumper(yaml.SafeDumper):
    pass

# Force lists to be inline only for in_shape
def represent_inline_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# Attach the custom representer only to lists of length 4 (or you can check for your specific key)
MyDumper.add_representer(list, represent_inline_list)

region_list = ["S.C.-sylv._left",
                "S.C.-sylv._right",
                "S.T.s._left",
                "S.T.s._right",
                "S.F.int.-F.C.M.ant._left",
                "S.F.int.-F.C.M.ant._right"]

modes = ['SWM','DWM','Comm']
databases = ['UKB']
Anomaly = ['Underconnectivity','Overconnectivity']
Path_sulci_masks = '/neurospin/dico/cmendoza/Runs/17_PhD_2026/Output/mask_skeleton'
yaml_folder = f'/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE/configs/dataset/Train_FakeAnom_{len(region_list)}Regions'
if not os .path.exists(yaml_folder):
    os.mkdir(yaml_folder)

for database in databases:
    for Anom in Anomaly:
        for region in region_list:
            if 'left' in region:
                hemi='L'
            elif 'right' in region:
                hemi='R'
            dims = nib.load(f'{Path_sulci_masks}/{region.replace('_left','').replace('_right','')}/{hemi}mask_skeleton_1mm_crop.nii.gz').get_fdata().shape
            for mode in modes:
                if mode == 'SWM':
                    minl=0
                    maxl=80
                elif mode == 'DWM':
                    minl= 80
                    maxl= 250
                elif mode == 'Comm':
                    minl = 0
                    maxl = 250


                print(f'{bcolors.CYAN}Creating Dataset for {database} {Anom} {region} {mode}{bcolors.RESET}')
                dataset_name = f'{database}_{region}_{mode}_{Anom}'
                config = {
                    "dataset_name"      : dataset_name,
                    "data_dir"          : None,
                    "subject_dir"       : None,
                    "in_shape"          : [1,dims[0],dims[1],dims[2]],
                    "Train_list"        : '${dataset_folder}/DataSplit_DL/train.tsv', 
                    "Rcon_val_list"     : '${dataset_folder}/DataSplit_DL/val_rconerror.tsv',
                    "Class_val_list"    : '${dataset_folder}/DataSplit_DL/val_anorm.tsv',
                    "Anomaly"           : Anom,
                    "Criteria"          : mode,
                    "Region"            : region,
                    "Database"          : database,
                    "path_crops"        : f'${{dataset_folder}}/crops/{region}/{mode}',
                    "path_anom"         : f'${{dataset_folder}}/FakeAnomaly_crops/{database}/{region}/{mode}/{Anom}',
                    "minl"              : minl,
                    "maxl"              : maxl,
                    "referential"       : "icbm09c"   
                    }
                with open(f"{yaml_folder}/{dataset_name}.yaml", "w") as f:
                    f.write("# @package _global_\n")
                    yaml.dump(config, f, sort_keys=False,  Dumper=MyDumper)