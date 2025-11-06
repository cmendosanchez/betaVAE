import argparse
import os
import numpy as np
from soma import aims
from subprocess import call
from nilearn.surface import vol_to_surf
from nilearn import surface, image
import pandas as pd
# Initialize the parser
def create_parser():
    parser = argparse.ArgumentParser(description="Generate nifti files from folder with numpy")
    
    # Add arguments
    parser.add_argument('-i', '--in_folder', type=str, help="Input folder", required=True)
    parser.add_argument('-o', '--out_folder', type=str, help="Output folder", required=True)
    parser.add_argument('-th', '--threshold', type=float, help="Threshold folder", required=True)
    return parser


def main():
    # Create the parser and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Access the arguments
    in_folder = args.in_folder
    out_folder = args.out_folder
    threshold = args.threshold
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    files = [f for f in os.listdir(in_folder) if f.endswith('.npy')]
    for file in files:  
        prefix = file.split('.npy')[0]
        vol_npy = np.load(f'{in_folder}/{file}')
        vol_npy = np.where(vol_npy>threshold,vol_npy,0)
        vol_nifty = aims.Volume(vol_npy)
        vol_nifty.header()['voxel_size'] = [1.0, 1.0, 1.0]
        aims.write(vol_nifty, f'{out_folder}/{prefix}_{threshold}.nii.gz')
        if not os.path.exists(f'{out_folder}/tmp_mesh'):
            os.mkdir(f'{out_folder}/tmp_mesh')
        if not os.path.exists(f'{out_folder}/tmp_thres'):
            os.mkdir(f'{out_folder}/tmp_thres')
        call([f'AimsThreshold -i {out_folder}/{prefix}_{threshold}.nii.gz             -o {out_folder}/tmp_thres/{prefix}_{threshold}_binary.nii.gz -b -m gt -t 0 --fg 1 --verbose 1'],shell=True)
        call([f'AimsMesh      -i {out_folder}/tmp_thres/{prefix}_{threshold}_binary.nii -o {out_folder}/tmp_mesh/{prefix}_{threshold}.mesh --smooth True --smoothIt 100']      ,shell=True)
        call([f'AimsZCat      -i {out_folder}/tmp_mesh/*.mesh -o {out_folder}/{prefix}_{threshold}.mesh                           ']                    ,shell=True)
        call([f'AimsVol2Tex   -i {out_folder}/{prefix}_{threshold}.nii -m {out_folder}/{prefix}_{threshold}.mesh -o {out_folder}/{prefix}_{threshold}.tex -v 1 -height 4 -radius 3 -mode 1'],shell=True)
        call([f'rm -rfv          {out_folder}/tmp_mesh' ]                                        ,shell=True)
        call([f'rm -rfv          {out_folder}/tmp_thres']                                        ,shell=True)


if __name__ == "__main__":
    main()
