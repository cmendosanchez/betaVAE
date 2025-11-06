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
    parser.add_argument('-l', '--list_subjects', type=str, help="List of subject", required=True)
    parser.add_argument('-o', '--out_folder', type=str, help="Output folder", required=True)
    parser.add_argument('-th', '--threshold', type=float, help="Threshold", required=True)
    return parser


def main():
    # Create the parser and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Access the arguments
    in_folder = args.in_folder
    Subjects_path = args.list_subjects
    out_folder = args.out_folder
    threshold = args.threshold

    decode_ids = pd.read_csv(Subjects_path, dtype=str)['ID'].values.tolist()
    print(decode_ids)
    if decode_ids[0][:4]=='sub-':
        decode_ids_cleaned = [s.removeprefix("sub-") for s in decode_ids]
    else:
        decode_ids_cleaned = decode_ids
    print(decode_ids_cleaned)

    

    for sub in decode_ids_cleaned:
        out_folder_sub = f'{out_folder}/{sub}_{threshold}'

        if not os.path.exists(out_folder_sub):
            os.makedirs(out_folder_sub)

        for vol in ['input','output','additions','omissions']:
            vol_npy = np.load(f'{in_folder}/{sub}_{vol}.npy')
            vol_npy = np.where(vol_npy>threshold,vol_npy,0)
            vol_nifty = aims.Volume(vol_npy)
            vol_nifty.header()['voxel_size'] = [1.0, 1.0, 1.0]
            aims.write(vol_nifty, f'{out_folder_sub}/{sub}_{vol}.nii.gz')
            
            if vol == 'input' or vol=='output':
                if not os.path.exists(f'{out_folder_sub}/tmp_mesh'):
                    os.mkdir(f'{out_folder_sub}/tmp_mesh')
                if not os.path.exists(f'{out_folder_sub}/tmp_thres'):
                    os.mkdir(f'{out_folder_sub}/tmp_thres')

                call([f'AimsThreshold -i {out_folder_sub}/{sub}_{vol}.nii.gz             -o {out_folder_sub}/tmp_thres/{sub}_{vol}_thres.nii.gz -b -m gt -t 0 --fg 1 --verbose 1'],shell=True)
                call([f'AimsMesh      -i {out_folder_sub}/tmp_thres/{sub}_{vol}_thres.nii -o {out_folder_sub}/tmp_mesh/{sub}_{vol}.mesh --smooth True --smoothIt 100']      ,shell=True)
                call([f'AimsZCat      -i {out_folder_sub}/tmp_mesh/*.mesh -o {out_folder_sub}/{sub}_{vol}.mesh                           ']                    ,shell=True)
                call([f'AimsVol2Tex   -i {out_folder_sub}/{sub}_{vol}.nii -m {out_folder_sub}/{sub}_{vol}.mesh -o {out_folder_sub}/{sub}_{vol}.tex -v 1 -height 4 -radius 3 -mode 1'],shell=True)
                call([f'rm -rfv          {out_folder_sub}/tmp_mesh' ]                                        ,shell=True)
                call([f'rm -rfv          {out_folder_sub}/tmp_thres']                                        ,shell=True)
            elif vol=='additions':
                call([f'AimsVol2Tex   -i {out_folder_sub}/{sub}_{vol}.nii -m {out_folder_sub}/{sub}_output.mesh -o {out_folder_sub}/{sub}_{vol}.tex -v 1 -height 4 -radius 3 -mode 1'],shell=True)
            elif vol=='omissions':
                call([f'AimsVol2Tex   -i {out_folder_sub}/{sub}_{vol}.nii -m {out_folder_sub}/{sub}_input.mesh -o {out_folder_sub}/{sub}_{vol}.tex -v 1 -height 4 -radius 3 -mode 1'],shell=True)
        break
if __name__ == "__main__":
    main()
