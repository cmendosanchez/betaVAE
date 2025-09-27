from soma import aims
import numpy as np
import pickle 
import pandas as pd
from subprocess import os
#data = pd.read_pickle('/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/2mm/S.C.-sylv./mask/Rskeleton.pkl')
#data = np.load('/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/crops/2mm/S.C.-sylv./mask/Rskeleton.npy')

#print(data.shape)
'''
out_paths = os.listdir('/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/2025-08-30')
for path in out_paths:
    out_path = f"/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/2025-08-30/{path}/"
    print(out_path)
    try:
        for vol in ['_input','_output']:
            files = [f for f in os.listdir(out_path+'subjects') if f.endswith(vol+'.npy')]
            print('files:', files)
            for sub in files:
                print(sub)
                
                print('Split',out_path+'subjects/'+sub)
                vol_npy = np.load(out_path+'subjects/'+sub).astype(np.float32)
                #print('Volume shape',vol_npy.shape)
                vol_nifty = aims.Volume(vol_npy)
                vol_nifty.header()['voxel_size'] = [2.0, 2.0, 2.0]
                aims.write(vol_nifty, out_path+'subjects/'+sub.split('.npy')[0]+'.nii.gz')
                #call([f'rm -rf '+out_path+'subjects/'+sub+' -v'],shell=True)
    except:
        print('except')
'''

out_path = f"/neurospin/dico/cmendoza/Runs/14_PhD_UKB/JeanZay/R_S.C.-sylv./full_brain_R_S.C.-sylv._Track_0_250_sift2_icbm09c_dim_64_beta_1_14-07-42/"
print(out_path)
try:
    for vol in ['_input','_output']:
        files = [f for f in os.listdir(out_path+'subjects') if f.endswith(vol+'.npy')]
        print('files:', files)
        for sub in files:
            print(sub)
            
            print('Split',out_path+'subjects/'+sub)
            vol_npy = np.load(out_path+'subjects/'+sub).astype(np.float32)
            #print('Volume shape',vol_npy.shape)
            vol_nifty = aims.Volume(vol_npy)
            vol_nifty.header()['voxel_size'] = [1.0, 1.0, 1.0]
            aims.write(vol_nifty, out_path+'subjects/'+sub.split('.npy')[0]+'.nii.gz')
            #call([f'rm -rf '+out_path+'subjects/'+sub+' -v'],shell=True)
except:
    print('except')