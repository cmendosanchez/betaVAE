from soma import aims
import numpy as np
import pickle 
import pandas as pd
import os
import glob
from subprocess import call
#data = pd.read_pickle('/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/2mm/S.C.-sylv./mask/Rskeleton.pkl')
#data = np.load('/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/crops/2mm/S.C.-sylv./mask/Rskeleton.npy')

#print(data.shape)
'''
models_folder = '/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'
models = os.listdir(models_folder)

#for model in ['LARGE_CINGULATE_right_UKB','FIP_right_UKB','SC-sylv_right_UKB','SC-sylv_left_UKB','SC-sylv_right_HCP','SC-sylv_left_HCP','SOr_left_UKB']:
#    for ndim in ['32','75','256']:
for model in ['LARGE_CINGULATE_right_UKB','FIP_right_UKB','SC-sylv_right_UKB','SC-sylv_left_UKB','SC-sylv_right_HCP','SC-sylv_left_HCP','SOr_left_UKB','ACCpatterns_LARGE_CINGULATE_RIGHT','FIP_right_HCP','SOr_left_HCP']:
    break
    for ndim in ['32','75','256']:
        print(model,ndim)

        out_path = '/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'  + model + '_' + ndim + '/'

        #try:
        
        try:
            
            for vol in ['input','output']:
                vol_npy = np.load(out_path+vol+'.npy')[0,0,:,:,:].astype(np.float32)
                print('Volume shape',vol_npy.shape)
                vol_nifty = aims.Volume(vol_npy)
                aims.write(vol_nifty, out_path+vol+'.nii.gz')

            files = [f for f in os.listdir(out_path+'subjects') if f.endswith('.npy')]
            print(files)
            for sub in files:
                for vol in ['_input','_output']:
                    sub_split = sub.split('.')
                    print('Split',sub_split)
                    vol_npy = np.load(out_path+'subjects/'+sub_split[0]+'.npy').astype(np.float32)
                    print('Volume shape',vol_npy.shape)
                    vol_nifty = aims.Volume(vol_npy)
                    vol_nifty.header()['voxel_size'] = [2.0, 2.0, 2.0]
                    aims.write(vol_nifty, out_path+'subjects/'+sub_split[0]+'.nii.gz')
        except:
            files = [f for f in os.listdir(out_path+'subjects') if f.endswith('.npy')]
            print(files)
            for sub in files:
                for vol in ['_input','_output']:
                    sub_split = sub.split('.')
                    print('Split',sub_split)
                    vol_npy = np.load(out_path+'subjects/'+sub_split[0]+'.npy').astype(np.float32)
                    #print('Volume shape',vol_npy.shape)
                    vol_nifty = aims.Volume(vol_npy)
                    vol_nifty.header()['voxel_size'] = [2.0, 2.0, 2.0]
                    aims.write(vol_nifty, out_path+'subjects/'+sub_split[0]+'.nii.gz')

'''


out_path = '/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/2025-08-31/R_S.C.-sylv_crops_two_ends_R_S.C.-sylv._Track_0_40_icbm09c_11-56-43/'
for vol in ['input','output']:
    vol_npy = np.load(out_path+vol+'.npy')[0,0,:,:,:].astype(np.float32)
    print('Volume shape',vol_npy.shape)
    vol_nifty = aims.Volume(vol_npy)
    aims.write(vol_nifty, out_path+vol+'.nii.gz')


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
            call([f'rm -rf '+out_path+'subjects/'+sub+' -v'],shell=True)
except:
    print('except')