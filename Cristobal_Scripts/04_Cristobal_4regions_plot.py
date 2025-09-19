import anatomist.api as ana
from soma.qt_gui.qtThread import QtThreadCall
a = ana.Anatomist()
from soma import aims
import pandas as pd
import numpy as np
import os
import shutil
dic_windows = {}
volumes = {}
nb_columns=4
block = a.createWindowsBlock(nb_columns)

#for model in ['LARGE_CINGULATE_right_UKB','FIP_right_UKB','SC-sylv_right_UKB','SC-sylv_left_UKB','SC-sylv_right_HCP','SC-sylv_left_HCP','SOr_left_UKB']:
#    for ndim in ['32','75','256']:


for model in ['LARGE_CINGULATE_right_UKB','FIP_right_UKB','SOr_left_UKB','SC-sylv_right_UKB','SC-sylv_left_UKB','SC-sylv_right_HCP','SC-sylv_left_HCP']:
#for model in ['LARGE_CINGULATE_right_UKB','FIP_right_UKB','SOr_left_UKB']:
#for model in ['SC-sylv_right_UKB','SC-sylv_left_UKB','SC-sylv_right_HCP','SC-sylv_left_HCP']:

    if model.split('_')[-1] == 'UKB':
        subject = '1000021'
    else:
        subject = '100206'
                

    out_path = '/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'  + model + '_32/subjects/'+subject

    val = '_input'

    vol = aims.read(out_path + val + '.nii.gz')

    ndim = '32'
    volumes[model+'_'+ndim+val] = a.toAObject(vol)

    dic_windows[model+'_'+ndim+val+'_vol'] = a.fusionObjects(objects=[volumes[model+'_'+ndim+val]], method='VolumeRenderingFusionMethod')

    dic_windows[model+'_'+ndim+val+'_rvol'] = a.createWindow('3D', block=block)
    dic_windows[model+'_'+ndim+val+'_rvol'].windowConfig(cursor_visibility=0)
    dic_windows[model+'_'+ndim+val+'_rvol'].addObjects(dic_windows[model+'_'+ndim+val+'_vol'])


    for ndim in ['32','75','256']: 

        if not os.path.exists('/home/cm283129/Documentos/'+model):
            os.mkdir('/home/cm283129/Documentos/'+model)

        shutil.copy('/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'  + model + '_' + ndim +'/Embeddings_'+model+'_ndim_'+ndim+'.csv','/home/cm283129/Documentos/'+model)

        out_path = '/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'  + model + '_' + ndim + '/subjects/'+subject

        val = '_output'

        vol = aims.read(out_path + val + '.nii.gz')

        volumes[model+'_'+ndim+val] = a.toAObject(vol)

        dic_windows[model+'_'+ndim+val+'_vol'] = a.fusionObjects(objects=[volumes[model+'_'+ndim+val]], method='VolumeRenderingFusionMethod')
        dic_windows[model+'_'+ndim+val+'_rvol'] = a.createWindow('3D', block=block)
        dic_windows[model+'_'+ndim+val+'_rvol'].windowConfig(cursor_visibility=0)
        dic_windows[model+'_'+ndim+val+'_rvol'].addObjects(dic_windows[model+'_'+ndim+val+'_vol'])
