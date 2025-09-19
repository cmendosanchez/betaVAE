import numpy as np
from subprocess import call

nifti_files =   [
                'two_ends_R_S.C.-sylv._Track_0_40_icbm09c.nii.gz'          ,
                'two_ends_R_S.C.-sylv._Track_0_40_sift2_icbm09c.nii.gz'    ,
                'two_ends_R_S.C.-sylv._Track_40_80_icbm09c.nii.gz'         ,
                'two_ends_R_S.C.-sylv._Track_40_80_sift2_icbm09c.nii.gz'   ,
                'two_ends_R_S.C.-sylv._Track_0_80_icbm09c.nii.gz'          ,
                'two_ends_R_S.C.-sylv._Track_0_80_sift2_icbm09c.nii.gz'    ]

#nifti_files =   [
#                'two_ends_R_S.C.-sylv._Track_0_80_icbm09c.nii.gz'          ,
#                'two_ends_R_S.C.-sylv._Track_0_80_sift2_icbm09c.nii.gz'    ]

dims    = [75,128]
preprocs = ['LogMinMax']
for nifti in nifti_files:
    base = nifti.removesuffix('.nii.gz')
    for dim in dims:
        for preproc in preprocs:
            cmd = f"python3 main.py n={dim} +dataset_localization=neurospin_CorticalConnectivity +dataset=CorticalConnectivity/{base} +MSE_loss=True +binary=False +split=Random ++remove_subjects=True +preproc={preproc}"
            call([cmd],shell=True)