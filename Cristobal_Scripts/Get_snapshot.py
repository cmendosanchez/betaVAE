import os
from subprocess import call
from joblib import Parallel, delayed
import anatomist.api as anatomist
import anatomist.api as ana
from soma import aims
import numpy as np
region  = f'S.C.-sylv.'
hemi = f'R' 
res  = f'2mm'


def Process_mesh(region,res,sub,idx,hemi):
    print('Sub',sub)
    ROI_folder = f'/neurospin/tmp/cmendoza/UKB_processing/{res}/{region}/'

    if os.path.exists(f'{ROI_folder}QualityCheck_bv/{sub}/SWM/Sulci_mesh.mesh'):
        print('Mesh exists')
        return

    if not os.path.exists(f'{ROI_folder}/QualityCheck_bv/{sub}/SWM'):
        os.makedirs(f'{ROI_folder}/QualityCheck_bv/{sub}/SWM')

    if not os.path.exists(f'{ROI_folder}/QualityCheck_bv/{sub}/SWM/mesh_tmp'):
        os.makedirs(f'{ROI_folder}/QualityCheck_bv/{sub}/SWM/mesh_tmp')

    Sulci_crop   = f'/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/crops/{res}/{region}/mask/{hemi}crops/{sub}_cropped_skeleton.nii.gz' 
    if os.path.exists(Sulci_crop):
        print('Yes')
    else:
        print('No')
    Sulci_crop_thresholded = f'/neurospin/tmp/cmendoza/UKB_processing/{res}/{region}/crops/{sub}/Sulci_thresholded.nii.gz'
    Sulci_crop_mesh = f'{ROI_folder}/QualityCheck_bv/{sub}/SWM/mesh_tmp/Sulci_mesh.mesh'
    Sulci_crop_mesh_final = f'{ROI_folder}/QualityCheck_bv/{sub}/SWM/Sulci_mesh.mesh'

    call([f'AimsThreshold -i {Sulci_crop} -o {Sulci_crop_thresholded} -b -m ge -t 1 --verbose 1'],shell=True)
    call([f'AimsMesh -i {Sulci_crop_thresholded} -o {Sulci_crop_mesh}'],shell=True)
    call([f'AimsZCat  -i {ROI_folder}/QualityCheck_bv/{sub}/SWM/mesh_tmp/*.mesh -o {Sulci_crop_mesh_final}'],shell=True)
    call([f'rm -rfv {ROI_folder}QualityCheck_bv/{sub}/SWM/mesh_tmp'],shell=True)

def build_gradient(pal):
    """Builds a gradient palette."""
    gw = ana.cpp.GradientWidget(
        None, 'gradientwidget',
        pal.header()['palette_gradients'])
    gw.setHasAlpha(True)
    nc = pal.shape[0]
    rgbp = gw.fillGradient(nc, True)
    rgb = rgbp.data()
    npal = pal.np['v']
    pb = np.frombuffer(rgb, dtype=np.uint8).reshape((nc, 4))
    npal[:, 0, 0, 0, :] = pb
    npal[:, 0, 0, 0, :3] = npal[:, 0, 0, 0,
                                :3][:, ::-1]  # Convert BGRA to RGBA
    pal.update()


def QualityCheck(model_path, region,res,sub,idx,hemi,pal,anat = anatomist.Anatomist()):

    if not os.path.exists(f'{model_path}/snapshots/'):
        os.makedirs(f'{model_path}/snapshots/')

    print(idx,sub)
    window = {}
    volumes = {}
    fusion = {}
    window[sub] = anat.createWindow('3D')
    #palette = "semitransparent-peak"
    #window[sub].camera(view_quaternion=[0.627211,-0.326506,-0.326506,0.627211])
    window[sub].camera(view_quaternion= [0.182811, -0.250033, -0.182992, 0.932071])
    window[sub].windowConfig(cursor_visibility=0)
    window[sub].camera(zoom=1.3)
    
    
    #for img in ['input','output']:
    SWM_crop     = f'{model_path}/subjects/{sub}'
    vol_1 = aims.read(SWM_crop)
    volumes[sub+'_fibers'] = anat.toAObject(vol_1)
    palette = 'cold_hot'
    build_gradient(pal)
    fusion[sub+'_fibers'] = anat.fusionObjects(objects=[volumes[sub+'_fibers']], method='VolumeRenderingFusionMethod')
    fusion[sub+'_fibers'].setPalette('VR-palette', minVal=0,maxVal=0.5, absoluteMode=True)
    window[sub].addObjects(fusion[sub+'_fibers'])
    out_img = sub.split('.')[0]+'.png'
    window[sub].snapshot(f'{model_path}/snapshots/{out_img}', width=500, height=500)
    
# custom palette
pal = anatomist.Anatomist().createPalette('VR-palette')
pal.header()['palette_gradients'] = '0;0;0.0166976;0;0.278293;1;0.679035;1;1;1#0;0;0.263451;0;0.384045;0.141176;0.549165;0.435294;0.717996;0.647059;0.886827;1;1;1#0;0;0.369202;0;1;0#0;0;0.265306;0;0.597403;0.564706;1;1'


date = '2025-07-22/15-17-10'
model_path = f'/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/{date}'
SUBJECTS = [f for f in os.listdir(f'{model_path}/subjects') if f.endswith('.nii.gz')]
Parallel(n_jobs=1,prefer='threads')(delayed(QualityCheck)(model_path = model_path,region = region, res=res, sub = sub,idx = idx_sub, hemi= 'R',pal=pal) for idx_sub,sub in enumerate(SUBJECTS))