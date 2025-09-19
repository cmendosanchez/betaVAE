import os
import shutil
from subprocess import call
import yaml
import pandas as pd
models_folder = '/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'
models = os.listdir(models_folder)

'''
#Executed once to rename the folders
for model in models:

    model_config = models_folder + model + '/config.yaml'

    with open(model_config, "r") as file:
        data = yaml.safe_load(file)

    try:
        shutil.move(models_folder + model,models_folder + data['dataset_name'] +'_'+str(data['n']))
    except:
        continue

'''
'''
for HCP_region in ['SC-sylv_right_HCP','SC-sylv_left_HCP']:
    for ndim in ['32','75','256']:

        if not os.path.exists(models_folder     + HCP_region    +   '_'     +   ndim):
            os.mkdir(         models_folder  +   HCP_region  +   '_'        +   ndim)

            if HCP_region == 'SC-sylv_right_HCP':
                shutil.copy(models_folder   +   'SC-sylv_right_UKB_'    +   ndim    +   '/checkpoint.pt', models_folder + HCP_region + '_'  +   ndim+'/checkpoint.pt')

            elif HCP_region == 'SC-sylv_left_HCP':
                shutil.copy(models_folder   +   'SC-sylv_left_UKB_'     +    ndim   +   '/checkpoint.pt', models_folder + HCP_region + '_'  +   ndim+'/checkpoint.pt')

'''
for region in ['ACCpatterns_LARGE_CINGULATE_RIGHT','FIP_right_HCP','SOr_left_HCP']:
    for ndim in ['75','256']:

        if not os.path.exists(models_folder     + region    +   '_'     +   ndim):
            os.mkdir(         models_folder  +   region  +   '_'        +   ndim)

            if region == 'ACCpatterns_LARGE_CINGULATE_LEFT':
                shutil.copy(models_folder   +   'LARGE_CINGULATE_left_UKB_'    +   ndim    +   '/checkpoint.pt', models_folder + region + '_'  +   ndim+'/checkpoint.pt')

            elif region == 'ACCpatterns_LARGE_CINGULATE_RIGHT':
                shutil.copy(models_folder   +   'LARGE_CINGULATE_right_UKB_'    +   ndim    +   '/checkpoint.pt', models_folder + region + '_'  +   ndim+'/checkpoint.pt')

            elif region == 'FIP_right_HCP':
                shutil.copy(models_folder   +   'FIP_right_UKB_'     +    ndim   +   '/checkpoint.pt', models_folder + region + '_'  +   ndim+'/checkpoint.pt')

            elif region == 'SOr_left_HCP':
                shutil.copy(models_folder   +   'SOr_left_UKB_'     +    ndim   +   '/checkpoint.pt', models_folder + region + '_'  +   ndim+'/checkpoint.pt')


#for model in ['LARGE_CINGULATE_right_UKB','FIP_right_UKB','SC-sylv_right_UKB','SC-sylv_left_UKB','SC-sylv_right_HCP','SC-sylv_left_HCP','SOr_left_UKB']:
#for model in ['SC-sylv_right_UKB',]:
for model in ['ACCpatterns_LARGE_CINGULATE_RIGHT','FIP_right_HCP','SOr_left_HCP']:
    for ndim in ['75','256']:
        
        print(model, '-------------     Model: ',model, 'Latent space dimension: ', ndim)
        #print('python3 generate_embeddings.py dataset=cristobal/' + data['dataset_name'] + '.yaml +test_model_dir=/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'  + data['dataset_name'] +'_'+str(data['n']) + ' +dataset_localization=neurospin')
        #call(['python3 generate_embeddings.py dataset=cristobal/' + data['dataset_name'] + '.yaml +test_model_dir=/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'  + data['dataset_name'] +'_'+str(data['n']) + ' +dataset_localization=neurospin'],shell=True)
        #print('python3 generate_embeddings.py dataset=cristobal/' + model + '.yaml +test_model_dir=/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'  + model + '_'  + ndim + ' +dataset_localization=neurospin ++n='+ndim)
        #call(['python3 generate_embeddings.py dataset=cristobal/' + model + '.yaml +test_model_dir=/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'  + model + '_'  + ndim + ' +dataset_localization=neurospin ++n='+ndim],shell=True)
        #interrupted = pd.read_csv(models_folder+model+'_'+ndim+'/Interrupted_CS_subjects.csv').ID.values
        #print(interrupted)
        #val = pd.read_csv(models_folder+model+'_'+ndim+'/val_label.csv').iloc[:, 0].tolist()
        #print(val)
        #print(len(list(set(interrupted).intersection(val))))
        call(['python3 generate_embeddings.py dataset=cristobal/' + model + '.yaml +test_model_dir=/neurospin/dico/cmendoza/Runs/Jeanzay/01_betavae_sulci_crops/2025-06-12/'  + model + '_'  + ndim + ' +dataset_localization=neurospin ++n='+ndim],shell=True)