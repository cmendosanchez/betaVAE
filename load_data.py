# -*- coding: utf-8 -*-
# /usr/bin/env python3
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
#
# Initial code:
# https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate/
#                   betaVAE/load_data.py

"""
Tools in order to create pytorch dataloaders
"""
import os
import sys
import re
from random import sample as sample_2args
import pandas as pd
import numpy as np
from preprocess import *
import nibabel as nib
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor as Pool
import time

def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

def create_subset(config):
    """
    Creates dataset HCP_1 from HCP data
    Args:
        config: instance of class Config
    Returns:
        subset: Dataset corresponding to HCP_1
    """

    #We load the list of subjects 
    train_list = pd.read_csv(config.subject_dir)
    print('""" Reading .csv ',config.subject_dir,'"""')
    print(train_list)

    #We remove sub- prefix if exists
    train_list.columns=['subjects']
    train_list['subjects'] = train_list['subjects'].astype('str')
    tmp_sub = train_list['subjects'].tolist()
    if tmp_sub[0][:4]=='sub-':
        tmp_sub = [subject[4:] for subject in tmp_sub]
        train_list['subjects']=tmp_sub

    if config.nsamples!='None':
        train_list = train_list.head(config.nsamples)

    print('""" Train list without the prefix sub- """')
    print(train_list)

    filename, file_extension = os.path.splitext(config.data_dir)
    print('""" Filename and file-extension of subjects data\n',filename,file_extension,'"""')
    
    if file_extension=='.pkl':
        print('Reading pickle file')
        #The pickle file contrain a dataframe with id of the subjects as columns and one row with the numpy arrays
        tmp = pd.read_pickle(config.data_dir)
        #print('final tmp',tmp.shape,tmp.iloc[0][0].shape,type(tmp.iloc[0][0]))

    elif file_extension=='.npy':
        print('""" Loading numpy file """')
        #We load the numpy file and append the crop to a list ( [numpy array] ) 
        tmp = np.load(config.data_dir)
        if config.nsamples!='None':
            tmp = tmp[:config.nsamples]
        print('Shape of numpy file',tmp.shape)
        list_crops = []
        for crop in range(0,tmp.shape[0]):
            list_crops.append([tmp[crop,:,:,:,:]])

        #We create a dictionary containing the subject (key) and their crop (value)
        dict_sub_crop = dict(zip(train_list['subjects'].tolist(), list_crops))
        print('Size of dictionary containing Subject id (key) and Crop (value)', len(dict_sub_crop))

    tmp = pd.DataFrame.from_dict(dict_sub_crop)
    #We are almost there
    tmp = tmp.T
    tmp.index.astype('str')
    ''' Just as a reminder
    a = {'A':[123],'B':[245],'C':[678]}
    tmp = pd.DataFrame.from_dict(a)
    print(tmp,'\n',tmp.T)
    tmp = tmp.T
    print([tmp.index[k] for k in range(len(tmp))])
    Output:
         A    B    C
        0  123  245  678 
            0
        A  123
        B  245
        C  678
        ['A', 'B', 'C']
        ** Process exited - Return Code: 0 **
        Press Enter to exit terminal
    '''
    #Here we get a list with the ID of the subjects
    tmp['subjects'] = [tmp.index[k] for k in range(len(tmp))]
    print('Final input number of subject:',len(tmp['subjects'].tolist()))
    tmp = tmp.merge(tmp['subjects'], left_on = 'subjects', right_on='subjects', how='right')
    filenames = list(tmp['subjects'])
    subset = SkeletonDataset(config=config, dataframe=tmp, filenames=filenames)
    print('------- Succesfully loaded dataset')
    return subset

def split_filename(filename):
    # Use regular expression to match the subject ID pattern
    match = re.search(r'sub-\d+', filename)
    #print(match)
    if match:
        return match.group(0)

def read_nifti_parallel(folder_path,file_name):
    subject_id = split_filename(file_name)  # Assuming the file name is the subject ID
    file_path = os.path.join(folder_path, file_name)
    return subject_id,[nib.load(f'{folder_path}/{file_name}').get_fdata()]

def create_subset_from_folder(config,folder_path, num_subjects=None):
    """
    Creates a dataset subset from files in a folder.
    
    Args:
        folder_path (str): Path to the folder containing subject data.
        num_subjects (int, optional): Number of subjects to include in the subset. Defaults to None (all subjects).
    
    Returns:
        subset: Dataset corresponding to the subset of subjects.
    """
    print("~~~ Creating dataset from folder ~~~")
    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Filter files that are in the expected format (e.g., numpy arrays or pickle files)
    subject_files = [f for f in all_files if f.endswith(('.nii.gz'))][0:num_subjects]
    #print('""" Filtered subject files: ', subject_files, '"""')

    # Create an empty list to store the crops and a list of subject ids
    start_time_ser = time.time()
    '''
    list_crops = []
    subject_ids = []

    # Load data for each subject file
    for i, file_name in enumerate(tqdm(subject_files)):
        subject_id = split_filename(file_name)  # Assuming the file name is the subject ID
        subject_ids.append(subject_id)
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.nii.gz'):
            list_crops.append([nib.load(f'{folder_path}/{file_name}').get_fdata()])
    '''
    print("--- %s seconds ---" % (time.time() - start_time_ser))

    results = Parallel(n_jobs=8)(delayed(read_nifti_parallel)(folder_path, file_name) for file_name in tqdm(subject_files))
    # Create the dataframe with subject IDs and their respective crops
    #dict_sub_crop = dict(zip(subject_ids, list_crops))
    dict_sub_crop = dict(results)
    print(f'Size of dictionary containing Subject ID (key) and Crop (value): {len(dict_sub_crop)}')
    tmp = pd.DataFrame.from_dict(dict_sub_crop)
    #We are almost there
    tmp = tmp.T
    tmp.index.astype('str')
    ''' Just as a reminder
    a = {'A':[123],'B':[245],'C':[678]}
    tmp = pd.DataFrame.from_dict(a)
    print(tmp,'\n',tmp.T)
    tmp = tmp.T
    print([tmp.index[k] for k in range(len(tmp))])
    Output:
         A    B    C
        0  123  245  678 
            0
        A  123
        B  245
        C  678
        ['A', 'B', 'C']
        ** Process exited - Return Code: 0 **
        Press Enter to exit terminal
    '''
    #Here we get a list with the ID of the subjects
    tmp['subjects'] = [tmp.index[k] for k in range(len(tmp))]
    print('Final input number of subject:',len(tmp['subjects'].tolist()))
    tmp = tmp.merge(tmp['subjects'], left_on = 'subjects', right_on='subjects', how='right')
    filenames = list(tmp['subjects'])
    subset = SkeletonDataset(config=config, dataframe=tmp, filenames=filenames)
    print('------- Successfully created dataset subset')
    return subset



def process_subject_file(folder_path,file_name):
    """
    Process a single file: Load the .nii.gz file and return the subject_id and the corresponding data.
    """
    #print('process_subject_file args',folder_path,file_name,len(file_name))
    subject_id = split_filename(file_name)  # Get subject ID from the file name
    subject_data = None
    
    if file_name.endswith('.nii.gz'):
        # Load the NIfTI file and get the data
        subject_data = nib.load(os.path.join(folder_path, file_name)).get_fdata()
    
    return subject_id, subject_data

def parallel_process_files(subject_files, folder_path, num_workers=4):
    """
    Process the subject files in parallel using multiprocessing.
    """
    #print('Parallel processing on files')
    # Use multiprocessing Pool to distribute the workload
    with Pool(max_workers=num_workers) as pool:
        results = pool.map(process_subject_file, [folder_path]*len(subject_files) ,subject_files)
    
    #print('results',list(results)[0:1])
    # Collect results
    subject_ids = []
    list_crops = []
    
    for subject_id, subject_data in list(results):
        subject_ids.append(subject_id)
        if subject_data is not None:
            list_crops.append([subject_data])
    
    return subject_ids, list_crops


def create_subset_from_list(config,subjects):
    """
    Creates a dataset subset from files in a folder.
    
    Args:
        folder_path (str): Path to the folder containing subjects data.
        subject_files (list): List with the name of the files to load from folder_path.
    
    Returns:
        subset: Dataset corresponding to the subset of subjects.
    """
    #print("~~~ Creating dataset from folder ~~~")
    
    #start_time_ser = time.time()
    try:
        list_crops = []
        subject_ids = []
        # Load data for each subject file
        for i, sub in enumerate(subjects):
            sub_id = split_filename(sub)  #We get the subject ID
            file_name = f'{config.path_crops}/{sub}_{config.Region}_{config.Criteria}_{config.minl}_{config.maxl}_{config.referential}.nii.gz'
            if os.path.exists(file_name):
                subject_ids.append(sub_id)
                list_crops.append([nib.load(file_name).get_fdata()])
            else:
                continue
        #print("--- %s seconds ser ---" % (time.time() - start_time_ser))
        
        #start_time_par = time.time()
        #subject_ids, list_crops = parallel_process_files(subject_files, folder_path, num_workers=4)
        #print("--- %s seconds par---" % (time.time() - start_time_par))
        '''
        print('Checking if lists are equal',subject_ids_1==subject_ids)
        all_true = all(np.array_equal(list_crops_1[i], list_crops[i]) for i in range(len(list_crops_1)))
        print("All comparisons are True" if all_true else "Not all comparisons are True")
        '''
        # Create the dataframe with subject IDs and their respective crops
        dict_sub_crop = dict(zip(subject_ids, list_crops))
        tmp = pd.DataFrame.from_dict(dict_sub_crop)
        #We are almost there
        tmp = tmp.T
        tmp.index.astype('str')
        ''' Just as a reminder
        a = {'A':[123],'B':[245],'C':[678]}
        tmp = pd.DataFrame.from_dict(a)
        print(tmp,'\n',tmp.T)
        tmp = tmp.T
        print([tmp.index[k] for k in range(len(tmp))])
        Output:
            A    B    C
            0  123  245  678 
                0
            A  123
            B  245
            C  678
            ['A', 'B', 'C']
            ** Process exited - Return Code: 0 **
            Press Enter to exit terminal
        '''
        #Here we get a list with the ID of the subjects
        tmp['subjects'] = [tmp.index[k] for k in range(len(tmp))]
        tmp = tmp.merge(tmp['subjects'], left_on = 'subjects', right_on='subjects', how='right')
        filenames = list(tmp['subjects'])
        subset = SkeletonDataset(config=config, dataframe=tmp, filenames=filenames)
        print(f'create_subset_from_list ---> Successfully created dataset ~ size subjects_id: {len(subject_ids)} size list_crops {len(list_crops)}')
        return subset
    except:
        print('Error during creation of subset from list')




def create_anomaly_set(config,subjects_ids):
    """
    Creates a dataset subset from files in a folder.
    
    Args:
        folder_path (str): Path to the folder containing subjects data.
        subject_files (list): List with the name of the files to load from folder_path.
    
    Returns:
        subset: Dataset corresponding to the subset of subjects.
    """
    #print("~~~ Creating dataset from folder ~~~")
    
    #start_time_ser = time.time()
    try:
        list_crops = []
        subject_ids = []
        # Load data for each subject file
        
        for i, sub in enumerate(subjects_ids):
            subject_id = split_filename(sub)  #We get the subject ID
            subject_ids.append(subject_id)
            list_crops.append([nib.load(f'{config.path_crops}/{sub}_{config.Region}_{config.Criteria}_{config.minl}_{config.maxl}_{config.referential}.nii.gz').get_fdata()])
        #print("--- %s seconds ser ---" % (time.time() - start_time_ser))
        
        #start_time_par = time.time()
        #subject_ids, list_crops = parallel_process_files(subject_files, folder_path, num_workers=4)
        #print("--- %s seconds par---" % (time.time() - start_time_par))
        '''
        print('Checking if lists are equal',subject_ids_1==subject_ids)
        all_true = all(np.array_equal(list_crops_1[i], list_crops[i]) for i in range(len(list_crops_1)))
        print("All comparisons are True" if all_true else "Not all comparisons are True")
        '''
        # Create the dataframe with subject IDs and their respective crops
        dict_sub_crop = dict(zip(subject_ids, list_crops))
        tmp = pd.DataFrame.from_dict(dict_sub_crop)
        #We are almost there
        tmp = tmp.T
        tmp.index.astype('str')
        ''' Just as a reminder
        a = {'A':[123],'B':[245],'C':[678]}
        tmp = pd.DataFrame.from_dict(a)
        print(tmp,'\n',tmp.T)
        tmp = tmp.T
        print([tmp.index[k] for k in range(len(tmp))])
        Output:
            A    B    C
            0  123  245  678 
                0
            A  123
            B  245
            C  678
            ['A', 'B', 'C']
            ** Process exited - Return Code: 0 **
            Press Enter to exit terminal
        '''
        #Here we get a list with the ID of the subjects
        tmp['subjects'] = [tmp.index[k] for k in range(len(tmp))]
        tmp = tmp.merge(tmp['subjects'], left_on = 'subjects', right_on='subjects', how='right')
        filenames = list(tmp['subjects'])
        subset = SkeletonDataset(config=config, dataframe=tmp, filenames=filenames)
        #print('------- Successfully created dataset subset')
        return subset
    except Exception as e:
        print(f'Error during creation of subset from list: {e}')


def get_subjects_by_removed_number(config, number):
    subjects = []

    # regex pattern
    if config.Anomaly == 'Underconnectivity':
        pattern = re.compile(
            rf"sub-(\d+)_{config.Region}_{config.Criteria}_{config.minl}_{config.maxl}_removed_{number}_{config.referential}_crop\.nii\.gz$"
        )

    elif config.Anomaly == 'Overconnectivity':
        pattern = re.compile(
            rf"sub-(\d+)_{config.Region}_{config.Criteria}_{config.minl}_{config.maxl}_added_{number}_{config.referential}_crop\.nii\.gz$"
        )

    for file in os.listdir(config.path_anom):
        match = pattern.match(file)
        if match:
            subjects.append(match.group(1))  # Extract subject ID

    return subjects

def create_subset_for_anomaly(config,Anomaly,anomaly_ids,nbun):
    """
    Creates a dataset subset from files in a folder.
    
    Args:
        folder_path (str): Path to the folder containing subjects data.
        subject_files (list): List with the name of the files to load from folder_path.
    
    Returns:
        subset: Dataset corresponding to the subset of subjects.
    """
    #print("~~~ Creating dataset from folder ~~~")
    
    #start_time_ser = time.time()
    try:

        #subject_ids_filtered = list(set(get_subjects_by_removed_number(config, nbun)) & set(anomaly_ids))
        #print(f'{len(anomaly_ids)},{subject_ids_filtered}')

        list_crops = []
        subject_ids = []
        # Load data for each subject file
        
        nsubs = 0
        for i, sub in enumerate(anomaly_ids):
            subject_id = split_filename(sub)  #We get the subject ID
            
            if Anomaly == 'Underconnectivity':
                if os.path.exists(f'{config.path_anom}/{Anomaly}/{sub}_{config.Region}_{config.Criteria}_{config.minl}_{config.maxl}_removed_{nbun}_{config.referential}_crop.nii.gz'):
                    subject_ids.append(subject_id)
                    list_crops.append([nib.load(f'{config.path_anom}/{Anomaly}/{sub}_{config.Region}_{config.Criteria}_{config.minl}_{config.maxl}_removed_{nbun}_{config.referential}_crop.nii.gz').get_fdata()])
                    nsubs+=1

            if Anomaly == 'Overconnectivity':
                if os.path.exists(f'{config.path_anom}/{Anomaly}/{sub}_{config.Region}_{config.Criteria}_{config.minl}_{config.maxl}_added_{nbun}_{config.referential}_crop.nii.gz'):
                    subject_ids.append(subject_id)
                    list_crops.append([nib.load(f'{config.path_anom}/{Anomaly}/{sub}_{config.Region}_{config.Criteria}_{config.minl}_{config.maxl}_added_{nbun}_{config.referential}_crop.nii.gz').get_fdata()])
                    nsubs+=1
        #print("--- %s seconds ser ---" % (time.time() - start_time_ser))
        #print('Final subjects id anomaly',len(subject_ids))
        #start_time_par = time.time()
        #subject_ids, list_crops = parallel_process_files(subject_files, folder_path, num_workers=4)
        #print("--- %s seconds par---" % (time.time() - start_time_par))
        '''
        print('Checking if lists are equal',subject_ids_1==subject_ids)
        all_true = all(np.array_equal(list_crops_1[i], list_crops[i]) for i in range(len(list_crops_1)))
        print("All comparisons are True" if all_true else "Not all comparisons are True")
        '''
        # Create the dataframe with subject IDs and their respective crops
        dict_sub_crop = dict(zip(subject_ids, list_crops))
        tmp = pd.DataFrame.from_dict(dict_sub_crop)
        #We are almost there
        tmp = tmp.T
        tmp.index.astype('str')
        ''' Just as a reminder
        a = {'A':[123],'B':[245],'C':[678]}
        tmp = pd.DataFrame.from_dict(a)
        print(tmp,'\n',tmp.T)
        tmp = tmp.T
        print([tmp.index[k] for k in range(len(tmp))])
        Output:
            A    B    C
            0  123  245  678 
                0
            A  123
            B  245
            C  678
            ['A', 'B', 'C']
            ** Process exited - Return Code: 0 **
            Press Enter to exit terminal
        '''
        #Here we get a list with the ID of the subjects
        tmp['subjects'] = [tmp.index[k] for k in range(len(tmp))]
        tmp = tmp.merge(tmp['subjects'], left_on = 'subjects', right_on='subjects', how='right')
        filenames = list(tmp['subjects'])
        subset = SkeletonDataset(config=config, dataframe=tmp, filenames=filenames)
        #print('------- Successfully created dataset subset')
        return subset, nsubs
    except Exception as e:
        print(f'Error during creation of subset from list: {e}')