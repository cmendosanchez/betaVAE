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
        print('Shape of numpy file',tmp.shape)
        list_crops = []
        for crop in range(0,tmp.shape[0]):
            list_crops.append([tmp[crop,:,:,:,:]])

        #We create a dictionary containing the subject (key) and their crop (value)
        dict_sub_crop = dict(zip(train_list['subjects'].tolist(), list_crops))
        print('Size of dictionary containing Subject id (key) and Crop (value)', len(dict_sub_crop))

        #If we want to train with the whole dataset
        if config.remove_subjects == False:
            tmp = pd.DataFrame.from_dict(dict_sub_crop)
            #print('Print some pickle info',type(tmp),list(tmp.index),list(tmp.columns.values),tmp.info(),type(tmp.iloc[0]['1000021']),tmp.iloc[0]['1000021'].shape)

        #If we want to remove some subjects
        else:
            #We load the list of subjects to be removed from train
            subjects_to_exclude = pd.read_csv(config.subjects_to_remove)
            print('""" Subjects to exclude"""\n',subjects_to_exclude)
            subjects_to_exclude.columns = ['subjects']

            #We remove the sub- part from the string (if exists)
            subjects_to_exclude['subjects'] = subjects_to_exclude['subjects'].astype('str')
            tmp_excluded = subjects_to_exclude['subjects'].tolist()
            print('Before removing sub- prefix\n',tmp_excluded[0][:4])
            if tmp_excluded[0][:4]=='sub-':
                tmp_excluded = [subject[4:] for subject in tmp_excluded]
                subjects_to_exclude['subjects']=tmp_excluded
            print('After removing sub- prefix\n',subjects_to_exclude)

            #We get a list of the subjects to be removed
            subs_to_remove = subjects_to_exclude['subjects'].tolist()
            print(' ° dict_sub_crop size before removing subjects', len(dict_sub_crop))
            print(' ° train_list size before removing subjects',len(train_list))

            print('""" Removing subjects """')
            #We removed the subjects from the dictionary and from the whole list of subjects
            for sub in subs_to_remove:
                del dict_sub_crop[sub]
            train_list = filter_rows_by_values(train_list , "subjects", subs_to_remove)

            print('°° dict_sub_crop size after removing subjects', len(dict_sub_crop))
            print('°° train_list size before after removing subjects',len(train_list))
            #Save everything to csv
            train_dataset = pd.DataFrame.from_dict({'ID':train_list['subjects'].tolist()})
            train_dataset.to_csv(config.save_dir+'/Subjects_list.csv')

            #Finally, we create a dataframe from the dictionary
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



