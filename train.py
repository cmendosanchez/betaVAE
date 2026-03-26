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
# https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate/

import numpy as np
import pandas as pd
import torch.nn as nn
import optuna
import random
from beta_vae import *
from utils.pytorchtools import EarlyStopping
import time
from load_data import create_subset_from_list, create_subset_for_anomaly
from colors import bcolors
from General_utils import read_one_column_tsv
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import  roc_auc_score
import nibabel as nib
import pickle 
from itertools import chain
from utils.tools import EarlyStopping

def linear_weights(n):
    weights = np.arange(n, 0, -1)   # n, n-1, ..., 1
    return weights / weights.sum()

def get_AUC(config, vae, device, criterion):
    
    print(f'{bcolors.BG_RED} Launching Normal/Anomaly classification {bcolors.RESET}')
    resulting_aucs  = {}
    individual_aucs = {'Underconnectivity_list' : [] , 'Overconnectivity_list' : [] }
 
    for Anomaly in ['Underconnectivity','Overconnectivity']:
        try:
            aucs_list = []
            class_subjects       = read_one_column_tsv(config.Class_val_list)
            mid = int(len(class_subjects) // 2)
            normal_group = class_subjects[:mid]
            anomaly_group = class_subjects[mid:]
            normal_subset = create_subset_from_list(config, normal_group)
            normal_loader = torch.utils.data.DataLoader(normal_subset, batch_size=32, num_workers=4, shuffle=False)
            embeddings_normal = []
            for inputs, path in normal_loader:
                with torch.no_grad():
                    inputs = Variable(inputs).to(device, dtype=torch.float32)
                    output, z, logvar = vae(inputs)
                    embeddings_normal.append(z.cpu().numpy())
                    if config.loss == 'CrossEntropy':
                        target = torch.squeeze(inputs, dim=1).long()
                        partial_recon_loss_anom, partial_kl_val, loss = vae_loss(output, target, z, logvar, criterion, kl_weight=config.kl)
                        output = torch.argmax(output, dim=1)

                    elif config.loss == 'MSE':
                        partial_recon_loss_anom, partial_kl_val, loss = vae_loss(output, inputs, z, logvar, criterion, kl_weight=config.kl) 
            
            embeddings_normal = np.vstack(embeddings_normal)
            y_normal = np.asarray([0]*len(normal_loader.dataset)).reshape(-1)

            anomaly_group = class_subjects[mid:]
            with open(f'{config.path_stats}{Anomaly}_{config.Criteria}.pkl', 'rb') as file:
                results = pickle.load(file)

            data   = [x for x in results if not isinstance(x, tuple)]
            flat = list(chain.from_iterable(data))
            df = pd.DataFrame(flat)
            if df.empty:
                print(f'{bcolors.CYAN}Dataframe is empty!{bcolors.RESET}')
                individual_aucs[Anomaly+'_list'] = np.nan
                resulting_aucs[Anomaly] = np.nan
                continue 

            min_bundles = df['Bundles'].min()
            max_bundles = df['Bundles'].max()
            
            errors_weights = linear_weights(max_bundles)
            print(f'max min bundles: {max_bundles} {min_bundles}')
            auc_weights = linear_weights(max_bundles)
            for nbun in range(1,max_bundles+1):
                embeddings_anomaly = []
                anomaly_subset, nsubjects = create_subset_for_anomaly(config,Anomaly,anomaly_group,nbun)
                print(f'Nbundles {nbun} Nsubjects {nsubjects}')
                anom_loader = torch.utils.data.DataLoader(anomaly_subset,batch_size=32,num_workers=4, shuffle=False)
                for inputs, path in anom_loader:
                    with torch.no_grad():
                        inputs = Variable(inputs).to(device, dtype=torch.float32)
                        output, z, logvar = vae(inputs)
                        embeddings_anomaly.append(z.cpu().numpy())

                embeddings_anomaly = np.vstack(embeddings_anomaly)
                y_anomaly = np.asarray([1]*nsubjects).reshape(-1)

                X = np.vstack((embeddings_normal, embeddings_anomaly))
                y = np.concatenate((y_normal, y_anomaly))

                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                aucs = []
                for i,(train_index, test_index) in enumerate(kf.split(X, y)):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    model_svm = svm.SVC(probability=True, kernel='linear', random_state=42, C=0.01)
                    model_svm.fit(X_train, y_train)
                    y_prob = model_svm.predict_proba(X_test)[:,1]
                    roc_auc = roc_auc_score(y_test, y_prob)
                    aucs.append(roc_auc)
                aucs_list.append(np.mean(aucs))

            for idx,v in enumerate(aucs_list):
                individual_aucs[Anomaly+'_list'].append((idx+1,v,auc_weights[idx],len(normal_loader.dataset),len(anom_loader.dataset)))

            weighted_aucs = np.asarray(aucs_list) * auc_weights
            resulting_aucs[Anomaly] = np.sum(weighted_aucs)

            

        except Exception as e:
            print(e)
            individual_aucs[Anomaly+'_list'] = np.nan
            resulting_aucs[Anomaly] = np.nan
            continue
        
    print(f'{bcolors.RED}Final AUC: {resulting_aucs} {individual_aucs}{bcolors.RESET}')
    return resulting_aucs, individual_aucs



def train_vae_optuna(config, trial,root_dir=None):
    """ Trains beta-VAE for a given hyperparameter configuration
    Args:
        config: instance of class Config
        trainloader: torch loader of training data
        valloader: torch loader of validation data
        root_dir: str, directory where to save model
    Returns:
        vae: trained model
        final_loss_val
    """
    start_time = time.time()
    torch.manual_seed(5)

    #writer = SummaryWriter(log_dir= config.save_dir+'logs/',comment="")
    lr = config.lr
    weight_decay= config.weight_decay
    
    vae = VAE(config.in_shape, config.n, depth=config.depth, loss_selected=config.loss)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    vae.to(device)

    #summary(vae, list(config.in_shape))
    print(f'{bcolors.MAGENTA}train_vae_optuna() \n Config:\n{config}{bcolors.RESET}')
    if config.loss == 'CrossEntropy':
        print('Using Cross Entropy Loss, reduction=sum')
        weights = [1, 2]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
        
    elif config.loss == 'MSE':
        print('Using Mean Square Error Loss, reduction=sum')
        criterion = nn.MSELoss(reduction='sum')

    #optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)
    if config.early_stopping == 1:
        early_stopping = EarlyStopping(patience=config.patience, delta=config.delta, start_epoch=5, verbose=True)

    list_loss_train, list_val_recon_loss, = [], []
    list_aucs = []

    train_subjects      = read_one_column_tsv(config.Train_list)
    n_train             = int(len(train_subjects) * config.sub_perc)
    train_subjects      = train_subjects[:n_train]
    csv_train           = f'{config.save_dir}train_list.csv'
    df_t                = pd.DataFrame(train_subjects, columns=['Subject'])
    df_t.to_csv(csv_train, index=False)
    print(f"Data written to {csv_train}")
    print(f'Nsubjects Train: {len(train_subjects)}')
    set_train = create_subset_from_list(config,train_subjects)
    start_loading = time.time()
    trainloader = torch.utils.data.DataLoader(set_train,batch_size=config.batch_size,num_workers=6, shuffle=True)
    print(f"{bcolors.MAGENTA}-- -Created trainloader in  {time.time() - start_loading} seconds ---{bcolors.RESET}")

    validation_subjects = read_one_column_tsv(config.Rcon_val_list)
    n_val = int(len(validation_subjects) * config.sub_perc)
    validation_subjects = validation_subjects[:n_val]
    set_val   = create_subset_from_list(config,validation_subjects)
    valloader = torch.utils.data.DataLoader(set_val,batch_size=32,num_workers=4, shuffle=False)
    csv_val = f'{config.save_dir}validation_list.csv'
    df_v = pd.DataFrame(validation_subjects, columns=['Subject'])
    df_v.to_csv(csv_val, index=False)
    print(f"Data written to {csv_val}")
    print(f'Validation:{len(validation_subjects)}')
    
    vae.train()
    for epoch in range(1,config.nb_epoch+1):
        start_time_epoch = time.time()
        print(f'{bcolors.RED}{bcolors.UNDERLINE}~~ Starting epoch {epoch}{bcolors.RESET}')
        #Defined epoch losses
        train_recon_loss   = 0.0
        train_kl_loss      = 0.0
        train_running_loss = 0.0

        for inputs, path in trainloader: #Training
            optimizer.zero_grad()
            inputs = Variable(inputs).to(device, dtype=torch.float32)
            output, z, logvar = vae(inputs)

            if config.loss == 'CrossEntropy':
                target = torch.squeeze(inputs, dim=1).long()
                partial_recon_loss, partial_kl_loss, partial_loss = vae_loss(output, target, z, logvar, criterion, kl_weight=config.kl) 
                output = torch.argmax(output, dim=1) 
            
            elif config.loss== 'MSE':
                partial_recon_loss, partial_kl_loss, partial_loss = vae_loss(output, inputs, z, logvar, criterion, kl_weight=config.kl) 

            partial_loss.backward()
            optimizer.step()
            #Update errors
            train_recon_loss    += partial_recon_loss
            train_kl_loss       += partial_kl_loss
            train_running_loss  += partial_loss.item()

        print(f'--- %s seconds epoch --- {time.time() - start_time_epoch}')

        train_recon_loss      /=  len(train_subjects)
        train_kl_loss         /=  len(train_subjects)
        train_running_loss    /=  len(train_subjects)

        print(f"{bcolors.GREEN}[{epoch}] Train Recon loss: {train_recon_loss}  {bcolors.RESET}")
        print(f"{bcolors.GREEN}[{epoch}] Train KL loss: {train_kl_loss}        {bcolors.RESET}")
        print(f"{bcolors.GREEN}[{epoch}] Train loss: {train_running_loss}      {bcolors.RESET}")

        #Save epoch loss
        list_loss_train.append(train_running_loss)

        # Validation losses
        val_recon_loss   = 0.0
        val_kl_loss      = 0.0
        val_running_loss = 0.0

        vae.eval() #Eval mode
        for inputs, path in valloader:
            with torch.no_grad():
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, z, logvar = vae(inputs)
                #print('tensor shape',inputs.shape,output.shape)
                if config.loss == 'CrossEntropy':
                    target = torch.squeeze(inputs, dim=1).long()
                    partial_recon_loss_val, partial_kl_loss_val, partial_loss_val = vae_loss(output, target, z, logvar, criterion, kl_weight=config.kl)
                    output = torch.argmax(output, dim=1)

                elif config.loss == 'MSE':
                    partial_recon_loss_val, partial_kl_loss_val, partial_loss_val = vae_loss(output, inputs, z, logvar, criterion, kl_weight=config.kl)

                #Update losses for each sample
                val_recon_loss    += partial_recon_loss_val.cpu().numpy()
                val_kl_loss       += partial_kl_loss_val
                val_running_loss  += partial_loss_val.item()

        #Average
        val_recon_loss     /=  len(validation_subjects)
        val_kl_loss        /=  len(validation_subjects)
        val_running_loss   /=  len(validation_subjects)
        
        if not np.isfinite(val_recon_loss):
            print(f"NaN/Inf encountered at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

        trial.report(val_recon_loss, epoch)
        list_val_recon_loss.append(val_recon_loss)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if config.early_stopping == 1:
            early_stopping.check_early_stop(val_recon_loss, epoch,  vae, optimizer)

        if early_stopping.stop_training:
            affine = np.eye(4)
            nifti_input  = nib.Nifti1Image(np.array(np.squeeze(inputs[0]).cpu().detach().numpy()), affine)
            nifti_output = nib.Nifti1Image(np.array(np.squeeze(output[0]).cpu().detach().numpy()), affine)
            nib.save(nifti_input  , f'{config.save_dir}input.nii.gz')
            nib.save(nifti_output , f'{config.save_dir}output.nii.gz')

            resulting_auc, individual_auc = get_AUC(config, vae, device, criterion)

            for key,val in resulting_auc.items():
                trial.set_user_attr(key, val)

            for key,val in individual_auc.items():
                trial.set_user_attr(key, val)
            break
        
        if epoch == config.nb_epoch:
            affine = np.eye(4)
            nifti_input  = nib.Nifti1Image(np.array(np.squeeze(inputs[0]).cpu().detach().numpy()), affine)
            nifti_output = nib.Nifti1Image(np.array(np.squeeze(output[0]).cpu().detach().numpy()), affine)
            nib.save(nifti_input  , f'{config.save_dir}input.nii.gz')
            nib.save(nifti_output , f'{config.save_dir}output.nii.gz')
            resulting_auc, individual_auc = get_AUC(config, vae, device,criterion)
            for key,val in resulting_auc.items():
                    trial.set_user_attr(key, val)

            for key,val in individual_auc.items():
                trial.set_user_attr(key, val)
            break

        # prints on the terminal
        print(f"{bcolors.YELLOW}[{epoch}] Val Recon loss: {val_recon_loss}  {bcolors.RESET}")
        print(f"{bcolors.YELLOW}[{epoch}] Val KL loss: {val_kl_loss}        {bcolors.RESET}")
        print(f"{bcolors.YELLOW}[{epoch}] Val loss: {val_running_loss}      {bcolors.RESET}")

        torch.cuda.empty_cache()      
        vae.train()

    print(f"{bcolors.BG_GREEN}Finished Optuna Trial in  --- {time.time() - start_time} seconds ---{bcolors.RESET}") 
    return min(list_val_recon_loss)


def train_vae_model(config, root_dir=None):
    """ Trains beta-VAE for a given hyperparameter configuration
    Args:
        config: instance of class Config
        trainloader: torch loader of training data
        valloader: torch loader of validation data
        root_dir: str, directory where to save model
    Returns:
        vae: trained model
        final_loss_val
    """
    start_time = time.time()
    torch.manual_seed(5)

    lr = config.lr
    weight_decay= config.weight_decay
    
    vae = VAE(config.in_shape, config.n, depth=config.depth, loss_selected=config.loss)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    vae.to(device)

    #summary(vae, list(config.in_shape))
    print(f'{bcolors.MAGENTA}train_vae_model() \n Config:\n{config}{bcolors.RESET}')
    if config.loss == 'CrossEntropy':
        print('Using Cross Entropy Loss, reduction=sum')
        weights = [1, 2]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
        
    elif config.loss == 'MSE':
        print('Using Mean Square Error Loss, reduction=sum')
        criterion = nn.MSELoss(reduction='sum')

    #optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=config.patience, delta=config.delta, verbose=True, save_best=True,path=config.path_model, start_epoch=5)

    list_loss_train, list_kl_loss_train, list_recon_loss_train = [], [], []
    list_loss_val, list_kl_loss_val, list_recon_loss_val = [], [], []

    train_subjects      = read_one_column_tsv(config.Train_list)
    n_train             = int(len(train_subjects) * 1)
    train_subjects      = train_subjects[:n_train]
    csv_train           = f'{config.save_dir}train_list.csv'
    df_t                = pd.DataFrame(train_subjects, columns=['Subject'])
    df_t.to_csv(csv_train, index=False)
    print(f"Data written to {csv_train}")
    print(f'Nsubjects Train: {len(train_subjects)}')
    set_train = create_subset_from_list(config,train_subjects)
    start_loading = time.time()
    trainloader = torch.utils.data.DataLoader(set_train,batch_size=config.batch_size,num_workers=12, shuffle=True)
    print(f"{bcolors.MAGENTA}-- -Created trainloader in  {time.time() - start_loading} seconds ---{bcolors.RESET}")

    validation_subjects = read_one_column_tsv(config.Rcon_val_list)
    n_val = int(len(validation_subjects) * 1)
    validation_subjects = validation_subjects[:n_val]
    set_val   = create_subset_from_list(config,validation_subjects)
    valloader = torch.utils.data.DataLoader(set_val,batch_size=32,num_workers=6, shuffle=False)
    csv_val = f'{config.save_dir}validation_list.csv'
    df_v = pd.DataFrame(validation_subjects, columns=['Subject'])
    df_v.to_csv(csv_val, index=False)
    print(f"Data written to {csv_val}")
    print(f'Validation:{len(validation_subjects)}')
    
    vae.train()
    for epoch in range(1,config.nb_epoch+1):
        start_time_epoch = time.time()
        print(f'{bcolors.RED}{bcolors.UNDERLINE}~~ Starting epoch {epoch}{bcolors.RESET}')
        #Defined epoch losses
        train_recon_loss   = 0.0
        train_kl_loss      = 0.0
        train_running_loss = 0.0

        for inputs, path in trainloader: #Training
            optimizer.zero_grad()
            inputs = Variable(inputs).to(device, dtype=torch.float32)
            output, z, logvar = vae(inputs)

            if config.loss == 'CrossEntropy':
                target = torch.squeeze(inputs, dim=1).long()
                partial_recon_loss, partial_kl_loss, partial_loss = vae_loss(output, target, z, logvar, criterion, kl_weight=config.kl) 
                output = torch.argmax(output, dim=1) 
            
            elif config.loss== 'MSE':
                partial_recon_loss, partial_kl_loss, partial_loss = vae_loss(output, inputs, z, logvar, criterion, kl_weight=config.kl) 

            partial_loss.backward()
            optimizer.step()
            #Update errors
            train_recon_loss    += partial_recon_loss.cpu().detach().numpy()
            train_kl_loss       += partial_kl_loss.cpu().detach().numpy()
            train_running_loss  += partial_loss.cpu().detach().numpy()

        print(f'--- %s seconds epoch --- {time.time() - start_time_epoch}')

        train_recon_loss      /=  len(train_subjects)
        train_kl_loss         /=  len(train_subjects)
        train_running_loss    /=  len(train_subjects)

        print(f"{bcolors.GREEN}[{epoch}] Train Recon loss: {train_recon_loss}  {bcolors.RESET}")
        print(f"{bcolors.GREEN}[{epoch}] Train KL loss: {train_kl_loss}        {bcolors.RESET}")
        print(f"{bcolors.GREEN}[{epoch}] Train loss: {train_running_loss}      {bcolors.RESET}")

        if not np.isfinite(train_recon_loss):
            print(f"NaN/Inf encountered at epoch {epoch}")
            return

        #Save epoch train loss
        list_loss_train.append(train_running_loss)
        list_kl_loss_train.append(train_kl_loss)
        list_recon_loss_train.append(train_recon_loss)

        # Validation losses
        val_recon_loss   = 0.0
        val_kl_loss      = 0.0
        val_running_loss = 0.0

        vae.eval() #Eval mode
        for inputs, path in valloader:
            with torch.no_grad():
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, z, logvar = vae(inputs)
                #print('tensor shape',inputs.shape,output.shape)
                if config.loss == 'CrossEntropy':
                    target = torch.squeeze(inputs, dim=1).long()
                    partial_recon_loss_val, partial_kl_loss_val, partial_loss_val = vae_loss(output, target, z, logvar, criterion, kl_weight=config.kl)
                    output = torch.argmax(output, dim=1)

                elif config.loss == 'MSE':
                    partial_recon_loss_val, partial_kl_loss_val, partial_loss_val = vae_loss(output, inputs, z, logvar, criterion, kl_weight=config.kl)

                #Update losses for each sample
                val_recon_loss    += partial_recon_loss_val.cpu().detach().numpy()
                val_kl_loss       += partial_kl_loss_val.cpu().detach().numpy()
                val_running_loss  += partial_loss_val.cpu().detach().numpy()

        #Average
        val_recon_loss     /=  len(validation_subjects)
        val_kl_loss        /=  len(validation_subjects)
        val_running_loss   /=  len(validation_subjects)
        
        # prints on the terminal
        print(f"{bcolors.YELLOW}[{epoch}] Val Recon loss: {val_recon_loss}  {bcolors.RESET}")
        print(f"{bcolors.YELLOW}[{epoch}] Val KL loss: {val_kl_loss}        {bcolors.RESET}")
        print(f"{bcolors.YELLOW}[{epoch}] Val loss: {val_running_loss}      {bcolors.RESET}")

        #Save epoch val loss
        list_recon_loss_val.append(val_recon_loss)
        list_kl_loss_val.append(val_kl_loss)
        list_loss_val.append(val_running_loss)

        early_stopping.check_early_stop(val_recon_loss, epoch, vae, optimizer)

        if early_stopping.stop_training:
            affine = np.eye(4)
            nifti_input  = nib.Nifti1Image(np.array(np.squeeze(inputs[0]).cpu().detach().numpy()), affine)
            nifti_output = nib.Nifti1Image(np.array(np.squeeze(output[0]).cpu().detach().numpy()), affine)
            nib.save(nifti_input  , f'{config.save_dir}input.nii.gz')
            nib.save(nifti_output , f'{config.save_dir}output.nii.gz')
            data_dict = {'LossTrain': list_loss_train,'klTrain':list_kl_loss_train,'ReconTrain':list_recon_loss_train,
            'LossVal':list_recon_loss_val,'klVal':list_kl_loss_val,'ReconVal':list_loss_val}
            for key,val in data_dict.items():
                np.save(f'{config.save_dir}{key}.npy',np.asarray(val))
            break

        if epoch == config.nb_epoch:
            affine = np.eye(4)
            nifti_input  = nib.Nifti1Image(np.array(np.squeeze(inputs[0]).cpu().detach().numpy()), affine)
            nifti_output = nib.Nifti1Image(np.array(np.squeeze(output[0]).cpu().detach().numpy()), affine)
            nib.save(nifti_input  , f'{config.save_dir}input.nii.gz')
            nib.save(nifti_output , f'{config.save_dir}output.nii.gz')
            torch.save({
                "epoch": epoch,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, config.path_model)


            data_dict = {'LossTrain': list_loss_train,'klTrain':list_kl_loss_train,'ReconTrain':list_recon_loss_train,
            'LossVal':list_recon_loss_val,'klVal':list_kl_loss_val,'ReconVal':list_loss_val}
            for key,val in data_dict.items():
                np.save(f'{config.save_dir}{key}.npy',np.asarray(val))
            break

        

        torch.cuda.empty_cache()      
        vae.train()

    print(f"{bcolors.BG_GREEN}Finished Train in  --- {time.time() - start_time} seconds ---{bcolors.RESET}") 
    return 