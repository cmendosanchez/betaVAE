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
#from torchsummary import summary
#from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import optuna
import random
from beta_vae import *
from utils.pytorchtools import EarlyStopping
from postprocess import plot_loss
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

def train_vae(config, trainloader, valloader, root_dir=None):
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
    torch.manual_seed(5)
    #writer = SummaryWriter(log_dir= config.save_dir+'logs/',comment="")
    

    lr = config.lr
    vae = VAE(config.in_shape, config.n, depth=config.depth, loss_selected=config.loss)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    vae.to(device)
    #summary(vae, list(config.in_shape))
    #print(config)
    if config.loss == 'CrossEntropy':
        print('Using Cross Entropy Loss, reduction=sum')
        weights = [1, 2]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
        
    elif config.loss == 'MSE':
        print('Using Mean Square Error Loss, reduction=sum')
        criterion = nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    nb_epoch = config.nb_epoch
    early_stopping = EarlyStopping(patience=10, verbose=True, root_dir=root_dir)

    list_loss_train, list_loss_val = [], []

    # arrays enabling to see model reconstructions
    id_arr, phase_arr, input_arr, output_arr = [], [], [], []

    for epoch in range(config.nb_epoch):
        running_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        epoch_steps = 0
        for inputs, path in trainloader:
            optimizer.zero_grad()
            inputs = Variable(inputs).to(device, dtype=torch.float32)
            output, z, logvar = vae(inputs)

            if config.loss == 'CrossEntropy':
                target = torch.squeeze(inputs, dim=1).long()
                partial_recon_loss, partial_kl, loss = vae_loss(output, target, z,
                                        logvar, criterion,
                                        kl_weight=config.kl) 
                output = torch.argmax(output, dim=1) 
            
            elif config.loss== 'MSE':
                partial_recon_loss, partial_kl, loss = vae_loss(output, inputs, z,
                                        logvar, criterion,
                                        kl_weight=config.kl) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            recon_loss += partial_recon_loss
            kl_loss += partial_kl
            epoch_steps += 1

            #Edit
            #del inputs,output

        running_loss = running_loss / epoch_steps
        recon_loss = recon_loss / epoch_steps
        kl_loss = kl_loss / epoch_steps

        if config.loss == 'CrossEntropy':
            images = [inputs[0][0][10][:][:], output[0][10][:][:]]
        elif config.loss == 'MSE':
            images = [inputs[0][0][10][:][:], output[0][0][10][:][:]] #For CrossEntroy -> images = [inputs[0][0][10][:][:], output[0][10][:][:]]

        #grid = torchvision.utils.make_grid(images)
        '''
        writer.add_image('inputs', images[0].unsqueeze(0), epoch)
        writer.add_image('output', images[1].unsqueeze(0), epoch)
        writer.add_scalar('Loss/train', running_loss, epoch)
        writer.add_scalar('KL Loss/train', kl_loss, epoch)
        writer.add_scalar('recon Loss/train', recon_loss, epoch)
        writer.close()
        '''

        print("[%d] KL loss: %.2e" % (epoch + 1, kl_loss))
        print("[%d] recon loss: %.2e" % (epoch + 1, recon_loss))
        #print(kl_loss * config.kl + recon_loss)
        print("[%d] loss: %.2e" % (epoch + 1,
                                        running_loss))
        list_loss_train.append(running_loss)
        running_loss = 0.0

        """ Saving of reconstructions for visualization in Anatomist software """
        if epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('train')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

        # Validation loss
        val_loss = 0.0
        recon_loss_val = 0.0
        kl_val = 0.0
        val_steps = 0
        total = 0
        vae.eval()
        for inputs, path in valloader:
            with torch.no_grad():
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, z, logvar = vae(inputs)
                if config.loss == 'CrossEntropy':
                    target = torch.squeeze(inputs, dim=1).long()
                    partial_recon_loss_val, partial_kl_val, loss = vae_loss(output, target,  
                                            z, logvar, criterion,
                                            kl_weight=config.kl)
                    output = torch.argmax(output, dim=1)

                elif config.loss == 'MSE':
                    partial_recon_loss_val, partial_kl_val, loss = vae_loss(output, inputs,  
                                            z, logvar, criterion,
                                            kl_weight=config.kl)

                val_loss += loss.cpu().numpy()
                recon_loss_val += partial_recon_loss_val
                kl_val += partial_kl_val
                val_steps += 1
                
                #del inputs, output

        valid_loss = val_loss / val_steps
        recon_loss_val = recon_loss_val / val_steps
        kl_val = kl_val / val_steps

        if config.loss == 'CrossEntropy':
            images = [inputs[0][0][10][:][:],\
                    output[0][10][:][:]]
            
        elif config.loss == 'MSE':
            images = [inputs[0][0][10][:][:],\
                    output[0][0][10][:][:]]  
        '''
        writer.add_scalar('Loss/val', valid_loss, epoch)
        writer.add_scalar('KL Loss/val', kl_val, epoch)
        writer.add_scalar('recon Loss/val', recon_loss_val, epoch)
        writer.add_image('inputs VAL', images[0].unsqueeze(0), epoch)
        writer.add_image('output VAL', images[1].unsqueeze(0), epoch)
        writer.close()
        '''
        # prints on the terminal
        print("[%d] KL validation loss: %.2e" % (epoch + 1, kl_val))
        print("[%d] recon validation loss: %.2e" % (epoch + 1, recon_loss_val))
        #print(kl_val * config.kl + recon_loss_val)
        print("[%d] validation loss: %.2e" % (epoch + 1, valid_loss))

        list_loss_val.append(valid_loss)
        early_stopping(valid_loss, vae)
 
        """ Saving of reconstructions for visualization in Anatomist software """
        if early_stopping.early_stop or epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('val')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())
            break
    for key, array in {'input': input_arr, 'output' : output_arr,
                           'phase': phase_arr, 'id': id_arr}.items():
        np.save(config.save_dir+key, np.array([array]))

    plot_loss(list_loss_train[1:], config.save_dir+'tot_train_',label='Train')
    plot_loss(list_loss_val[1:], config.save_dir+'tot_val_',label='Validation')
    final_loss_val = list_loss_val[-1:]
    
    """Saving of trained model"""
    torch.save((vae.state_dict(), optimizer.state_dict()),
                config.save_dir + 'vae.pt')

    print("Finished Training")
    return vae, final_loss_val

def shuffle_and_batch(data, batch_size=32):
    random.shuffle(data)  # Shuffle the data
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

def linear_weights(n):
    weights = np.arange(n, 0, -1)   # n, n-1, ..., 1
    return weights / weights.sum()

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
    early_stopping = EarlyStopping(patience=config.patience, delta=config.delta, verbose=True)

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

    if config.Anomaly == None:
        validation_subjects = read_one_column_tsv(config.Rcon_val_list)
        print(f'Validation subs: {len(validation_subjects)}')
        n_val = int(len(validation_subjects) * config.sub_perc)
        validation_subjects = validation_subjects[:n_val]
        set_val   = create_subset_from_list(config,validation_subjects)
        valloader = torch.utils.data.DataLoader(set_val,batch_size=32,num_workers=4, shuffle=False)
        csv_val = f'{config.save_dir}validation_list.csv'
        df_v = pd.DataFrame(validation_subjects, columns=['Subject'])
        df_v.to_csv(csv_val, index=False)
        print(f"Data written to {csv_val}")
        print(f'Validation:{len(validation_subjects)}')
    

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

        if config.Anomaly == None:
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
            
            # If loss is NaN, prune the trial
            if not np.isfinite(val_recon_loss):
                print(f"NaN/Inf encountered at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

            trial.report(val_recon_loss, epoch)
            list_val_recon_loss.append(val_recon_loss)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            early_stopping.check_early_stop(val_loss, epoch)

            if early_stopping.stop_training:
                affine = np.eye(4)
                nifti_input  = nib.Nifti1Image(np.array(np.squeeze(inputs[0]).cpu().detach().numpy()), affine)
                nifti_output = nib.Nifti1Image(np.array(np.squeeze(output[0]).cpu().detach().numpy()), affine)
                nib.save(nifti_input  , f'{config.save_dir}input.nii.gz')
                nib.save(nifti_output , f'{config.save_dir}output.nii.gz')
                break


            if epoch == config.nb_epoch:
                affine = np.eye(4)
                nifti_input  = nib.Nifti1Image(np.array(np.squeeze(inputs[0]).cpu().detach().numpy()), affine)
                nifti_output = nib.Nifti1Image(np.array(np.squeeze(output[0]).cpu().detach().numpy()), affine)
                nib.save(nifti_input  , f'{config.save_dir}input.nii.gz')
                nib.save(nifti_output , f'{config.save_dir}output.nii.gz')

            # prints on the terminal
            print(f"{bcolors.YELLOW}[{epoch}] Val Recon loss: {val_recon_loss}  {bcolors.RESET}")
            print(f"{bcolors.YELLOW}[{epoch}] Val KL loss: {val_kl_loss}        {bcolors.RESET}")
            print(f"{bcolors.YELLOW}[{epoch}] Val loss: {val_running_loss}      {bcolors.RESET}")

    
        elif config.Anomaly != None:
            print(f'{bcolors.BG_RED}Launching Normal/Anomaly classification{bcolors.RESET}')
            # Shuffle in-place
            class_subjects       = read_one_column_tsv(config.Class_val_list)
            #random.shuffle(class_subjects)
            # Split
            mid = int(len(class_subjects) // 2)
            normal_group = class_subjects[:mid]
            normal_subset = create_subset_from_list(config, normal_group)
            normal_loader = torch.utils.data.DataLoader(normal_subset, batch_size=32, num_workers=4, shuffle=False)
            embeddings_normal = []
            for inputs, path in normal_loader:
                with torch.no_grad():
                    inputs = Variable(inputs).to(device, dtype=torch.float32)
                    output, z, logvar = vae(inputs)
                    embeddings_normal.append(z.cpu().numpy())
                    """ if config.loss == 'CrossEntropy':
                        target = torch.squeeze(inputs, dim=1).long()
                        partial_recon_loss_anom, partial_kl_val, loss = vae_loss(output, target, z, logvar, criterion, kl_weight=config.kl)
                        output = torch.argmax(output, dim=1)

                    elif config.loss == 'MSE':
                        partial_recon_loss_anom, partial_kl_val, loss = vae_loss(output, inputs, z, logvar, criterion, kl_weight=config.kl) """
                        
            embeddings_normal = np.vstack(embeddings_normal)
            y_normal = np.asarray([0]*len(normal_group)).reshape(-1)
            #print('Normal group shape', embeddings_normal.shape, y_normal.shape)

            anomaly_group = class_subjects[mid:]
            aucs_list = []
            ###
            with open(config.path_stats, 'rb') as file:
                results = pickle.load(file)

            data   = [x for x in results if not isinstance(x, tuple)]
            flat = list(chain.from_iterable(data))
            df = pd.DataFrame(flat)
            if df.empty:
                continue

            min_bundles = df['Bundles'].min()
            max_bundles = df['Bundles'].max()
            auc_weights = linear_weights(max_bundles)
            print(f'max min bundles: {max_bundles} {min_bundles}')

            for nbun in range(1,max_bundles+1):
                embeddings_anomaly = []
                #print(f'Nbundles {nbun}')
                anomaly_subset = create_subset_for_anomaly(config,anomaly_group,nbun)
                anomloader = torch.utils.data.DataLoader(anomaly_subset,batch_size=32,num_workers=4, shuffle=False)
                for inputs, path in anomloader:
                    with torch.no_grad():
                        inputs = Variable(inputs).to(device, dtype=torch.float32)
                        output, z, logvar = vae(inputs)
                        embeddings_anomaly.append(z.cpu().numpy())
                        #print(z.cpu().numpy().shape)
                        """ embeddings_anomaly.append(z.cpu().numpy().reshape(-1))
                        if config.loss == 'CrossEntropy':
                            target = torch.squeeze(inputs, dim=1).long()
                            partial_recon_loss_anom, partial_kl_val, loss = vae_loss(output, target,  
                                                    z, logvar, criterion,
                                                    kl_weight=config.kl)
                            output = torch.argmax(output, dim=1)

                        elif config.loss == 'MSE':
                            partial_recon_loss_anom, partial_kl_val, loss = vae_loss(output, inputs,  
                                                    z, logvar, criterion,
                                                    kl_weight=config.kl)
                    anom_loss+= partial_recon_loss_anom.cpu().numpy() """
                
                #print(f'Recon error anom partial {anom_loss/len(anomaly)}')
                
                embeddings_anomaly = np.vstack(embeddings_anomaly)
                y_anomaly = np.asarray([1]*len(anomaly_group)).reshape(-1)
                #print(embeddings_normal,embeddings_anomaly.shape,embeddings_normal.shape)

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
                #print(aucs_list)   

            weighted_aucs = np.asarray(aucs_list) * auc_weights
            epoch_auc = np.sum(weighted_aucs)
            # If loss is NaN, prune the trial
            print(f"{bcolors.MAGENTA}[{epoch}] {aucs_list} {auc_weights} AUC: {epoch_auc} {bcolors.RESET}")
            
            if not np.isfinite(epoch_auc):
                print(f"NaN/Inf encountered at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

            trial.report(epoch_auc, epoch)
            list_aucs.append(epoch_auc)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            early_stopping.check_early_stop(val_loss, epoch)

            if early_stopping.stop_training:
                affine = np.eye(4)
                nifti_input  = nib.Nifti1Image(np.array(np.squeeze(inputs[0]).cpu().detach().numpy()), affine)
                nifti_output = nib.Nifti1Image(np.array(np.squeeze(output[0]).cpu().detach().numpy()), affine)
                nib.save(nifti_input  , f'{config.save_dir}input.nii.gz')
                nib.save(nifti_output , f'{config.save_dir}output.nii.gz')
                break
            
        torch.cuda.empty_cache()      

    #np.save(f'{config.save_dir}train_loss.npy', np.asarray(list_loss_train))
    #np.save(f'{config.save_dir}val_loss.npy', np.asarray(list_val_recon_loss))
    #final_loss_val = list_val_recon_loss[-1]
    #final_auc      = list_aucs[-1]
    print(f"{bcolors.BG_GREEN}Finished Optuna Trial in  --- {time.time() - start_time} seconds ---{bcolors.RESET}") 
    if config.Anomaly == 'Overconnectivity' or config.Anomaly == 'Underconnectivity':
        return max(list_aucs)
    else:
        return min(list_val_recon_loss)

