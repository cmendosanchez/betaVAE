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
import torchvision
#from torchsummary import summary
#from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import optuna
import random
from beta_vae import *
from utils.pytorchtools import EarlyStopping
from postprocess import plot_loss
import time
from load_data import create_subset_from_list
import subprocess
import csv

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

        grid = torchvision.utils.make_grid(images)
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

def train_vae_optuna(config, dataset, trial,root_dir=None):
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
    print('~~~ Dataset:',dataset)
    start_time = time.time()
    torch.manual_seed(5)
    #writer = SummaryWriter(log_dir= config.save_dir+'logs/',comment="")
    lr = config.lr
    vae = VAE(config.in_shape, config.n, depth=config.depth, loss_selected=config.loss)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    vae.to(device)
    #summary(vae, list(config.in_shape))
    print(config)
    if config.loss == 'CrossEntropy':
        print('Using Cross Entropy Loss, reduction=sum')
        weights = [1, 2]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
        
    elif config.loss == 'MSE':
        print('Using Mean Square Error Loss, reduction=sum')
        criterion = nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    #early_stopping = EarlyStopping(patience=10,delta=0.1, verbose=True, root_dir=root_dir,save_model=False)

    list_loss_train, list_loss_val = [], []

    #Attemp of lazy loading
    print('Lazy loading')
    subject_files = [f for f in sorted(os.listdir(dataset)) if f.endswith(('.nii.gz'))][0:config.nsamples]
    # Split into 80% and 20%
    split_index = int(0.8 * len(subject_files))
    train_subjects = subject_files[:split_index]  # 80% data
    validation_subjects = subject_files[split_index:]   # 20% data

    # Open the file for writing
    csv_train = f'{config.save_dir}train_list.csv'
    csv_val = f'{config.save_dir}validation_list.csv' 
    # Convert the list to a DataFrame (single column)
    df_t = pd.DataFrame(train_subjects, columns=['Subject'])
    df_v = pd.DataFrame(validation_subjects, columns=['Subject'])
    df_t.to_csv(csv_train, index=False)
    df_v.to_csv(csv_val, index=False)
    print(f"Data written to {csv_train}")
    print(f"Data written to {csv_val}")
    print(f'Nsubjects Train: {len(train_subjects)} Validation:{len(validation_subjects)}')

    #set_train = create_subset_from_list(config,dataset,train_subjects)
    set_val = create_subset_from_list(config,dataset,validation_subjects)
    start_loading = time.time()
    #trainloader = torch.utils.data.DataLoader(set_train,batch_size=config.batch_size,num_workers=4, shuffle=True)
    valloader = torch.utils.data.DataLoader(set_val,batch_size=1,num_workers=4, shuffle=False)
    print("-- -Create val data loader  %s seconds ---" % (time.time() - start_loading))

    for epoch in range(config.nb_epoch):
        print(f'~~ Starting epoch {epoch}')
        start_time_epoch = time.time()
        #Defined epoch losses
        running_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        epoch_steps = 0
        #Shuffle and batch. Load 4096 crops in memory
        shuffled_batches = shuffle_and_batch(train_subjects, 2048)
        for idx_batch,batch in enumerate(shuffled_batches):
            #print('Create subset from list')
            #start_loading = time.time()
            set_ = create_subset_from_list(config,dataset,batch)
            #print(f'epoch {epoch} set {idx_batch} ready')
            #print("--- %s seconds ---" % (time.time() - start_loading))
            trainloader = torch.utils.data.DataLoader(set_,batch_size=config.batch_size,num_workers=4, shuffle=False)
            for inputs, path in trainloader: #Training
                optimizer.zero_grad()
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, z, logvar = vae(inputs)
                #print('tensor shape',inputs.shape,output.shape)
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

                #Update errors
                running_loss += loss.item()
                recon_loss += partial_recon_loss
                kl_loss += partial_kl
                epoch_steps += 1

                del inputs, output

            del set_, trainloader
        
        print(f'--- %s seconds epoch --- {time.time() - start_time_epoch}')

        running_loss = running_loss / epoch_steps
        recon_loss = recon_loss / epoch_steps
        kl_loss = kl_loss / epoch_steps

        print("[%d] KL loss: %.2e" % (epoch + 1, kl_loss))
        print("[%d] recon loss: %.2e" % (epoch + 1, recon_loss))
        print("[%d] loss: %.2e" % (epoch + 1,running_loss))

        #Save epoch loss
        list_loss_train.append(running_loss)

        # Validation losses
        val_loss = 0.0
        recon_loss_val = 0.0
        kl_val = 0.0
        val_steps = 0

        vae.eval() #Eval mode

        for inputs, path in valloader:
            with torch.no_grad():
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, z, logvar = vae(inputs)
                #print('tensor shape',inputs.shape,output.shape)
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
                #Update losses for each sample
                val_loss += loss.cpu().numpy()
                recon_loss_val += partial_recon_loss_val.cpu().numpy()
                kl_val += partial_kl_val
                val_steps += 1
                #del inputs,output
        #Average
        valid_loss = val_loss / val_steps
        recon_loss_val = recon_loss_val / val_steps
        kl_val = kl_val / val_steps
        

        # If loss is NaN, prune the trial
        if np.isnan(recon_loss_val):
            print(f"NaN encountered at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

        #Report to Optuna
        trial.report(recon_loss_val, epoch)
        # If Optuna determines that the trial should be pruned, raise an exception to stop training early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()  # This will stop the trial early if it is underperforming
        

        if epoch == config.nb_epoch-1:
            #print('numpy shape',np.array(np.squeeze(inputs[0]).cpu().detach().numpy()).shape,np.array(np.squeeze(output[0]).cpu().detach().numpy()).shape)
            np.save(f'{config.save_dir}input.npy', np.array(np.squeeze(inputs[0]).cpu().detach().numpy()))
            np.save(f'{config.save_dir}output.npy',np.array(np.squeeze(output[0]).cpu().detach().numpy()))


        # prints on the terminal
        print("[%d] KL validation loss: %.2e" % (epoch + 1, kl_val))
        print("[%d] recon validation loss: %.2e" % (epoch + 1, recon_loss_val))
        print("[%d] validation loss: %.2e" % (epoch + 1, valid_loss))

        list_loss_val.append(recon_loss_val)
        print("")
        torch.cuda.empty_cache()
    
    np.save(f'{config.save_dir}train_loss.npy', np.asarray(list_loss_train))
    np.save(f'{config.save_dir}val_loss.npy', np.asarray(list_loss_val))

    final_loss_val = list_loss_val[-1:]
    print(f"Finished train Ndimensions {config.n} Beta {config.kl} Total Subjects {config.nsamples} --- %s seconds --- {time.time() - start_time}")
    return final_loss_val[0]

