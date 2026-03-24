from train import * 

def get_AUC_testing(config, vae, device, criterion):
    
    print(f'{bcolors.RED} Launching Normal/Anomaly classification {bcolors.RESET}')
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