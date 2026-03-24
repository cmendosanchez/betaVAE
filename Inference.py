import os
import argparse
from types import SimpleNamespace
from beta_vae import * 
import pandas as pd
import yaml
from General_utils import read_one_column_tsv
from load_data import create_subset_from_list, create_subset_for_anomaly
import time
from colors import bcolors
import random
from sklearn.model_selection import StratifiedKFold
import pickle
from itertools import chain

def create_parser():
    parser = argparse.ArgumentParser(
        description="Run inference using a trained VAE model"
    )

    parser.add_argument(
        "-m", "--model_dir",
        type=str,
        required=True,
        help="Path to folder where the VAE model is stored"
    )

    parser.add_argument(
        "-r", "--region",
        type=str,
        required=True,
        help="Region"
    )

    parser.add_argument(
        "-c", "--criteria",
        type=str,
        required=True,
        help="Segmentation criteria (e.g., Comm, DWM, SWM)"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data (crops)"
    )

    parser.add_argument(
         "--database",
        type=str,
        required=True,
        help="Path to data (crops)"
    )


    parser.add_argument(
        "-o", "--outdir",
        type=str,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "-s", "--subjects",
        type=str,
        required=True,
        help=".tsv file listing the subjects to process"
    )

    return parser


def run(model_dir, region, criteria, outdir, subjects, data, database):
    print("=== Running VAE Inference ===")
    print(f"Model dir: {model_dir}")
    print(f"Region: {region}")
    print(f"Criteria: {criteria}")
    print(f"Output dir: {outdir}")

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)


    torch.manual_seed(0)
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)

    # ---- Example: locate model checkpoint ----
    # (adapt depending on your naming)
    checkpoint_path = os.path.join(model_dir, "model.pt")
    config_path     = os.path.join(model_dir, "config.yaml")

    # ---- Sanity check ---- 
    if not os.path.exists(checkpoint_path):
        print(f"{bcolors.RED}Model not found: {checkpoint_path}{bcolors.RESET}")
        return
    else:
        print(f"{bcolors.CYAN}Found model: {checkpoint_path}{bcolors.RESET}")

    if not os.path.exists(config_path):
        print(f"{bcolors.RED}Config not found: {config_path}{bcolors.RESET}")
        return
    else:
        print(f"{bcolors.CYAN}Config found: {config_path}{bcolors.RESET}")

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = SimpleNamespace(**config)
    print(f"{bcolors.BLUE}---- Running Inference ----{bcolors.RESET}")

    # ---- loading model ----
    model = VAE(config.in_shape, config.n, depth=config.depth, loss_selected= config.loss)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model = model.to(device)
    model.eval()

    if config.loss == 'CrossEntropy':
        weights = [1, 2]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
        print('"""Using Cross Entropy Loss, reduction=sum')

    elif config.loss == 'MSE':
        criterion = nn.MSELoss( reduction='sum')
        print('"""Using MSE Loss, reduction=sum')

    config.path_crops = data

    subjects_list = read_one_column_tsv(subjects)[0:100]
    subjects_set  = create_subset_from_list(config,subjects_list)
    start_loading = time.time()
    subjects_loader = torch.utils.data.DataLoader(subjects_set, batch_size=1,num_workers=6, shuffle=False)
    print(f"{bcolors.MAGENTA}--- Created subjects loader in  {time.time() - start_loading} seconds ---{bcolors.RESET}")

    embeddings_list = []
    recon_error_list = []
    subs_ids = []

    start = time.time()
    with torch.no_grad():

        for inputs, path in subjects_loader: #We iterate the subjects dataset one by one
            #print(path)
            inputs = Variable(inputs).to(device, dtype=torch.float32)
            target = torch.squeeze(inputs, dim=1).long()
            z, logvar = model.encode(inputs) # z = mean because no random sampling
            outputs = model.decode(z)
           

            if config.loss == 'CrossEntropy':
                recon_loss_val, kl_val, loss_val = vae_loss(outputs, target, z, logvar, criterion,
                                kl_weight=config.kl) 
                outputs = torch.argmax(outputs, dim=1) 

            elif config.loss == 'MSE':
                recon_loss_val, kl_val, loss_val = vae_loss(outputs, inputs, z, logvar, criterion,
                                kl_weight=config.kl) 
            embeddings_list.append(z.cpu().detach().numpy())
            recon_error_list.append(recon_loss_val.cpu().detach().numpy())
            subs_ids.append(path[0])

    print(f"{bcolors.MAGENTA}--- Embeddings and Recon Error computed in {time.time() - start} seconds ---{bcolors.RESET}")
    embeddings = np.asarray(embeddings_list)
    recon_error = np.asarray(recon_error_list)
    n = embeddings.shape[1]
    # ---- Embeddings dataframe ----
    columns = ["subject"] + [f"dim_{i}" for i in range(n)]
    df_embeddings = pd.DataFrame(
        data=[[subj] + emb.tolist() for subj, emb in zip(subs_ids, embeddings)],
        columns=columns)
    df_embeddings.to_csv(f"{outdir}/embeddings.csv", index=False)
    # ---- Reconstruction error dataframe ----
    # Case 1: scalar error per subject
    df_recon = pd.DataFrame({
        "subject": subs_ids,
        "recon_error": recon_error})
    df_recon.to_csv(f"{outdir}/recon_error.csv", index=False)

    if True:
        print(f"{bcolors.YELLOW}--- Test in fake anomaly dataset ---{bcolors.RESET}")
        config.path_stats = f'/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/Stats_Anomaly/{database}'
        config.path_anom  = f'/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/FakeAnomaly_crops/{database}'
        print(config)

        if database=='UKB':
            test_anom_subjects = read_one_column_tsv('/lustre/fsn1/projects/rech/miu/ugf68us/PhD_2026/Crops_6Regions/DataSplit_DL/test_anom.tsv')
        elif database =='HCP':
            test_anom_subjects = read_one_column_tsv(subjects)


        for Anomaly in ['Underconnectivity','Overconnectivity']:
            for i in range(1,11):
                random.shuffle(test_anom_subjects)
                mid = int(len(test_anom_subjects) // 2)
                normal_group  = test_anom_subjects[:mid]
                anomaly_group = test_anom_subjects[mid:]
                normal_subset = create_subset_from_list(config, normal_group)
                normal_loader = torch.utils.data.DataLoader(normal_subset, batch_size=1, num_workers=4, shuffle=False)
                embeddings_normal = []
                reconerror_normal = []
                for inputs, path in normal_loader:
                    with torch.no_grad():
                        inputs = Variable(inputs).to(device, dtype=torch.float32)
                        target = torch.squeeze(inputs, dim=1).long()
                        z, logvar = model.encode(inputs) # z = mean because no random sampling
                        outputs = model.decode(z)

                        if config.loss == 'CrossEntropy':
                            target = torch.squeeze(inputs, dim=1).long()
                            partial_recon_loss_norm, partial_kl_val, loss = vae_loss(outputs, target, z, logvar, criterion, kl_weight=config.kl)
                            output = torch.argmax(output, dim=1)

                        elif config.loss == 'MSE':
                            partial_recon_loss_norm, partial_kl_val, loss = vae_loss(outputs, inputs, z, logvar, criterion, kl_weight=config.kl) 

                        embeddings_normal.append(z.cpu().detach().numpy())
                        reconerror_normal.append(partial_recon_loss_norm.cpu().detach().numpy())

                #embeddings_normal = np.vstack(embeddings_normal)
                embeddings_normal = np.asarray(embeddings_normal)
                y_normal = np.asarray([0]*len(normal_loader.dataset)).reshape(-1)

                with open(f'{config.path_stats}/{database}_{region}_{Anomaly}_{criteria}.pkl', 'rb') as file:
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

                #auc_weights = linear_weights(max_bundles)
                for nbun in range(1,max_bundles+1):
                    embeddings_anomaly = []
                    reconerror_anomaly = []
                    anomaly_subset, nsubjects = create_subset_for_anomaly(config,Anomaly,anomaly_group,nbun)
                    print(f'Nbundles {nbun} Nsubjects {nsubjects}')
                    anom_loader = torch.utils.data.DataLoader(anomaly_subset,batch_size=1,num_workers=4, shuffle=False)

                    for inputs, path in anom_loader:
                        with torch.no_grad():
                            inputs = Variable(inputs).to(device, dtype=torch.float32)
                            target = torch.squeeze(inputs, dim=1).long()
                            z, logvar = model.encode(inputs) # z = mean because no random sampling
                            outputs = model.decode(z)

                            if config.loss == 'CrossEntropy':
                                target = torch.squeeze(inputs, dim=1).long()
                                partial_recon_loss_anom, partial_kl_val, loss = vae_loss(outputs, target, z, logvar, criterion, kl_weight=config.kl)
                                output = torch.argmax(output, dim=1)

                            elif config.loss == 'MSE':
                                partial_recon_loss_anom, partial_kl_val, loss = vae_loss(outputs, inputs, z, logvar, criterion, kl_weight=config.kl) 

                            embeddings_anomaly.append(z.cpu().detach().numpy())
                            reconerror_anomaly.append(partial_recon_loss_anom.cpu().detach().numpy())

                    #embeddings_anomaly = np.vstack(embeddings_anomaly)
                    embeddings_anomaly = np.asarray(embeddings_anomaly)
                    y_anomaly = np.asarray([1]*len(anom_loader.dataset)).reshape(-1)

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

                #weighted_aucs = np.asarray(aucs_list) * auc_weights
                resulting_aucs[Anomaly] = aucs
                print('AUCS',resulting_aucs)
                df = pd.DataFrame(X, columns=[f"dim_{i}" for i in range(X.shape[1])])
                df["label"] = y  
                df.to_csv(f"{outdir}/{Anomaly}_shuffle_{i}.csv", sep="\t", index=False) 

def main():
    parser = create_parser()
    args = parser.parse_args()

    run(
        model_dir=args.model_dir,
        region=args.region,
        criteria=args.criteria,
        outdir=args.outdir,
        subjects=args.subjects,
        data=args.data,
        database=args.database
    )


if __name__ == "__main__":
    main()