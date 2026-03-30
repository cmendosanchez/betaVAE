import numpy as np
import os 
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, GridSearchCV
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.cross_decomposition import CCA



""" Region = 'S.C.-sylv.'
Validation2  = pd.read_csv(f'/home_local/cm283129/PhD_UKB/R_S.C.-sylv._numpy/Test_1.csv')
Validation2_interrupted = Validation2[Validation2["Interrupted"] == 1]["Subject"].values.tolist()
Validation2_continuous  = Validation2[Validation2["Interrupted"] == 0]["Subject"].values.tolist()
ndims = [32,256]
for idx,embeddings_csv in enumerate(['/neurospin/dico/cmendoza/Inference/UKB_S.C.-sylv._right_SWM/embeddings_allsubs.csv']):
    df = pd.read_csv(embeddings_csv)
    df = df.rename(columns={'subject': 'ID'})
    df = df.sort_values(by='ID')
    print(df.dtypes)
    df['dim_1'] = df['dim_1'].apply(ast.literal_eval)
    dims = pd.DataFrame(df['dim_1'].tolist(), index=df.index)
    # Rename columns (optional)
    dims.columns = [f'dim{i+1}' for i in range(dims.shape[1])]
    # Concatenate with original dataframe (dropping old column if you want)
    embeddings = pd.concat([df.drop(columns=['dim_1']), dims], axis=1)


    Embeddings_interrupted = embeddings[embeddings['ID'].isin(Validation2_interrupted)]
    Embeddings_continuous = embeddings[embeddings['ID'].isin(Validation2_continuous)]
    print(Embeddings_interrupted,Embeddings_continuous)

    labels_0 = [0]*len(Embeddings_continuous )
    Embeddings_continuous['labels'] = labels_0 

    labels_1 = [1]*len(Embeddings_interrupted )
    Embeddings_interrupted['labels'] = labels_1

    Full_dataset = pd.concat([Embeddings_interrupted,Embeddings_continuous], axis=0)
    print('Interrupted and Continuous Dataset (Full dataset)',Full_dataset)
    #ROC Curve
    X = Full_dataset.drop(columns=['ID','labels']).to_numpy()
    y = Full_dataset['labels'].to_numpy()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    best_auc = 0
    best_model = None
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure()
    print(f'~~~ Stratified Cross fold Validation ~~~')
    fig2, axes2 = plt.subplots(1,1, figsize=(8, 8),sharey=True,sharex=True) 
    for i,(train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model_svm = svm.SVC(probability=True, kernel='linear', random_state=42,C=0.01)
        model_svm.fit(X_train, y_train)
        y_prob = model_svm.predict_proba(X_test)[:,1]
        fpr, tpr, threshold = roc_curve(y_test, y_prob)
        #print(fpr,tpr,threshold)
        roc_auc = roc_auc_score(y_test, y_prob)
        aucs.append(roc_auc)
        print(f'Fold:{i+1}, ROC AUC:{roc_auc}')
        # Interpolación para tener todos los TPR en el mismo eje
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        axes2.plot(fpr, tpr, alpha=0.5,lw=2)
    #Mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(np.asarray(mean_fpr), np.asarray(mean_tpr))
    print(mean_auc)

    axes2.plot(mean_fpr, mean_tpr, color='black',label=f'Mean ROC\n(AUC = {mean_auc:.2f})',lw=3)
    # Sombra ± 1 desviación estándar
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    axes2.fill_between(mean_fpr, tpr_lower, tpr_upper, color='black', alpha=0.2, label='± 1 std. dev.')
    # Línea aleatoria
    axes2.plot([0, 1], [0, 1], 'r--', lw=1,label='Random')
    axes2.legend(loc='lower right',fontsize=8)
    axes2.grid(True)
    plt.title(f'Classification in Validation 2,  Dimensions: {ndims[idx]}')
    plt.show()
    #fig2.savefig(f'/neurospin/dico/cmendoza/Runs/14_PhD_UKB/Output/Experiment_1/R_S.C.-sylv._Train_2_sift2/Champollion_{ndims[idx]}.png',dpi=300) """


#df_champ = pd.read_csv('/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/SC-sylv_right/name06-17-02_84/hcp_random_embeddings/full_embeddings.csv')
#print(df_champ)
#parameters={'l1_ratio': np.linspace(0,1,11), 'alpha': [10**k for k in range(-3,4)], 'max_iter': [10000]}
parameters = {
    'l1_ratio': np.linspace(0, 1, 21),  # finer (0.05 steps)
    'alpha': np.logspace(-4, 2, 20),    # much finer log scale
    'max_iter': [10000]
}

n_jobs=24

dataset_localization = '/neurospin/dico/data/deep_folding/current/datasets/' # Jean Zay : '/lustre/fswork/projects/rech/tgu/umy22uu/Runs/70_self-supervised_two-regions/Input/'
hemisphere = 'right'  # 'left' or 'right'
labels_dir = f'{dataset_localization}/hcp/hcp_isomap_labels_SC-sylv_{hemisphere}.csv'
label_list = [f'Isomap_central_{hemisphere}_dim{k}' for k in range(1,7)]
splits_basedir = f'{dataset_localization}/hcp/Isomap/splits/train_val_split_'
test_subs_dir = f'{dataset_localization}/hcp/Isomap/splits/test_split.csv'
subject_name = 'Subject'



#models = [f'/neurospin/dico/cmendoza/Inference/HCP_S.C.-sylv._{hemisphere}_SWM/embeddings_allsubs.csv']
models = [f'/neurospin/dico/cmendoza/Inference/HCP_S.C.-sylv._{hemisphere}_SWM/embeddings_allsubs.csv']
res = {}
""" for embds_dir in models:


    df = pd.read_csv(embds_dir)
    df = df.rename(columns={'subject': 'ID'})
    df = df.sort_values(by='ID')
    print(df.dtypes)
    df['dim_1'] = df['dim_1'].apply(ast.literal_eval)
    value = df.loc[df['ID'] == 585862, 'dim_1'].values[0][0]
    print(value)
    dims = pd.DataFrame(df['dim_1'].tolist(), index=df.index)
    # Rename columns (optional)
    dims.columns = [f'dim{i+1}' for i in range(dims.shape[1])]
    # Concatenate with original dataframe (dropping old column if you want)
    df_expanded = pd.concat([df.drop(columns=['dim_1']), dims], axis=1)
    print(df_expanded,df_expanded.dtypes)


    # store score for each regression
    cross_val_r2_list = []
    test_r2_list = []

    # load embeddings
    embds = df_expanded
    #embds = pd.read_csv('/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/SC-sylv_right/name06-17-02_84/hcp_random_embeddings/full_embeddings.csv')
    embds.columns = ['ID'] + [f'dim{i}' for i in range(embds.shape[1]-1)]

    # remove duplicates
    embds = embds.drop_duplicates(subset=['ID'])
    # load labels
    labels = pd.read_csv(labels_dir)
    # restrict embds to Subjects with labels
    embds = embds[embds['ID'].isin(labels[subject_name])].reset_index(drop=True)
    # same for labels
    labels = labels[labels[subject_name].isin(embds['ID'])].reset_index(drop=True)
    print('labels:',labels)


    


    # align labels and embds on 'ID'
    labels = labels.merge(embds[['ID']], left_on=subject_name, right_on='ID', how='right')
    # order all by ID
    embds = embds.sort_values(by='ID').reset_index(drop=True)
    labels = labels.sort_values(by='ID').reset_index(drop=True)
    subjects = embds['ID']
    print('Embeddings after merging',embds)

    



    # define X, Y and subjects
    X = embds.drop(columns=['ID'])
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # get the custom splits
    subs_embeddings = pd.DataFrame({'ID': subjects, 'X': list(X)})
    root_dir = '/'.join(splits_basedir.split('/')[:-1])
    basedir = splits_basedir.split('/')[-1]
    splits_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.startswith(basedir) and '.csv' in f]
    splits_subs = [pd.read_csv(file, header=None) for file in splits_dirs]
    folds = np.concatenate([[i] * len(K) for i, K in enumerate(splits_subs)])
    splits_subs = pd.concat(splits_subs)
    splits_subs.columns=['ID']
    splits_subs['labels'] = folds
    df = subs_embeddings.merge(splits_subs, on='ID')
    groups, X_train_val = df['labels'], np.vstack(df['X'].values)

    # get test
    test_subjects = pd.read_csv(test_subs_dir, header=None)
    test_subjects.columns = ['ID']
    subs_embeddings_test = subs_embeddings.merge(test_subjects, on='ID')
    X_test = np.vstack(subs_embeddings_test['X'].values)

    for label in label_list:

        # merge label with embeddings
        # first, train val
        df_label = labels[['ID', label]].rename(columns={label: 'Y'})
        df_y = df.merge(df_label, on='ID')
        Y_train_val = df_y['Y']
        # then, test
        subs_embeddings_test_y = subs_embeddings_test.merge(df_label, on='ID')
        Y_test = subs_embeddings_test_y['Y']

        # instantiate cross-validation
        logo = LeaveOneGroupOut()
        cv = [*(logo.split(X_train_val, Y_train_val, groups=groups))]
        # define model
        model = ElasticNet()
        clf = GridSearchCV(model, parameters, cv=cv, scoring='r2', refit=True, n_jobs=n_jobs,verbose=0)

        # fit cross-validation
        clf.fit(X_train_val,Y_train_val)
        print(f'best params : {clf.best_params_}')
        print(f'best score : {clf.best_score_}')
        md = clf.best_estimator_
        # compute r2 on cross-validation
        cross_val_r2 = cross_val_score(md, X_train_val, Y_train_val, cv=cv, scoring='r2')
        print(f'Cross-val R2: {cross_val_r2.mean():.3f}')

        # compute r2 on test
        test_r2 = md.score(X_test, Y_test)
        print(f'Test R2: {test_r2:.3f}')

        cross_val_r2_list.append(cross_val_r2)
        test_r2_list.append(test_r2)

    mean_cross_val_r2 = np.mean(cross_val_r2_list)
    mean_test_r2 = np.mean(test_r2_list)
    print(f'Mean cross-val R2 across labels: {mean_cross_val_r2:.3f}')
    print(f'Mean test R2 across labels: {mean_test_r2:.3f}')
    res[embds_dir]=mean_cross_val_r2

print(res) """

hemisphere='right'
champo_embds = pd.read_csv(f'/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation_latent_256/SC-sylv_{hemisphere}/name09-42-54_107/ukb40_random_embeddings/full_embeddings.csv')
fibers_embds = pd.read_csv(f'/neurospin/dico/cmendoza/Inference/UKB_S.C.-sylv._{hemisphere}_SWM/embeddings_allsubs.csv')
fibers_embds = fibers_embds.rename(columns={'subject': 'ID'})
fibers_embds['dim_1'] = fibers_embds['dim_1'].apply(ast.literal_eval)
dims = pd.DataFrame(fibers_embds['dim_1'].tolist(), index=fibers_embds.index)
# Rename columns (optional)
dims.columns = [f'dim{i+1}' for i in range(dims.shape[1])]
# Concatenate with original dataframe (dropping old column if you want)
df_expanded = pd.concat([fibers_embds.drop(columns=['dim_1']), dims], axis=1)

print(champo_embds,df_expanded)

# Set ID as index
df1 = champo_embds.set_index('ID')
df2 = df_expanded.set_index('ID')

print(df1,df2)
# Find common IDs
common_ids = df1.index.intersection(df2.index)

# Align rows ONLY (preserve original columns)
df1 = df1.loc[common_ids]
df2 = df2.loc[common_ids]


X1 = df1.values
X2 = df2.values
print(X1,X1.shape,X2,X2.shape)


scaler1 = StandardScaler()
scaler2 = StandardScaler()

X1 = scaler1.fit_transform(X1)
X2 = scaler2.fit_transform(X2)

cca = CCA(n_components=10)
E1_c, E2_c = cca.fit_transform(X1, X2)

corrs = [np.corrcoef(E1_c[:,i], E2_c[:,i])[0,1] for i in range(10)]
print(np.mean(corrs))