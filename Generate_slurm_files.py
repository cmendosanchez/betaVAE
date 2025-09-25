import os

Hemi   = f'R'
Region = f'S.C.-sylv.'
mode   = f'full_brain'
config_names =   []
for conn in ['','_sift2']:
    for mode in ['full_brain']:
        l_ranges = [('0','250')]
        for l_range in l_ranges:
            config_names.append(f'{mode}_{Hemi}_{Region}_Track_{l_range[0]}_{l_range[1]}{conn}_icbm09c')

latent_dimensions = ['64','128','256','512','1024','2048']
betas = ['1','2','4','8','16','32']
train_index = ['1','2']
output = f'/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE/configs/slurm_files/{Hemi}_{Region}'

if not os.path.exists(output):
    os.makedirs(output)

for config_name in config_names:

    for latent_dim in latent_dimensions:

        for beta in betas:

            for train_idx in train_index:

                job_name  = f'{config_name}'
                job_name += f'_ndim_{latent_dim}'
                job_name += f'_beta_{beta}'
                job_name += f'_TrainIdx_{train_idx}'

                script = f"""#!/bin/bash\n
                #SBATCH --job-name={job_name}          # nom du job\n
                # Il est possible d'utiliser une autre partition que celle par défaut\n
                # en activant l'une des 5 directives suivantes :\n
                #SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go\n
                ##SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go\n
                ##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)\n
                ##SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)\n
                ##SBATCH -C h100                     # decommenter pour la partition gpu_p6 (GPU H100 80 Go)\n
                # Ici, reservation de 10 CPU (pour 1 tache) et d'un GPU sur un seul noeud :\n
                #SBATCH --nodes=1                    # on demande un noeud\n
                #SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)\n
                #SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)\n
                # Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant\n
                # qu'ici on ne reserve qu'un seul GPU (soit 1/4 ou 1/8 des GPU du noeud suivant la partition),\n
                # l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour la seule tache:\n
                #SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU V100)\n
                ##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 des CPU du noeud 8-GPU V100)\n
                ##SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 des CPU du noeud 8-GPU A100)\n
                ##SBATCH --cpus-per-task=24           # nombre de CPU par tache pour gpu_p6 (1/4 des CPU du noeud 4-GPU H100)\n
                # /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm\n
                #SBATCH --hint=nomultithread         # hyperthreading desactive\n
                #SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)\n
                #SBATCH --output=$WORK/PhD_UKB/Program/betaVAE/configs/logs/{job_name}%j.out      # nom du fichier de sortie\n
                #SBATCH --error=$WORK/PhD_UKB/Program/betaVAE/configs/logs/{job_name}%j.out       # nom du fichier d'erreur (ici commun avec la sortie)\n
                #SBATCH -A tgu@v100                  # choice of the partition\n

                # Nettoyage des modules charges en interactif et herites par defaut\n
                module purge\n
                
                # Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"\n
                # pour avoir acces aux modules compatibles avec cette partition\n
                #module load arch/a100
                # Decommenter la commande module suivante si vous utilisez la partition "gpu_p6"\n
                # pour avoir acces aux modules compatibles avec cette partition\n
                #module load arch/h100\n
                dataset=$WORK/PhD_UKB/Program/betaVAE/PhD_UKB/{config_name}\n

                # Chargement des modules\n
                # module load ...\n
                module purge\n 
                module load pytorch-gpu/py3/2.4.0\n

                # Echo des commandes lancees\n
                set -x\n
                
                # Pour les partitions "gpu_p5" et "gpu_p6", le code doit etre compile avec les modules compatibles\n
                # avec la partition choisie\n
                # Execution du code\n
                cd $WORK\n
                cd PhD_UKB/Program/betaVAE\n
                python3 main.py n={latent_dim} kl={beta} +dataset_folder=/lustre/fsn1/projects/rech/tgu/ugf68us/{Hemi}_{Region}_numpy +save_dir=/lustre/fswork/projects/rech/tgu/ugf68us/PhD_UKB/betaVAE_Output +dataset=$dataset +MSE_loss=True +preproc=LogMinMax +split=CustomSplit +train_list=/lustre/fsn1/projects/rech/tgu/ugf68us/{Hemi}_{Region}_numpy/Train_{train_idx}.csv +validation_list=/lustre/fsn1/projects/rech/tgu/ugf68us/{Hemi}_{Region}_numpy/Validation_{train_idx}.csv\n"""
                with open(f"{output}/{job_name}.slurm", "w") as f:
                    f.write(script)
