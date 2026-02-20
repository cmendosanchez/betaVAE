import os
import textwrap
import argparse

#Example python3 Generate_kraken_files_Optuna.py --s /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output --d PhD_UKB/two_ends_R_S.C.-sylv._Track_0_40_sift2_icbm09c 
# --p /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/crops/R.S.C.-sylv/two_ends_R_S.C.-sylv._Track_0_40_sift2_icbm09c --f /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/OptunaResults/toto --w 3 --t 2 
# --o configs/kraken_files --n two_ends_R_S.C.-sylv._Track_0_40_sift2_icbm09c

#vnode           state host    mem     ncpus   nmics   ngpus  comment
#node-177        free node-177 376gb      48       0       4 node_group: KRAKEN
#node-180        free node-180 251gb     128       0       2 node_group: KRAKEN
#node-181        free node-181 251gb     128       0       2 node_group: KRAKEN
#node-182        free node-182 251gb     128       0       2 node_group: KRAKEN
#node-178        free node-178 376gb      48       0       4 node_group: GPU_PRIO


def Create_submision(save_dir,dataset,path_crops,optuna_folder,optuna_workers,optuna_trials,output_folder,file_name):
    job_name  = f'{file_name}'
    n = 0
    for node in ['node-177','node-178','node-180','node-181','node-182']:
        if node == 'node-178':
            qname = 'gpu_prio'
            host = 'node-178'
            nfiles=2

        elif node == 'node-177':
            qname = 'kraken'
            host = 'node-177'
            nfiles=2
            
        else:
            qname = 'kraken'
            host = node
            nfiles=2

        cpus='24'
        mem='125g'
        walltime = '100'
        script = textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q {qname}
        #PBS -l walltime={walltime}:00:00
        #PBS -N OptunaFibers
        #PBS -l select=1:ncpus={cpus}:ngpus=1:mem={mem}:host={host}
        #PBS -o {optuna_folder}
        #PBS -e {optuna_folder}
        echo "$(whoami)@$(hostname)"
        nvidia-smi
        . /home_local/cm283129/env_torch/bin/activate
        cd /neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Program/betaVAE
        python3 Optuna_tuning.py +save_dir={save_dir} +dataset={dataset} +path_crops={path_crops} +optuna_folder={optuna_folder} +optuna_workers={optuna_workers} +optuna_trials_per_worker={optuna_trials}
    """)
        for i in range(0,nfiles):
            with open(f"{output_folder}/{job_name}_{n}.sh", "w") as f:
                f.write(script)
            n+=1


#python3 Optuna_tuning.py +save_dir={save_dir} +dataset={dataset} +path_crops={path_crops} +optuna_folder={optuna_folder} +optuna_workers={optuna_workers} +optuna_trials_per_worker={optuna_trials}

def create_parser():
    parser = argparse.ArgumentParser(description="Generate kraken submision")
    # Add arguments
    parser.add_argument('-s', '--save_dir' , type=str, required=True)
    parser.add_argument('-d', '--dataset' , type=str, required=True)
    parser.add_argument('-p', '--path_crops' , type=str, required=True)
    parser.add_argument('-f', '--folder_optuna' , type=str, required=True)
    parser.add_argument('-w', '--workers_optuna' , type=int, required=True)
    parser.add_argument('-t', '--trials_optuna' , type=int, required=True)
    parser.add_argument('-o', '--output_folder' , type=str, required=True)
    parser.add_argument('-n', '--name'      , type=str, required=True)
    return parser


def main():
    # Create the parser and parse arguments
    try:
        print('Generating Kraken files')
        parser = create_parser()
        args = parser.parse_args()
        save_dir       = args.save_dir
        dataset        = args.dataset
        path_crops     = args.path_crops 
        optuna_folder  = args.folder_optuna
        optuna_workers = args.workers_optuna
        optuna_trials  = args.trials_optuna
        output_folder  = args.output_folder
        file_name      = args.name

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        Create_submision(save_dir,dataset,path_crops,optuna_folder,optuna_workers,optuna_trials,output_folder,file_name)

    except Exception as e:
        print(f'Exception {e}')


if __name__ == '__main__':
    main()
