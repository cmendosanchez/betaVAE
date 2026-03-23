import argparse
import os
import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from colors import bcolors
from optuna.trial import TrialState
from collections import defaultdict
import numpy as np

#python OptunaParamsTable.py ../../../../OptunaResults --study-name journal_storage_multiprocess  --out ../../../../OptunaResults/summary.csv --regions S.C.-sylv._left S.C.-sylv._right S.T.s._left S.T.s._right S.F.int.-F.C.M.ant._right S.F.int.-F.C.M.ant._left  --modes SWM DWM Comm

def avg_auc(t):
    over_auc = t.user_attrs.get("Overconnectivity")
    under_auc = t.user_attrs.get("Underconnectivity")
    
    # Handle missing values safely
    if over_auc is None:
        return under_auc
    elif under_auc is None:
        return over_auc
    
    return (over_auc + under_auc) / 2

def average_num_steps(study):
    step_counts = []

    for trial in study.trials:
        if len(trial.intermediate_values) > 0:
            step_counts.append(len(trial.intermediate_values))

    if len(step_counts) == 0:
        return None

    return sum(step_counts) / len(step_counts)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root folder containing experiment subfolders")
    parser.add_argument("--study-name",type=str, required=True)
    parser.add_argument("--database",type=str, default='UKB')
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", default=["SWM", "DWM", "Comm"])
    parser.add_argument("--out", default="optuna_summary.csv")


    args = parser.parse_args()


    best_params = {}

    for region in args.regions:
        best_params[region] = {}

        for mode in args.modes:

            study_folder = f'{args.root}/{args.database}_{region}_{mode}'
            print(f'{bcolors.GREEN}{study_folder}/journal.log{bcolors.RESET}')
            storage = JournalStorage(JournalFileBackend(f'{study_folder}/journal.log'))

            study = optuna.load_study(
                study_name=args.study_name,
                storage=storage)
            params = study.best_trial.params
            #print(params)

            best_params[region][mode] = {}
            avg_steps = average_num_steps(study)
            #print(f"{bcolors.YELLOW}Average steps per trial: {avg_steps:.2f}{bcolors.RESET}")

            #print(best_params,f'{bcolors.YELLOW}{best_trial.last_step}{bcolors.RESET}')

            completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
            # Ordenar trials por valor objetivo (mayor es mejor; usa reverse=False si minimizas)
            completed_trials = sorted(
                completed_trials,
                key=lambda t: t.value
            )[0:5]
            #top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]

            for t in completed_trials:
                over_auc = t.user_attrs.get("Overconnectivity")
                under_auc = t.user_attrs.get("Underconnectivity")
                avg = avg_auc(t)
                print(f'{bcolors.RED}{t.params} {t.value} AUC under:{under_auc} AUC over:{over_auc} AVG:{avg:.4f} {bcolors.RESET}')
                
                #print(f"{bcolors.YELLOW}Trial {t.number} | value = {t.value} | Overconnectivity AUC = {val}{bcolors.RESET}")
            best_trial_auc = max(completed_trials, key=avg_auc)
            print(f'{bcolors.GREEN}Best trial {best_trial_auc}{bcolors.RESET}')
            best_params[region][mode]['Epochs']    = best_trial_auc.last_step
            best_params[region][mode]['Recon Error']   = best_trial_auc.value
            best_params[region][mode]['Trial id']   = best_trial_auc.number
            #best_params[region][mode]['Status']  = study.best_trial.state
            n_completed = sum(t.state == TrialState.COMPLETE for t in study.trials)
            best_params[region][mode]['Completed Trials']  = n_completed
            best_params[region][mode]['Learning Rate'] = best_trial_auc.params['Learning Rate']
            best_params[region][mode]['Weight decay']  = best_trial_auc.params['Weight decay']
            best_params[region][mode]['Dimensions']  = best_trial_auc.params['Dimensions']
            best_params[region][mode]['Beta']  = best_trial_auc.params['Beta']
            best_params[region][mode]['Batch size']  = best_trial_auc.params['Batch size']
            best_params[region][mode]['AUC Over']  = best_trial_auc.user_attrs['Overconnectivity']
            best_params[region][mode]['AUC Under']  = best_trial_auc.user_attrs['Underconnectivity']
    rows = []

    for region in best_params:
        for mode in best_params[region]:
            row = {"Region": region, "Seg. Criteria": mode}
            row.update(best_params[region][mode])
            rows.append(row)

    df = pd.DataFrame(rows)
    df["Learning Rate"] = df["Learning Rate"].map("{:.2e}".format)
    df["Weight decay"]  = df["Weight decay"].map("{:.2e}".format)
    print(df)
    #print(df.sort_values(by="Region"))
    df_sorted = df.sort_values(by="Region")
    #col = df_sorted.pop("Trial number")
    col = df_sorted.pop("Completed Trials")
    df_sorted.insert(2, "Completed Trials", col)
    print(df_sorted)
    df_sorted.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()