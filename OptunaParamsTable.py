import argparse
import os
import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from colors import bcolors
from optuna.trial import TrialState


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
            print(params)

            best_params[region][mode] = params
            best_trial = study.best_trial
            best_params[region][mode]['Step']  = best_trial.last_step
            best_params[region][mode]['Value'] = best_trial.value
            best_params[region][mode]['Trial'] =    study.best_trial.number
            best_params[region][mode]['Status']                               = study.best_trial.state
            n_completed = sum(t.state == TrialState.COMPLETE for t in study.trials)
            best_params[region][mode]['Completed Trials']  = n_completed

            print(best_params,f'{bcolors.YELLOW}{best_trial.last_step}{bcolors.RESET}')

            # Ordenar trials por valor objetivo (mayor es mejor; usa reverse=False si minimizas)
            top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]

            for t in top_trials:
                over_auc = t.user_attrs.get("Overconnectivity")
                under_auc = t.user_attrs.get("Underconnectivity")
                print(over_auc,under_auc)
                #print(f"{bcolors.YELLOW}Trial {t.number} | value = {t.value} | Overconnectivity AUC = {val}{bcolors.RESET}")


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
    df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()