import argparse
import os
import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from colors import bcolors

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

    print(best_params)
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