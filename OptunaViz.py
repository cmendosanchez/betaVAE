import argparse
import os
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_parallel_coordinate,
    plot_intermediate_values,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Optuna plots from a journal.log file"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing journal.log",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name (if multiple studies exist)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="optuna_figures",
        help="Output directory for figures",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    journal_path = os.path.join(args.folder, "journal.log")
    if not os.path.isfile(journal_path):
        raise FileNotFoundError(f"journal.log not found in {args.folder}")

    os.makedirs(args.outdir, exist_ok=True)

    storage = JournalStorage(JournalFileBackend(journal_path))

    # Load study
    if args.study_name is None:
        studies = optuna.get_all_study_names(storage)
        if len(studies) == 0:
            raise RuntimeError("No studies found in journal.log")
        study_name = studies[0]
        print(f"Using study: {study_name}")
    else:
        study_name = args.study_name

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )

    """ # --- Optimization history ---
    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(args.outdir, "optimization_history.png"))

    # --- Parameter importance ---
    fig = plot_param_importances(study)
    fig.write_image(os.path.join(args.outdir, "param_importance.png"))

    # --- Contour plot (top 2 parameters) ---
    if len(study.best_params) >= 2:
        params = list(study.best_params.keys())[:2]
        fig = plot_contour(study, params=params)
        fig.write_image(os.path.join(args.outdir, "contour.png"))

    # --- Parallel coordinate ---
    fig = plot_parallel_coordinate(study)
    fig.write_image(os.path.join(args.outdir, "parallel_coordinate.png"))

    # --- Intermediate values (if present) ---
    try:
        fig = plot_intermediate_values(study)
        fig.write_image(os.path.join(args.outdir, "intermediate_values.png"))
    except ValueError:
        print("No intermediate values found, skipping.")
 """
    print(f"Figures saved in: {args.outdir}")
    best_trial = study.best_trials

    print("\n=== BEST TRIAL ===")
    """ print(f"Trial number: {best_trial.number}")
    print(f"Best value: {best_trial.value}")
    print("Best parameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")"""
    for trial in study.best_trials:
        print(f"\nTrial number: {trial.number}")
        print(f"Objective values: {trial.values}")  # tuple
        print("Parameters:")
        for k, v in trial.params.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()