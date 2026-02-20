import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf, plot_timeline
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_contour
print('optuna')
study_path = '/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/OptunaResults/two_ends_R_S.C.-sylv._Track_0_80_sift2_icbm09c_trial_example2'
#study = optuna.load_study(
#    study_name = "trial_7_allnodes",
#    storage=JournalStorage(JournalFileBackend(f"{study_path}/journal_gpu_prio.log"))
#)

study = optuna.load_study(study_name = "example2_study",storage="mysql+mysqlconnector://gaia:Optima1Pass!@rosette:3306/example2")




fig = plot_optimization_history(study)
fig.write_image(f"{study_path}/1.png")
fig2 = plot_param_importances(study)
fig2.write_image(f"{study_path}/2.png")
fig3 = plot_timeline(study)
fig3.write_image(f"{study_path}/3.png")
fig4 = plot_intermediate_values(study)
fig4.write_image(f"{study_path}/4.png")
fig5 = plot_contour(study,params=["LEARNING_RATE", 'BETA'])
fig5.write_image(f"{study_path}/5.png")
fig5 = plot_contour(study,params=["LEARNING_RATE", 'LATENT_DIMENSIONS'])
fig5.write_image(f"{study_path}/6.png")

def print_trial_info(study):
    for trial in study.trials:
        state = trial.state
        
        # Print the state as a string (from the enumeration)
        state_str = state.name if state is not None else "None"
        
        print(f"Trial {trial.number}:")
        print(f"  State: {state_str} ({state})")  # State as both enum name and integer value
        print(f"  Value: {trial.value}")
        print(f"  Parameters: {trial.params}")
        print(f"  Intermediate values: {trial.intermediate_values}")
        print(f"  Date created: {trial.datetime_start}")
        print(f"  Date completed: {trial.datetime_complete}")
        print("-" * 40)

    
print_trial_info(study)
print("Number of trials:", len(study.trials))
print("Best value:", study.best_value)
print("Best params:", study.best_params)

# Get parameters sorted by the importance values
importances = optuna.importance.get_param_importances(study)
params_sorted = list(importances.keys())

# Plot
fig6 = optuna.visualization.plot_rank(study, params=params_sorted[:4])
fig6.write_image(f"{study_path}/6.png")
fig7 = plot_contour(study,params=["BETA", 'N_SUBJECTS'])
fig7.write_image(f"{study_path}/7.png")
