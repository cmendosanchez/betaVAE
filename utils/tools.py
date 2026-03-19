import torch
import copy

class EarlyStopping:
    def __init__(self, patience=5, delta=0, start_epoch=4, verbose=False,
                 save_best=False, path="best_model.pt"):
        
        self.patience = patience
        self.delta = delta
        self.start_epoch = start_epoch
        self.verbose = verbose
        
        self.save_best = save_best
        self.path = path
        
        self.best_loss = None
        self.best_model_state = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss, epoch, model):

        # Do not activate early stopping yet
        if epoch < self.start_epoch:
            if self.best_loss is None or val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save(model)
            return

        # Improvement
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            self._save(model)

        # No improvement
        else:
            self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

    def _save(self, model):
        if self.save_best:
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print("✅ Best model saved")
        else:
            # store in memory instead
            self.best_model_state = copy.deepcopy(model.state_dict())