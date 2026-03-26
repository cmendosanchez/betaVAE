import torch
import copy

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, start_epoch=5, verbose=False,
                 save_best=True, path="best_model.pt"):

        self.patience = patience
        self.delta = delta
        self.start_epoch = start_epoch
        self.verbose = verbose

        self.save_best = save_best
        self.path = path

        self.best_loss = float("inf")
        self.best_epoch = -1

        self.best_model_state = None
        self.best_optimizer_state = None

        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss, epoch, model, optimizer):

        # Always track best during warm-up, but don't early stop
        if epoch < self.start_epoch:
            if val_loss < self.best_loss:
                self._update_best(val_loss, epoch, model, optimizer)
            return False

        # Improvement condition
        if val_loss < self.best_loss - self.delta:
            self._update_best(val_loss, epoch, model, optimizer)
            self.no_improvement_count = 0

        else:
            self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}. "
                          f"Best epoch was {self.best_epoch} "
                          f"with loss {self.best_loss:.6f}")

        return self.stop_training

    def _update_best(self, val_loss, epoch, model, optimizer):
        self.best_loss = val_loss
        self.best_epoch = epoch

        #ALWAYS save on improvement
        if self.save_best:
            torch.save({
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, self.path)

            if self.verbose:
                print(f"[Epoch {epoch}] New best loss: {val_loss:.6f} → model saved")

        else:
            # store in memory
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_optimizer_state = copy.deepcopy(optimizer.state_dict())

            if self.verbose:
                print(f"[Epoch {epoch}] New best loss: {val_loss:.6f} → stored in memory")