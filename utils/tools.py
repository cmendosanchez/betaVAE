class EarlyStopping:
    def __init__(self, patience=5, delta=0, start_epoch=5, verbose=False):
        self.patience = patience
        self.delta = delta
        self.start_epoch = start_epoch
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss, epoch):

        # Do not activate early stopping yet
        if epoch < self.start_epoch:
            if self.best_loss is None or val_loss < self.best_loss:
                self.best_loss = val_loss
            return

        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")