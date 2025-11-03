

from .dataset import create_dataloader
from .training import TrainingModel
class EarlyStopping:
    def __init__(self, init_score=None, patience=1, verbose=False, delta=0):
        self.best_score = init_score
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.count_down = self.patience
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            if self.verbose:
                print(f'Score set to {score:.6f}.')
            self.best_score = score
            self.count_down = self.patience
            return True
        elif score <= self.best_score + self.delta:
            self.count_down -= 1
            if self.verbose:
                print(f'EarlyStopping count_down: {self.count_down} on {self.patience}')
            if self.count_down <= 0:
                self.early_stop = True
            return False
        else:
            if self.verbose:
                print(f'Score increased from ({self.best_score:.6f} to {score:.6f}).')
            self.best_score = score
            self.count_down = self.patience
            return True

    def reset_counter(self):
        self.count_down = self.patience
        self.early_stop = False
