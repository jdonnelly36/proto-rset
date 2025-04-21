from .ham10000 import HAM10000, train_dataloaders
import logging

log = logging.getLogger(__name__)


class HAM10000_7class(HAM10000):
    def __init__(self, df, transform=None):
        super().__init__(df, transform)
        self.labels = self.df["diagnosis_2"].values


# Need to import traindataloader from HAM10000 so that
# our scripts know to run with the same loading script
# as HAM10000. But the linter fails if we import
# train_dataloaders without calling it.
# So we have a dummy call to it here...
def _implicit_train_dataloaders():
    train_dataloaders()
