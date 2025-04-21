import importlib
import logging

from . import util

log = logging.getLogger(__name__)


def training_dataloaders(dataset_name: str, **kwargs):
    log.info("loading dataloaders for %s", dataset_name)
    return util.load_train_dataloaders_method(
        importlib.import_module(__package__), module_name=dataset_name
    )(**kwargs)
