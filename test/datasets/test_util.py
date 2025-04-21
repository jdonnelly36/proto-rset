import pytest

import protopnet.datasets
import protopnet.datasets.cars
import protopnet.datasets.cub200
from protopnet.datasets.util import (
    load_train_dataloaders_method,  # replace 'your_module' with the actual name of your module
)


@pytest.mark.parametrize(
    "module_name,module_method",
    [
        ("cub200", protopnet.datasets.cub200.train_dataloaders),
        ("CUB200", protopnet.datasets.cub200.train_dataloaders),
        ("cars", protopnet.datasets.cars.train_dataloaders),
    ],
)
def test_load_train_dataloaders_method(module_name, module_method):
    # Test loading from cub200
    module_train_method = load_train_dataloaders_method(protopnet.datasets, module_name)
    assert callable(module_train_method)
    assert module_method == module_train_method


def test_failure_of_nonexistant_module():
    # Test non-existing module
    with pytest.raises(ImportError):
        load_train_dataloaders_method(protopnet.datasets, "nonexistent")


def test_calling_test_load_train_dataloaders_method_through_init():
    import protopnet.datasets

    protopnet.datasets.training_dataloaders(
        dataset_name="cub200", data_path="test/dummy_test_files/test_dataset"
    )


if __name__ == "__main__":
    pytest.main()
