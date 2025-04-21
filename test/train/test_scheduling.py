from unittest.mock import MagicMock

import pytest

from protopnet.train.scheduling import (
    EarlyStopping,
    ProtoPNetBackpropEpoch,
    ProtoPNetProjectEpoch,
    prototype_embedded_epoch,
)


class TestEarlyStopping:
    @pytest.fixture
    def min_mode_stopper(self):
        metric_source = MagicMock(return_value=0)
        return EarlyStopping(
            patience=3, min_delta=0.1, mode="min", metric_source=metric_source
        )

    @pytest.fixture
    def max_mode_stopper(self):
        metric_source = MagicMock(return_value=0)
        return EarlyStopping(
            patience=2, min_delta=0.1, mode="max", metric_source=metric_source
        )

    def test_initialization_min_mode(self, min_mode_stopper: EarlyStopping):
        assert min_mode_stopper.best == float(
            "inf"
        ), "Initial best should be inf for min mode"

    def test_initialization_max_mode(self, max_mode_stopper: EarlyStopping):
        assert max_mode_stopper.best == float(
            "-inf"
        ), "Initial best should be -inf for max mode"

    def test_check_method_improvement(self, min_mode_stopper: EarlyStopping):
        min_mode_stopper.metric_source.return_value = 0.5
        assert (
            not min_mode_stopper.check()
        ), "Stop should be False when improvement is observed"
        assert min_mode_stopper.best == 0.5, "Best should update to the new best value"
        assert min_mode_stopper.counter == 0, "Counter should reset after improvement"

    def test_check_method_no_improvement(self, min_mode_stopper: EarlyStopping):
        min_mode_stopper.metric_source.return_value = 0.5
        min_mode_stopper.check()  # First check to set the best
        min_mode_stopper.metric_source.return_value = 0.6
        min_mode_stopper.check()  # No improvement
        assert (
            min_mode_stopper.counter == 1
        ), "Counter should increment without improvement"

    def test_stopping_condition_triggered(self, min_mode_stopper: EarlyStopping):
        min_mode_stopper.metric_source.return_value = 0.5
        min_mode_stopper.check()  # Set initial best
        min_mode_stopper.metric_source.return_value = 0.6
        for _ in range(3):  # Exceed patience
            min_mode_stopper.check()
        assert min_mode_stopper.stop, "Stop should be True after patience is exceeded"

    def test_reset_functionality(self, min_mode_stopper: EarlyStopping):
        min_mode_stopper.metric_source.return_value = 0.5
        min_mode_stopper.check()  # Set initial values
        min_mode_stopper.reset()
        assert min_mode_stopper.best == float(
            "inf"
        ), "Best should reset to inf for min mode"
        assert min_mode_stopper.counter == 0, "Counter should reset to 0"
        assert not min_mode_stopper.stop, "Stop should reset to False"


@pytest.mark.parametrize(
    "backbone,add_on_layers,prototype,conv_offset,head,expected",
    [
        (True, True, True, True, True, False),
        (True, True, True, True, False, False),
        (True, False, False, False, False, False),
        (False, False, True, False, False, False),
        (False, False, False, False, False, True),
        (False, False, False, False, True, True),
    ],
)
def test_prototype_embedded_epoch_backprop(
    backbone: bool,
    add_on_layers: bool,
    prototype: bool,
    conv_offset: bool,
    head: bool,
    expected: bool,
):
    layer = ProtoPNetBackpropEpoch(
        phase="something",
        train_backbone=backbone,
        train_add_on_layers=add_on_layers,
        train_prototype_layer=prototype,
        train_conv_offset=conv_offset,
        train_prototype_prediction_head=head,
    )

    assert (
        prototype_embedded_epoch(layer) == expected
    ), f"Expected {expected} but got {prototype_embedded_epoch(layer)}"


def test_prototype_embedded_epoch_project():
    layer = ProtoPNetProjectEpoch()

    assert (
        prototype_embedded_epoch(layer) == True
    ), f"Expected True but got {prototype_embedded_epoch(layer)}"
