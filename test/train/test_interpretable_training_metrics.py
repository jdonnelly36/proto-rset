from unittest.mock import patch

import pytest
import torch

from protopnet.train.metrics import InterpretableTrainingMetrics


class TestInterpretableTrainingMetrics:
    # Fixtures for InterpretableTrainingMetrics
    @pytest.fixture
    def protopnet_mock(self):
        class ProtoPNetMock:
            def __call__(self, x, return_prototype_layer_output_dict=False):
                return {"prototype_activations": torch.rand(len(x), 2, 1, 2, 2)}

            def get_prototype_complexity(self):
                return {
                    "n_unique_proto_parts": torch.tensor(8),
                    "n_unique_protos": torch.tensor(4),
                    "prototype_sparsity": torch.tensor(0.5),
                }

        return ProtoPNetMock()

    @pytest.fixture
    def training_metrics(self, protopnet_mock) -> InterpretableTrainingMetrics:
        return InterpretableTrainingMetrics(
            protopnet=protopnet_mock,
            num_classes=2,
            part_num=2,
            proto_per_class=1,
            img_size=224,
            half_size=36,
        )

    # Test that TrainingMetrics initializes correctly
    def test_TrainingMetrics_initialization(self, training_metrics):
        for metric in [
            "prototype_consistency",
            "prototype_stability",
            "prototype_sparsity",
            "accuracy",
        ]:
            assert metric in training_metrics.metrics

    def test_compute_dict(self, training_metrics):
        # Calculate the metrics from project first
        forward_args = {
            "img": torch.rand(2, 3, 224, 224),
            "target": torch.tensor([1, 0]),
            "sample_parts_centroids": [[], []],
            "sample_bounding_box": torch.randint(0, 224, (2, 4)),
        }

        forward_output = {
            "logits": torch.rand(2, 2),
            "prototype_activations": torch.rand(2, 2, 1, 2, 2),
        }

        training_metrics.update_all(forward_args, forward_output, phase="project")

        with patch(
            "protopnet.metrics.InterpMetrics.proto2part_and_masks",
            return_value=(torch.ones(224, 224), torch.ones(224, 224)),
        ):
            result = training_metrics.compute_dict()

        expected_keys = {
            "accuracy",
            "prototype_sparsity",
            "n_unique_protos",
            "n_unique_proto_parts",
            "prototype_consistency",
            "prototype_stability",
            "prototype_score",
            "acc_proto_score",
        }
        assert isinstance(result, dict)
        assert set(result.keys()) == expected_keys

        for key in expected_keys:
            assert isinstance(result["prototype_consistency"], torch.Tensor), (
                key,
                result[key],
            )
            if key.startswith("n_"):
                assert result[key] // 1 == result[key], key
            else:
                assert result[key] >= 0 and result[key] <= 1, key

        assert (
            result["prototype_score"]
            == (
                result["prototype_consistency"]
                + result["prototype_stability"]
                + result["prototype_sparsity"]
            )
            / 3
        )
        assert (
            result["acc_proto_score"] == result["accuracy"] * result["prototype_score"]
        )

        # update the metrics where only accuracy gets calculated
        # change the labels
        forward_args["target"] = torch.tensor([1, 1])
        training_metrics.update_all(forward_args, forward_output, phase="last_only")

        # no patch
        result_2 = training_metrics.compute_dict()

        for field in ["accuracy", "acc_proto_score"]:
            assert (
                result_2[field] != result[field]
            ), f"{field} should change when acc is recalculated"
            del result_2[field]
            del result[field]

        assert result_2 == result, "other results should be cached"
