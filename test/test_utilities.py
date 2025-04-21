import os
import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

from protopnet.utilities.general_utilities import check_pip_environment, parse_yaml_file
from protopnet.utilities.project_utilities import custom_unravel_index, hash_func
from protopnet.utilities.trainer_utilities import get_learning_rates
from protopnet.utilities.visualization_utilities import indices_to_upsampled_boxes


class TestHashFunction(unittest.TestCase):
    def test_hash_function_returns_correct_type(self):
        tensor = torch.randn(3, 224, 224)
        assert isinstance(hash_func(tensor), str), "Hash output should be a string"

    def test_hash_function_consistency(self):
        tensor = torch.randn(10, 10)
        hash1 = hash_func(tensor)
        hash2 = hash_func(tensor)
        assert hash1 == hash2, "Hashes should be identical for the same input"

    def test_hash_is_true_function(self):
        tensor1 = torch.arange(100).reshape(2, 5, 10)
        tensor2 = torch.arange(100).reshape(2, 5, 10)
        hash1 = hash_func(tensor1)
        hash2 = hash_func(tensor2)
        assert hash1 == hash2, "Hashes should be identical for the same input"

    def test_hash_function_different_tensors(self):
        tensor1 = torch.randn(10, 10)
        tensor2 = torch.randn(10, 10)
        assert hash_func(tensor1) != hash_func(
            tensor2
        ), "Hashes should be different for different inputs"

    def test_hash_function_empty_tensor(self):
        empty_tensor = torch.tensor([])
        assert isinstance(
            hash_func(empty_tensor), str
        ), "Empty tensor should return a string hash"


class TestCustomUnravelIndex(unittest.TestCase):
    # Test cases from https://pytorch.org/docs/stable/generated/torch.unravel_index.html

    def test_single_index(self):
        result = custom_unravel_index(torch.tensor(4), (3, 2))
        expected = (torch.tensor(2), torch.tensor(0))
        self.assertTrue(
            torch.equal(result[0], expected[0]) and torch.equal(result[1], expected[1])
        )

    def test_multiple_indices(self):
        result = custom_unravel_index(torch.tensor([4, 1]), (3, 2))
        expected = (torch.tensor([2, 0]), torch.tensor([0, 1]))
        self.assertTrue(all(torch.equal(r, e) for r, e in zip(result, expected)))

    def test_full_range_indices(self):
        result = custom_unravel_index(torch.tensor([0, 1, 2, 3, 4, 5]), (3, 2))
        expected = (torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 1, 0, 1, 0, 1]))
        print(result, expected)
        self.assertTrue(all(torch.equal(r, e) for r, e in zip(result, expected)))

    def test_large_indices(self):
        result = custom_unravel_index(torch.tensor([1234, 5678]), (10, 10, 10, 10))
        expected = (
            torch.tensor([1, 5]),
            torch.tensor([2, 6]),
            torch.tensor([3, 7]),
            torch.tensor([4, 8]),
        )
        self.assertTrue(all(torch.equal(r, e) for r, e in zip(result, expected)))

    def test_large_indices_2d_input(self):
        result = custom_unravel_index(torch.tensor([[1234], [5678]]), (10, 10, 10, 10))
        expected = (
            torch.tensor([[1], [5]]),
            torch.tensor([[2], [6]]),
            torch.tensor([[3], [7]]),
            torch.tensor([[4], [8]]),
        )
        self.assertTrue(all(torch.equal(r, e) for r, e in zip(result, expected)))

    def test_large_indices_different_shape(self):
        result = custom_unravel_index(torch.tensor([[1234], [5678]]), (100, 100))
        expected = (torch.tensor([[12], [56]]), torch.tensor([[34], [78]]))
        self.assertTrue(all(torch.equal(r, e) for r, e in zip(result, expected)))


class TestParseYamlFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary YAML file
        temp_path = Path("test/tmp")
        temp_path.mkdir(parents=True, exist_ok=True)
        self.sample_yaml_file = temp_path / "sample.yaml"
        with open(self.sample_yaml_file, "w") as f:
            f.write(
                """
name: Bob
age: 25
is_student: true
courses:
    - Mathematics
    - History
"""
            )

    def tearDown(self):
        # Remove temporary YAML file after test
        os.remove(self.sample_yaml_file)

    def test_yaml_file_parsed_correctly(self):
        args = Namespace()
        parsed_args = parse_yaml_file(self.sample_yaml_file, args)

        self.assertEqual(parsed_args.name, "Bob")
        self.assertEqual(parsed_args.age, 25)
        self.assertEqual(parsed_args.is_student, True)
        self.assertListEqual(parsed_args.courses, ["Mathematics", "History"])

    def test_no_yaml_file(self):
        args = Namespace()
        args.name = "Alice"
        parsed_args = parse_yaml_file(None, args)

        # Args should remain unchanged
        self.assertEqual(parsed_args.name, "Alice")

    def test_overwrite_existing_args(self):
        args = Namespace()
        args.name = "Alice"
        args.major = "Protocol Engineering"
        parsed_args = parse_yaml_file(self.sample_yaml_file, args)

        # Args should be updated by the YAML file
        self.assertEqual(parsed_args.name, "Bob")
        self.assertEqual(parsed_args.major, "Protocol Engineering")
        self.assertEqual(parsed_args.age, 25)


class TestEnvironment(unittest.TestCase):
    def test_environment(self):
        test_reqs = (Path("test") / "tmp" / "requirements-frozen.txt").resolve()

        with open(test_reqs, "w") as test_reqs_file:
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze"], stdout=test_reqs_file
            )
        expected, installed, differences = check_pip_environment(
            requirements_file=test_reqs
        )
        assert type(differences) == list
        assert len(differences) == 0, (len(expected), len(installed), differences)


def test_indices_to_boxes():
    image = torch.randn(3, 100, 100)
    proto_acts = torch.randn(10, 2, 2)
    box_coords = indices_to_upsampled_boxes(
        -1 * torch.ones(2), proto_acts.shape[1:], image.shape[1:], align_corners=True
    )
    assert box_coords[0][0] == 0
    assert box_coords[0][1] == 0
    assert box_coords[1][0] == 50
    assert box_coords[1][1] == 50

    box_coords = indices_to_upsampled_boxes(
        torch.zeros(2), proto_acts.shape[1:], image.shape[1:], align_corners=True
    )
    assert box_coords[0][0] == 25
    assert box_coords[0][1] == 25
    assert box_coords[1][0] == 75
    assert box_coords[1][1] == 75

    box_coords = indices_to_upsampled_boxes(
        -1 * torch.ones(2), proto_acts.shape[1:], image.shape[1:], align_corners=False
    )
    assert box_coords[0][0] == -50
    assert box_coords[0][1] == -50
    assert box_coords[1][0] == 50
    assert box_coords[1][1] == 50

    box_coords = indices_to_upsampled_boxes(
        torch.zeros(2), proto_acts.shape[1:], image.shape[1:], align_corners=True
    )
    assert box_coords[0][0] == 25
    assert box_coords[0][1] == 25
    assert box_coords[1][0] == 75
    assert box_coords[1][1] == 75

    box_coords = indices_to_upsampled_boxes(
        -1 * torch.ones(2), proto_acts.shape[1:], image.shape[1:], align_corners=False
    )
    assert box_coords[0][0] == -50
    assert box_coords[0][1] == -50
    assert box_coords[1][0] == 50
    assert box_coords[1][1] == 50

    box_coords = indices_to_upsampled_boxes(
        torch.zeros(2), proto_acts.shape[1:], image.shape[1:], align_corners=True
    )
    assert box_coords[0][0] == 25
    assert box_coords[0][1] == 25
    assert box_coords[1][0] == 75
    assert box_coords[1][1] == 75

    box_coords = indices_to_upsampled_boxes(
        -1 * torch.ones(2), proto_acts.shape[1:], image.shape[1:], align_corners=False
    )
    assert box_coords[0][0] == -50
    assert box_coords[0][1] == -50
    assert box_coords[1][0] == 50
    assert box_coords[1][1] == 50


class SimplerModel(torch.nn.Module):
    def __init__(self):
        super(SimplerModel, self).__init__()
        self.fc1 = torch.nn.Linear(50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        pass


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.backbone = SimplerModel()

        self.fc1 = torch.nn.Linear(50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        pass


def test_different_learning_rates():
    model = SimpleModel()
    optimizer = torch.optim.SGD(
        [
            {"params": model.backbone.parameters(), "lr": 0.01},
            {"params": model.fc1.parameters(), "lr": 0.001},
            {"params": model.fc2.parameters(), "lr": 0.0005},
        ],
        momentum=0.9,
    )

    lr_log_detailed = get_learning_rates(optimizer, model, detailed=True)
    lr_log_not_detailed = get_learning_rates(optimizer, model, detailed=False)

    assert (
        lr_log_detailed["lr.backbone.fc1"] == 0.01
    ), "Incorrect learning rate for backbone.fc1"
    assert (
        lr_log_detailed["lr.backbone.fc2"] == 0.01
    ), "Incorrect learning rate for backbone.fc2"
    assert lr_log_detailed["lr.fc1"] == 0.001, "Incorrect learning rate for fc1"
    assert lr_log_detailed["lr.fc2"] == 0.0005, "Incorrect learning rate for fc2"

    assert (
        lr_log_not_detailed["lr.backbone"] == 0.01
    ), "Incorrect learning rate for backbone"
    assert lr_log_not_detailed["lr.fc1"] == 0.001, "Incorrect learning rate for fc1"
    assert lr_log_not_detailed["lr.fc2"] == 0.0005, "Incorrect learning rate for fc2"

    expected_layers = ["backbone.fc1", "backbone.fc2", "fc1", "fc2"]
    for layer in expected_layers:
        assert f"lr.{layer}" in lr_log_detailed, f"{layer} is missing from lr_dict"

    expected_layers = ["backbone", "backbone", "fc1", "fc2"]
    for layer in expected_layers:
        assert f"lr.{layer}" in lr_log_not_detailed, f"{layer} is missing from lr_dict"
