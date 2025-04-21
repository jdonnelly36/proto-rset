import pytest
import torch

from protopnet.activations import CosPrototypeActivation, L2Activation


def test_cosine_activation():
    cosine_activation = CosPrototypeActivation()

    cosine_activations = cosine_activation(
        torch.randn(10, 512, 7, 7), torch.randn(15, 512, 1, 1)
    )

    assert cosine_activations.shape == (10, 15, 7, 7)
    assert cosine_activations.max().item() <= 1.0
    assert cosine_activations.min().item() >= 0.0


def test_l2_activation():
    l2_activation = L2Activation()

    l2_activations = l2_activation(
        torch.randn(10, 512, 7, 7), torch.randn(15, 512, 1, 1)
    )

    assert l2_activations.shape == (10, 15, 7, 7)
    assert l2_activations.max().item() <= 1.0
    assert l2_activations.min().item() >= 0.0


@pytest.mark.parametrize(
    "epsilon, activation_inclass, activation_outclass",
    [(1e-4, 9.2103, 0.5312), (1e-5, 11.5129, 0.5312)],
)
def test_l2_deterministic(epsilon, activation_inclass, activation_outclass):
    # Since images are 0s and 1s, we should know what the activations are with 0-1 prototypes
    l2_activation = L2Activation(epsilon_val=epsilon)

    prototypes01 = torch.zeros((2, 2, 1, 1))
    prototypes10 = torch.zeros((2, 2, 1, 1))

    prototypes01[:, 1, :, :] = 1

    prototypes10[:, 0, :, :] = 1

    image01 = torch.zeros((2, 2, 7, 7))
    image10 = torch.zeros((2, 2, 7, 7))

    image01[:, 1, :, :] = 1

    image10[:, 0, :, :] = 1

    assert l2_activation(image01, prototypes10).min().item() >= 0.0

    # Distance is estimated using L2 convolution; small numerical errors in distance lead to larger activation errors
    assert (
        abs(l2_activation(image01, prototypes10)[0, 0, 0, 0] - activation_outclass)
        < 0.2
    )
    assert (
        abs(l2_activation(image10, prototypes01)[0, 0, 0, 0] - activation_outclass)
        < 0.2
    )

    assert (
        abs(l2_activation(image01, prototypes01)[0, 0, 0, 0] - activation_inclass)
        < 1e-3
    )
    assert (
        abs(l2_activation(image10, prototypes10)[0, 0, 0, 0] - activation_inclass)
        < 1e-3
    )


def test_cos_deterministic():
    cos_activation = CosPrototypeActivation()

    prototypes01 = torch.zeros((2, 2, 1, 1))
    prototypes10 = torch.zeros((2, 2, 1, 1))

    prototypes01[:, 1, :, :] = 1

    prototypes10[:, 0, :, :] = 1

    image01 = torch.zeros((2, 2, 7, 7))
    image10 = torch.zeros((2, 2, 7, 7))

    image01[:, 1, :, :] = 1

    image10[:, 0, :, :] = 1

    assert cos_activation(image01, prototypes10).min().item() >= 0.0
    assert cos_activation(image01, prototypes10).min().item() <= 1.0

    assert abs(cos_activation(image01, prototypes10)[0, 0, 0, 0] - 0) < 1e-3
    assert abs(cos_activation(image10, prototypes01)[0, 0, 0, 0] - 0) < 1e-3

    assert abs(cos_activation(image01, prototypes01)[0, 0, 0, 0] - 1) < 1e-1
    assert abs(cos_activation(image10, prototypes10)[0, 0, 0, 0] - 1) < 1e-1


@pytest.mark.cuda
@pytest.mark.skip("needs a new training loop")
def test_cos_cpu_gpu_agreement(self):
    cpu = "cpu"
    gpu = "cuda"

    cpu_vppn = self.cos_vppn
    gpu_vppn = copy.deepcopy(self.cos_vppn)

    warm_pre_offset_optimizer_specs_cpu = [
        {
            "params": cpu_vppn.parameters(),
            "lr": 0.00001,
            "weight_decay": 1e-3,
        },
    ]

    warm_pre_offset_optimizer_specs_gpu = [
        {
            "params": gpu_vppn.parameters(),
            "lr": 0.00001,
            "weight_decay": 1e-3,
        },
    ]

    cpu_optimizer = torch.optim.Adam(warm_pre_offset_optimizer_specs_cpu)
    gpu_optimizer = torch.optim.Adam(warm_pre_offset_optimizer_specs_gpu)

    # TODO: train on cpu
    # TODO: train on gpu

    cpu_weights = torch.nn.utils.parameters_to_vector(cpu_vppn.parameters()).to(gpu)
    gpu_weights = torch.nn.utils.parameters_to_vector(gpu_vppn.parameters())

    assert torch.allclose(cpu_weights, gpu_weights, atol=1e-4)
