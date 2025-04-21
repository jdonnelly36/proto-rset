import torch
import torch.nn as nn
import torch.nn.functional as F


class CosPrototypeActivation:
    """
    Computes the cosine activation (arc distance) between convolutional features as in
        https://arxiv.org/pdf/1801.07698.pdf
    """

    def __init__(
        self,
        relu_on_cos: bool = True,
        normalization_constant: int = 64,
        episilon_val: float = 1e-4,
    ):
        """
        Args:
            margin: Margin for the cosine similarity. If None, then no margin is used.
            relu_on_cos: Whether to apply a ReLU on the cosine similarity. If False, then the cosine similarity is
                returned as is.
            normalization_constant: The normalization constant for the cosine similarity. This is used to scale the
                cosine similarity to a reasonable range. The default value of 64 is chosen to be consistent with the
                original ProtoPNet implementation.
            episilon_val: A small value to prevent division by zero.
        """
        self.relu_on_cos = relu_on_cos
        self.epsilon_val = episilon_val
        self.input_vector_length = normalization_constant
        self.normalization_constant = normalization_constant

    def _normalize(self, x: torch.Tensor, prototype_tensor: torch.Tensor):
        normalizing_factor = (
            prototype_tensor.shape[-2] * prototype_tensor.shape[-1]
        ) ** 0.5

        x_length = torch.sqrt(torch.sum(torch.square(x), dim=-3) + self.epsilon_val)
        x_length = x_length.view(
            x_length.size()[0], 1, x_length.size()[1], x_length.size()[2]
        )
        x_normalized = self.normalization_constant * x / x_length
        x_normalized = x_normalized / normalizing_factor
        return x_normalized, normalizing_factor

    def __call__(
        self,
        x: torch.Tensor,
        prototype_tensor: torch.Tensor,
    ):
        """
        Args:
            x: The input tensor of shape (batch_size, feature_dim, latent_height, latent_width)
            prototype_tensor: The prototype tensor of shape (num_prototypes, feature_dim, latent_height, latent_width)
            prototypes_of_wrong_class: The prototypes of the wrong class. This is used for the margin loss.

        Returns: activations (torch.Tensor): Tensor of the activations. This is of shape (batch_size, num_prototypes, activation_height, activation_width).
        """

        x_normalized, normalizing_factor = self._normalize(x, prototype_tensor)

        # We normalize prototypes to unit length
        prototype_vector_length = torch.sqrt(
            torch.sum(torch.square(prototype_tensor), dim=-3) + self.epsilon_val
        )
        prototype_vector_length = prototype_vector_length.view(
            prototype_vector_length.size()[0],
            1,
            prototype_vector_length.size()[1],
            prototype_vector_length.size()[2],
        )
        normalized_prototypes = prototype_tensor / (
            prototype_vector_length + self.epsilon_val
        )
        normalized_prototypes = normalized_prototypes / normalizing_factor

        if x_normalized.device != normalized_prototypes.device:
            normalized_prototypes = normalized_prototypes.to(x_normalized.device)
        activations_dot = F.conv2d(x_normalized, normalized_prototypes)

        renormed_activations = activations_dot / (self.normalization_constant * 1.01)

        if self.relu_on_cos:
            renormed_activations = torch.relu(renormed_activations)

        return renormed_activations


class L2Activation:
    def __init__(self, epsilon_val=1e-4):
        self.epsilon_val = epsilon_val

    def __call__(
        self,
        x: torch.Tensor,
        prototype_tensors: torch.Tensor,
    ):
        x2 = x**2
        ones = torch.ones(prototype_tensors.shape, requires_grad=False).to(
            prototype_tensors.device
        )

        x2_patch_sum = F.conv2d(input=x2, weight=ones)

        p2 = prototype_tensors**2
        # TODO: Support more dimensions
        p2 = torch.sum(p2, dim=(1, 2, 3))

        # Reshape from (num_prototypes,) to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_tensors)
        intermediate_result = -2 * xp + p2_reshape

        distances = x2_patch_sum + intermediate_result

        distances = F.relu(distances)
        activations = torch.log((distances + 1) / (distances + self.epsilon_val))

        return activations


class ConvolutionalSharedOffsetPred(nn.Module):
    """
    Computes the activation for a deformable prototype as in
        https://arxiv.org/pdf/1801.07698.pdf, but with renormalization
        after deformation instead of norm-preserving interpolation.
    """

    def __init__(
        self,
        prototype_shape: tuple,
        input_feature_dim: int = 512,
        kernel_size: int = 3,
        prototype_dilation: int = 1,
    ):
        """
        Args:
            prototype_shape: The shape of the prototypes the convolution will be applied to
            input_feature_dim: The expected latent dimension of the input
            kernel_size: The size of the kernel used for offset prediction
        """
        assert (kernel_size % 2 == 1) or (
            prototype_dilation % 2 == 0
        ), f"Error: kernel size {kernel_size} with dilation {prototype_dilation} is not supported because even kernel sizes without even dilation break symmetric padding"
        assert (
            len(prototype_shape) == 4
        ), "Error: Code assumes prototype_shape is a (num_protos, channel, height, width) tuple."

        super(ConvolutionalSharedOffsetPred, self).__init__()
        self.prototype_shape = prototype_shape

        self.prototype_dilation = prototype_dilation

        # Compute out channels as 2 * proto_h * proto_w
        out_channels = 2 * prototype_shape[2] * prototype_shape[3]
        self.offset_predictor = torch.nn.Conv2d(
            input_feature_dim,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        torch.nn.init.zeros_(self.offset_predictor.weight)
        self._init_offset_bias()

    def _init_offset_bias(self):
        # out channels is ordered as (tl_x, tl_y, tm_x, tm_y, ...)
        # Initialize our offset predictor to put us at normal grid sample locations
        new_bias = torch.zeros_like(self.offset_predictor.bias)
        for py in range(self.prototype_shape[-2]):
            for px in range(self.prototype_shape[-1]):
                new_bias[
                    (py * self.prototype_shape[-2] + px) * 2 + 1
                ] = self.prototype_dilation * (py - (self.prototype_shape[-2] - 1) / 2)
                new_bias[
                    (py * self.prototype_shape[-2] + px) * 2
                ] = self.prototype_dilation * (px - (self.prototype_shape[-1] - 1) / 2)

        self.offset_predictor.bias = torch.nn.Parameter(new_bias).to(
            self.offset_predictor.bias.device
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Args:
            x: The input tensor of shape (batch_size, feature_dim, latent_height, latent_width)

        Returns: activations (torch.Tensor): Tensor of the activations. This is of shape (batch_size, num_prototypes, activation_height, activation_width).
        """
        # predicted_offsets will be (batch, 2 * proto_h * proto_w, height, width)
        predicted_offsets = self.offset_predictor(x)

        return predicted_offsets
