import hashlib
import io
import torch
import itertools
import operator
from typing import Sequence, Tuple, Union


def hash_func(img: torch.tensor):
    """
    Takes in a tensor, outputs hash of that tensor as string.
    """
    buffer = io.BytesIO()
    torch.save(img, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def custom_unravel_index(
    indices: torch.Tensor, shape: Union[int, Sequence[int], torch.Size]
) -> Tuple[torch.Tensor, ...]:
    """
    Converts a tensor of flat indices into a tuple of coordinate tensors.
    """
    # Validate input tensor type
    if (
        not indices.dtype.is_floating_point
        and indices.dtype != torch.bool
        and not indices.is_complex()
    ):
        pass
    else:
        raise ValueError("expected 'indices' to be an integer tensor")

    # Ensure shape is in correct format
    if isinstance(shape, int):
        shape = torch.Size([shape])
    elif isinstance(shape, Sequence):
        for dim in shape:
            if not isinstance(dim, int):
                raise ValueError("expected 'shape' sequence to contain only integers")
        shape = torch.Size(shape)
    else:
        raise ValueError("expected 'shape' to be an integer or sequence of integers")

    # Check for non-negative dimensions
    if any(dim < 0 for dim in shape):
        raise ValueError("'shape' cannot have negative values")

    # Calculate coefficients for unraveling
    coefs = list(
        reversed(
            list(
                itertools.accumulate(
                    reversed(shape[1:] + torch.Size([1])), func=operator.mul
                )
            )
        )
    )

    # Return from original
    # indices.unsqueeze(-1).floor_divide(
    #     torch.tensor(coefs, device=indices.device, dtype=torch.int64)
    # ) % torch.tensor(shape, device=indices.device, dtype=torch.int64)

    indices = indices.unsqueeze(-1)
    coefs_tensor = torch.tensor(coefs, device=indices.device, dtype=torch.int64)
    shape_tensor = torch.tensor(shape, device=indices.device, dtype=torch.int64)

    unravelled_indices = (
        torch.div(indices, coefs_tensor, rounding_mode="floor") % shape_tensor
    )

    return tuple(unravelled_indices.unbind(-1))
