# TODO: consider moving all these to helpers.py

import torch

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def preprocess(x, mean, std):
    """
    Normalize an input tensor x using mean and standard deviation values.

    Parameters:
    -----------
        x (torch.Tensor): Input tensor to normalize.
        mean (tuple): Tuple of mean values for each channel.
        std (tuple): Tuple of standard deviation values for each channel.

    Returns:
    --------
        y (torch.Tensor): Normalized tensor.

    Raises:
        AssertionError: If the input tensor does not have 3 channels.

    Called in the following files:
        - preprocess.py: preprocess_input_function
    """
    # assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(1):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    """
    Allocate a new tensor like x and apply the normalization used in the
    pretrained model.

    Args:
        x (torch.Tensor): Input tensor to preprocess.

    Returns:
        y (torch.Tensor): Preprocessed tensor.

    Called in the following files:
        - find_nearest.py: find_k_nearest_patches_to_prototypes()
        - push.py: push_prototypes()
        - global_analysis.py (calls push.py: push_prototypes)
        - main.py (calls find_nearest: find_k_nearest_patches_to_prototypes)
    """
    return preprocess(x, mean=mean, std=std)


def undo_preprocess(x, mean, std):
    """
    Invert normalization applied by preprocess function.

    Parameters:
    ----------
        x (torch.Tensor): Normalized tensor to invert.
        mean (tuple): Tuple of mean values for each channel.
        std (tuple): Tuple of standard deviation values for each channel.

    Returns:
    --------
        y (torch.Tensor): Inverted tensor.

    Raises:
    -------
        AssertionError: If the input tensor does not have 3 channels.
    """
    # assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(1):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y


def undo_preprocess_input_function(x):
    """
    Allocate a new tensor like x and undo the normalization used in the
    pretrained model.

    Parameters:
    -----------
        x (torch.Tensor): Input tensor to undo preprocessing.

    Returns:
    --------
        y (torch.Tensor): Unprocessed tensor.


    Called in the following files:
        - local_analysis.py: save_preprocessed_img()
    """
    return undo_preprocess(x, mean=mean, std=std)
