import torch


def spikenet_transform(x):
    """
    This function created spikenet data by concatenating AVG and L-bipolar data formats

    Parameters:
    -----------
        x (torch.Tensor): The input data to be transformed.

    Returns:
    --------
        torch.Tensor: The transformed data in the Spikenet format.

    Notes:
    ------
        The input data should be a torch.Tensor of shape (C, T), where C represents the number of channels and T represents the number of time steps.
        The Spikenet data format is created by concatenating the AVG (average) and L-bipolar data formats.

    Examples:
    ---------
        input_data = torch.randn(19, 1000)
        transformed_data = spikenet_transform(input_data)

    """

    # indices: list of pairs where list[0]-list[1] is l2bipolar
    bp_indices = [
        [0, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [11, 15],
        [15, 16],
        [16, 17],
        [17, 18],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 7],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 18],
        [8, 9],
        [9, 10],
    ]

    x = x[:-1]  # take out last row (ekg)
    avg = x - x.mean(axis=0)

    bipolar = torch.clone(x)
    for pair in range(len(bp_indices)):
        i, j = bp_indices[pair]
        bipolar[pair] = x[i] - x[j]

    bipolar = bipolar[:-1]

    return torch.cat((avg, bipolar), axis=0).transpose(0, 1).unsqueeze(0)
