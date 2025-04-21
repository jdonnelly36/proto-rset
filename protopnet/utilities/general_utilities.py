import os
import subprocess
import sys

import numpy as np
import yaml


def makedir(path):
    """
    Create a directory at the specified path if it does not already exist.

    Parameters:
    -----------
        path (str): The path to the directory to create.

    Returns:
    --------
        None.

    Raises:
    -------
        OSError: If the directory could not be created.

    Called in the following files:
        - find_nearest.py: find_k_nearest_patches_to_prototypes()
        - global_analysis.py: save_def_prototype_patches()
        - local_analysis.py
        - main.py
        - push.py: push_prototypes()
    """
    if not os.path.exists(path):
        os.makedirs(path)


def check_pip_environment(requirements_file="env/requirements-frozen.txt"):
    """
    Checks to see if the current environment matches the requirements (as determined by pip).

    Parameters:
    -----------
        requirements_file (str): The path to the requirements file to check against.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    )

    installed = set((line for line in result.stdout.split("\n") if line.strip() != ""))

    with open(requirements_file) as f:
        requirements_txt = f.read()

    expected = set(
        (line for line in requirements_txt.split("\n") if line.strip() != "")
    )

    differences = installed.symmetric_difference(expected)

    return sorted(expected), sorted(installed), sorted(differences)


def parse_yaml_file(yaml_file, args):
    """
    Adds arguments from a YAML file to an argument class.
    Arguments:
        yaml_file (str): Path to the YAML file containing arguments.
        args (argparse.Namespace): An argument class.
    """
    if yaml_file:
        print("Using YAML file to parse arguments, ignoring other arguments")
        with open(yaml_file, "r") as f:
            yaml_args = yaml.safe_load(f)

        # Update args with YAML arguments if present
        for key, value in yaml_args.items():
            setattr(args, key, value)

    return args


def find_high_activation_crop(activation_map, percentile=95):
    """
    Given an activation map, find the rectangular crop that contains the top `percentile` percent of activations.

    Parameters:
    -----------
        activation_map (np.ndarray): A 2D array of activation values.
        percentile (float): The percentile of activations to include in the crop. Defaults to 95.

    Returns:
    --------
        A tuple of integers (lower_y, upper_y, lower_x, upper_x), representing the coordinates of the rectangular crop
        that contains the top `percentile` percent of activations. `lower_y` and `upper_y` are the indices of the top and
        bottom rows of the crop, respectively, and `lower_x` and `upper_x` are the indices of the left and right columns
        of the crop, respectively.

    Raises:
    -------
        ValueError: If `percentile` is not between 0 and 100, or if `activation_map` is not a 2D array.

    Called in the following files:
        - find_nearest.py (not used)
        - local_analysis.py
        - push.py: update_prototypes_on_batch(), save_projected_prototype_images()
    """
    # if not isinstance(activation_map, np.ndarray) or activation_map.ndim != 2:
    #     raise ValueError("`activation_map` must be a 2D numpy array")
    # if not 0 <= percentile <= 100:
    #     raise ValueError("`percentile` must be between 0 and 100")

    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1
