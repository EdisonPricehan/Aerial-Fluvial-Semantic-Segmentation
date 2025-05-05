from typing import Union
import numpy as np
from patchify import patchify, unpatchify

import torch


def patch_array(
    mask_array: np.ndarray,
    is_uint8: bool = False,
    patch_size_x: int = 16,
    patch_size_y: int = 16,
    patch_step: int = 16,
    binary_threshold: float = 0.5,
    patch_threshold: float = 0.5,
) -> np.ndarray:
    """
    Converts a mask into patchified representation based on two thresholds:
    1. binary_threshold: Converts the input mask to binary values.
    2. patch_threshold: Determines the proportion of '1' values within each patch for it to be considered 'water'.

    Args:
        mask_array (np.ndarray): Input 2D mask array with values in [0, 1] (float) or [0, 255] (uint8).
        is_uint8 (bool): Indicates if the input array is in uint8 format (values in [0, 255]).
        patch_size_x (int): Pixel number in x axis of a patch.
        patch_size_y (int): Pixel number in y axis of a patch.
        patch_step (int): The step size for extracting patches.
        binary_threshold (float): The value threshold to convert the mask into binary values.
        patch_threshold (float): The percentage threshold to determine if a patch is water.

    Returns:
        np.ndarray: A flattened binary array representing the coarsened mask patches.
    """
    assert len(mask_array.shape) == 2, f'Only support 2D mask array.'

    # Convert mask to binary using binary_threshold
    if is_uint8:
        mask_binary = (mask_array > (binary_threshold * 255)).astype(np.uint8)
    else:
        mask_binary = (mask_array > binary_threshold).astype(np.uint8)

    # Patchify the binary mask to shape (num_patches_x, num_patches_y, patch_size_x, patch_size_y)
    patchified_mask = patchify(mask_binary, (patch_size_x, patch_size_y), step=patch_step)

    # Calculate the percentage of '1's (water pixels) in each patch
    percentage_water = np.mean(patchified_mask == 1, axis=(2, 3))

    # Create a 2D binary array where each patch is considered water (1) if percentage exceeds patch_threshold
    condensed_mask = np.where(percentage_water > patch_threshold, 1, 0).flatten()

    return condensed_mask


def get_patchified_mask(
    mask: Union[np.ndarray, torch.Tensor],
    is_uint8: bool = False,
    patch_size_x: int = 16,
    patch_size_y: int = 16,
    patch_step: int = 16,
    binary_threshold: float = 0.5,
    patch_threshold: float = 0.5,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Patchify the mask
    Args:
        mask: given mask for patchification.
        is_uint8: Indicates if the input array is in uint8 format (values in [0, 255]).
        patch_size_x: Pixel number in x axis of a patch.
        patch_size_y: Pixel number in y axis of a patch.
        patch_step: The step size for extracting patches.
        binary_threshold: The value threshold to convert the mask into binary values.
        patch_threshold: The percentage threshold to determine if a patch is water.

    Returns: A flattened binary array or tensor representing the coarsened mask patches.

    """
    # Arg validation
    assert 0 <= binary_threshold < 1, f'Binary threshold should be in [0, 1), given {binary_threshold}'
    assert 0 <= patch_threshold < 1, f'Patch threshold should be in [0, 1), given {patch_threshold}'
    assert patch_size_x == patch_size_y == patch_step, (f'Only support square image patchification without overlapping, '
                                                        f'{patch_size_x=} {patch_size_y=} {patch_step=}.')

    if isinstance(mask, np.ndarray):
        # Channel dim should be last
        assert mask.shape[-1] <= 4, f'Channel dim for numpy array should be the last dim.'

        # Get the 1-channel 2D water mask
        mask_arr = mask[..., 0]

        return patch_array(mask_arr, is_uint8, patch_size_x, patch_size_y, patch_step,
                           binary_threshold, patch_threshold)
    elif isinstance(mask, torch.Tensor):
        if len(mask.shape) == 2:  # Row x Column
            patch_tensor = torch.Tensor(
                patch_array(mask.numpy(), is_uint8, patch_size_x, patch_size_y, patch_step,
                            binary_threshold, patch_threshold))
            return patch_tensor
        elif len(mask.shape) == 3:  # Channel x Row x Column
            # Channel dim should be first
            assert mask.shape[0] <= 4, f'Channel dim for torch Tensor should be the first dim.'

            patch_tensor = torch.Tensor(patch_array(mask[0].numpy(), is_uint8, patch_size_x, patch_size_y, patch_step,
                                                    binary_threshold, patch_threshold))
            return patch_tensor
        else:  # Batch x Channel x Row x Column
            # Channel dim should be first
            assert mask.shape[1] <= 4, f'Channel dim for batched torch Tensor should be the second dim.'

            patch_list = []
            for mask3d in mask:
                patch_list.append(torch.Tensor(patch_array(mask3d[0].numpy(), is_uint8, patch_size_x, patch_size_y,
                                                           patch_step, binary_threshold, patch_threshold)))
            return torch.stack(patch_list)
    else:
        raise NotImplementedError


def inflate_patch_mask(
    obs: np.ndarray,
    image_size: int = 128,
    patch_dim_x: int = 8,
    patch_dim_y: int = 8,
    patch_size_x: int = 16,
    patch_size_y: int = 16,
) -> np.ndarray:
    """
    Inflate the coarsened water mask to the original size for parallel display
    Args:
        obs: coarsened water mask for RL training
        image_size: pixels number in image axis, assuming square image
        patch_dim_x: x dim of patchified image
        patch_dim_y: y dim of patchified image
        patch_size_x: pixel number in x axis of a patch
        patch_size_y: pixel number in y axis of a patch

    Returns:

    """
    if len(obs.shape) == 1:
        obs_2d = obs.reshape((patch_dim_x, patch_dim_y))
    elif len(obs.shape) == 2:
        assert obs.shape[0] == patch_dim_x and obs.shape[1] == patch_dim_y, f'obs dim not match, {obs.shape}'
        obs_2d = obs.copy()
    else:
        raise NotImplementedError

    inflated_mask = np.zeros((patch_dim_x, patch_dim_y, patch_size_x, patch_size_y))
    for row in range(patch_dim_x):
        for col in range(patch_dim_y):
            inflated_mask[row, col] = np.full((patch_size_x, patch_size_y), obs_2d[row, col])

    inflated_mask_2d = unpatchify(inflated_mask, (image_size, image_size))

    return inflated_mask_2d


