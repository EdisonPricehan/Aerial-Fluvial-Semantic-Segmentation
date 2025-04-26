#!E:\anaconda/python

import torch
from torchvision.transforms import Resize

import os
from typing import Callable, Literal, Dict, Tuple, List, get_args


# Define types
RESIZE_TRANSFORM = Callable[[torch.Tensor], torch.Tensor]
RESOLUTIONS = Literal['low', 'medium', 'high']

# Define static mapping
resolution_mapping: Dict[str, Tuple[int, int]] = {
    'low': (128, 128),
    'medium': (320, 544),
    'high': (1080, 1920),
}


def supported_resolution_strings() -> Tuple[str]:
    return get_args(RESOLUTIONS)


def supported_resolutions() -> List[Tuple[int, int]]:
    return list(resolution_mapping.values())


def get_transform_by_resolution(h: int, w: int) -> RESIZE_TRANSFORM:
    return Resize((h, w))


def get_transform_by_resolution_level(res_level: RESOLUTIONS = 'medium') -> RESIZE_TRANSFORM:
    assert res_level in resolution_mapping.keys(), f'{res_level} is not supported. Acceptable resolution levels: {supported_resolution_strings()}.'

    h, w = resolution_mapping[res_level]
    return get_transform_by_resolution(h, w)


if __name__ == '__main__':
    from networks.dataset import FluvialDataset
    import visualizer

    dataset_dir = os.path.join(os.path.dirname(__file__), '../dataset/afid/dataset.csv')

    # Change resolution based on the project
    resize_transform = get_transform_by_resolution_level('medium')
    # resize_transform = get_transform_by_resolution_level('low')
    # resize_transform = get_transform_by_resolution_level('high')

    training_dataset = FluvialDataset(dataset_dir, transform=resize_transform, target_transform=resize_transform)
    print(len(training_dataset))

    img, mask = training_dataset[0]  # plot the first training data pair

    # plot the images
    visualizer.plot_img_and_mask(img.permute(1, 2, 0), mask.squeeze())
