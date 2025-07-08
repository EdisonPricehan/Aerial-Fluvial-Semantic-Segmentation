#!E:\anaconda/python

import argparse
import os
from pprint import pprint
from typing import Optional, List

from networks.dataset import FluvialDataset
from networks.model import LitSegModel as LSM, check_encoder_existence, check_decoder_existence
from utils.image_transforms import get_transform_by_resolution_level, supported_resolution_strings, RESOLUTIONS

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger


# Change matmul precision based on preference-overhead tradeoff
torch.set_float32_matmul_precision('medium')  # {'highest', 'high', 'medium'}


def inference(model: LSM, test_dataloader: DataLoader, logger: Optional[WandbLogger] = None):
    """
    Perform inference on the test dataset using the provided model.
    The dataset of the test_dataloader expects images and masks as input and target respectively.
    Args:
        model: The lightning model to be used for inference.
        test_dataloader: The DataLoader for the test dataset.
        logger: Optional; a logger to log the results. If None, no logging will be performed.

    Returns:

    """
    trainer = L.Trainer(max_epochs=1, logger=(logger if logger is not None else False))

    trainer.test(model, test_dataloader, verbose=True)


def predict(model: LSM, test_dataloader: DataLoader, logger: Optional[WandbLogger] = None) -> List[torch.Tensor]:
    """
    Perform prediction on the test dataset using the provided model.
    The dataset of the test_dataloader expects ONLY images as input.
    Args:
        model: The lightning model to be used for prediction.
        test_dataloader: The DataLoader for the test dataset.
        logger: Optional; a logger to log the results. If None, no logging will be performed.

    Returns:
        A list of tensors (predictions) from the model.
    """
    trainer = L.Trainer(max_epochs=1, logger=(logger if logger is not None else False))
    return trainer.predict(model, dataloaders=test_dataloader)


def get_model(model_path: str) -> LSM:
    """
    Load a model from the specified checkpoint path.
    Args:
        model_path: Relative path of the model checkpoint to be loaded.

    Returns:
        An instance of the LSM model loaded from the checkpoint.
    """
    checkpoint_path = os.path.join(os.path.dirname(__file__), model_path)
    assert os.path.exists(checkpoint_path), f'{checkpoint_path} does not exist!'
    m = LSM.load_from_checkpoint(checkpoint_path=checkpoint_path)
    return m


def get_dataloader(ds_path: str, resolution_level: RESOLUTIONS = 'low') -> DataLoader:
    """
    Create a DataLoader for the test dataset with the specified resolution level.
    Args:
        ds_path: Relative path of the dataset csv file in src/dataset/
        resolution_level: The resolution level for the images and masks.

    Returns:
        A DataLoader for the test dataset.
    """
    resize_transform = get_transform_by_resolution_level(resolution_level)
    test_data = FluvialDataset(ds_path, use_augment=False, transform=resize_transform, target_transform=resize_transform)
    print(f"Test num: {len(test_data)}")
    print(f"Image size: {test_data[0][0].shape}, mask size: {test_data[0][1].shape}")

    test_dataloader = DataLoader(test_data, batch_size=test_bs, shuffle=False)

    return test_dataloader


def get_logger(
        encoder_name: str = '',
        decoder_name: str = '',
        project_name: str = '',
        test_batch_size: Optional[int] = None,
        trainset: str = 'pretraining',
        testset: str = 'wabash_wildcat',
):
    """
    Initialize a WandbLogger for logging the test results.
    Args:
        encoder_name: The name of the encoder used in the model.
        decoder_name: The name of the decoder used in the model.
        project_name: The name of the project for logging.
        test_batch_size: The batch size used for testing.
        trainset: The name of the training dataset used.
        testset: The name of the test dataset used.

    Returns:
        A WandbLogger instance if all parameters are provided, otherwise None.
    """
    if encoder_name == '' or decoder_name == '' or project_name == '' or test_batch_size is None:
        print("Need arguments to init logger, return None logger.")
        return None

    wandb_logger = WandbLogger(project=project_name,
                               name='-'.join([encoder_name, decoder_name, trainset, testset, 'test']),
                               log_model=False, anonymous=False,
                               save_dir=os.path.join(os.path.dirname(__file__), '../logs'))
    wandb_logger.experiment.config.update({'test_bs': test_batch_size,
                                           'trainset': trainset,
                                           'testset': testset})
    return wandb_logger


if __name__ == '__main__':
    # Parse necessary arguments if run as script
    parser = argparse.ArgumentParser(description="Pytorch Lightning Testing")
    parser.add_argument('dataset_path', type=str, help='Relative path to dataset')
    parser.add_argument('model_path', type=str, default='', help='Relative path of model to be loaded')
    parser.add_argument('-r', '--resolution', type=str, default='low',
                        choices=supported_resolution_strings(),
                        help='Desired resolution level of image and mask')
    parser.add_argument('-p', '--project-name', type=str, default='', help='Project name')
    parser.add_argument('-tb', '--test-batch-size', type=int, default=1, metavar='N',
                        help='Test batch size override (default: 1)')
    parser.add_argument('-e', '--encoder', type=str, default='resnet34', help='Encoder type')
    parser.add_argument('-d', '--decoder', type=str, default='Unet', help='Decoder type')
    parser.add_argument('-c', '--out-class-num', type=int, default=1, metavar='N', help='Output class number')

    # Init args
    args = parser.parse_args()
    pprint(args.__dict__)

    # Parse arguments as local variables
    dataset_path = args.dataset_path
    model_path = args.model_path
    res_level = args.resolution
    test_bs = args.test_batch_size
    encoder = args.encoder
    decoder = args.decoder
    out_class_num = args.out_class_num
    project_name = args.project_name

    # Check encoder existence
    if not check_encoder_existence(encoder_name=encoder):
        exit(0)

    # Check decoder existence
    if not check_decoder_existence(decoder_name=decoder):
        exit(0)

    # Init dataloader
    test_dataloader = get_dataloader(ds_path=dataset_path, resolution_level=res_level)

    # Init logger
    logger = get_logger(encoder_name=encoder, decoder_name=decoder, project_name=project_name, test_batch_size=test_bs,
                        trainset='wabash_wildcat_train', testset='wabash_wildcat_test')

    # Construct desired model if you are sure your checkpoint has no hyper-parameters
    # model = LSM(arch=decoder, encoder_name=encoder, in_channels=3, out_classes=out_class_num)
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint["state_dict"])

    # Otherwise load model directly from checkpoint path, model hyper-params will be loaded automatically
    model = get_model(model_path)
    print(f"{model.hparams=}")

    # test
    print("Start inferencing ...")
    inference(model, test_dataloader, logger)
    print("Inference finished!")
