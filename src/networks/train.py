#!E:\anaconda/python

import argparse
import os
from pprint import pprint

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from networks.dataset import FluvialDataset
from networks.model import LitSegModel as LSM, check_encoder_existence, check_decoder_existence
from utils.custom_transforms import get_transform_by_resolution_level, supported_resolution_strings

# Change matmul precision based on preference-overhead tradeoff
torch.set_float32_matmul_precision('medium')  # {'highest', 'high', 'medium'}


if __name__ == '__main__':
    # Define all arguments
    parser = argparse.ArgumentParser(description="Pytorch Lightning Training")
    parser.add_argument('train_file_path', type=str, help='Relative path to train set')
    parser.add_argument('valid_file_path', type=str, help='Relative path to validation set')
    parser.add_argument('-r', '--resolution', type=str, default='low',
                        choices=supported_resolution_strings(),
                        help='Desired resolution level of image and mask')
    parser.add_argument('-p', '--project', type=str, help='Project name')
    parser.add_argument('-a', '--augment', type=int, default=0, metavar='N',
                        help='Whether to load augmented training data')
    parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                        help='Input batch size for training (default: 16)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=1, metavar='N',
                        help='Validation batch size override (default: 1)')
    parser.add_argument('-e', '--encoder', type=str, default='resnet34', help='Encoder type')
    parser.add_argument('-d', '--decoder', type=str, default='Unet', help='Decoder type')
    parser.add_argument('--epochs', type=int, default=75, metavar='N', help='Epochs number')
    parser.add_argument('-c', '--out-class-num', type=int, default=1, metavar='N', help='Output class number')

    # Parse all arguments
    args = parser.parse_args()
    pprint(args.__dict__)

    # Parse arguments as local variables
    train_file_path = args.train_file_path
    valid_file_path = args.valid_file_path
    res_level = args.resolution
    project_name = args.project
    do_aug = args.augment
    training_bs = args.batch_size
    val_bs = args.validation_batch_size
    encoder = args.encoder
    decoder = args.decoder
    epochs = args.epochs
    out_class_num = args.out_class_num

    # Check encoder existence
    if not check_encoder_existence(encoder):
        exit(0)

    # Check decoder existence
    if not check_decoder_existence(decoder):
        exit(0)

    # Define resolution level
    res_tf = get_transform_by_resolution_level(res_level=res_level)

    # Init dataset and dataloader
    training_data = FluvialDataset(dataset_path=train_file_path,
                                   use_augment=do_aug,
                                   transform=res_tf,
                                   target_transform=res_tf,
                                   multi_class=False)
    valid_data = FluvialDataset(dataset_path=valid_file_path,
                                use_augment=False,
                                transform=res_tf,
                                target_transform=res_tf,
                                multi_class=False)
    print(f"Image size: {training_data[0][0].shape}, mask size: {training_data[0][1].shape}")
    print(f"Train num: {len(training_data)}")
    print(f"Valid num: {len(valid_data)}")
    train_dataloader = DataLoader(training_data, batch_size=training_bs, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(valid_data, batch_size=val_bs, shuffle=False, num_workers=2)

    # Init wandb logger and update training params
    wandb_logger = WandbLogger(project=project_name, name='-'.join([decoder, encoder, 'train']),
                               log_model="all", anonymous=False,
                               save_dir=os.path.join(os.path.dirname(__file__), '../logs'))
    wandb_logger.experiment.config.update({'epochs': epochs, 'train_bs': training_bs, 'val_bs': val_bs})

    # Init checkpoint callback
    checkpoint_callback = ModelCheckpoint(monitor='valid_dataset_f1', mode='max', save_top_k=1)

    # Init early stopping callback, stop training if no improvement for 10 epochs
    early_stop_callback = EarlyStopping(monitor="valid_dataset_f1", min_delta=0.00,
                                        patience=10, verbose=False, mode="max")

    # Construct desired model
    model = LSM(arch=decoder, encoder_name=encoder, in_channels=3, out_classes=out_class_num, lr=1e-4)

    # Construct model trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        precision=None,
        enable_progress_bar=True,
        deterministic=True,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    print("Start training ...")
    # Start training, uncomment ckpt_path arg if want to load model for continue training
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        # ckpt_path=os.path.join(os.path.dirname(__file__), '../logs/baseline/smquglle/checkpoints/epoch=8-step=2025.ckpt'),
        ckpt_path=None,
    )

    print("Training finished!")
