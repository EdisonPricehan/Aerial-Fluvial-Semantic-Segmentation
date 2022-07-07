#!E:\anaconda/python

import argparse
import os
from pprint import pprint

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataset import FluvialDataset
from src.utils.custom_transforms import resize
from SS_Model_Lit import SSModelGeneric as SSMG, check_encoder_existence, check_decoder_existence


if __name__ == '__main__':
    # define all accepted arguments
    parser = argparse.ArgumentParser(description="Pytorch Lightning Training")
    parser.add_argument('train_file_path', type=str, help='Relative path to train set')
    parser.add_argument('valid_file_path', type=str, help='Relative path to validation set')
    parser.add_argument('-p', '--project', type=str, help='Project name')
    parser.add_argument('-a', '--augment', type=int, default=0, metavar='N',
                        help='Whether to load augmented training data')
    parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                        help='Input batch size for training (default: 16)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=1, metavar='N',
                        help='Validation batch size override (default: 1)')
    parser.add_argument('-e', '--encoder', type=str, default='resnet34', help='Encoder type')
    parser.add_argument('-d', '--decoder', type=str, default='Unet', help='Decoder type')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='Epochs number')
    parser.add_argument('-c', '--out-class-num', type=int, default=1, metavar='N', help='Output class number')

    # parse all arguments
    args = parser.parse_args()
    pprint(args.__dict__)

    # parse arguments as local variables
    train_file_path = args.train_file_path
    valid_file_path = args.valid_file_path
    project_name = args.project
    do_aug = args.augment
    training_bs = args.batch_size
    val_bs = args.validation_batch_size
    encoder = args.encoder
    decoder = args.decoder
    epochs = args.epochs
    out_class_num = args.out_class_num

    # check encoder existence
    if not check_encoder_existence(encoder):
        exit(0)

    # check decoder existence
    if not check_decoder_existence(decoder):
        exit(0)

    # init dataset and dataloader
    training_data = FluvialDataset(train_file_path, use_augment=do_aug, transform=resize, target_transform=resize)
    valid_data = FluvialDataset(valid_file_path, use_augment=False, transform=resize, target_transform=resize)
    print(f"Train num: {len(training_data)}")
    print(f"Image size: {training_data[0][0].shape}, mask size: {training_data[0][1].shape}")
    # print(f"Valid size: {len(valid_dataset)}")
    print(f"Valid num: {len(valid_data)}")
    train_dataloader = DataLoader(training_data, batch_size=training_bs, shuffle=True)
    val_dataloader = DataLoader(valid_data, batch_size=val_bs, shuffle=False)

    # init wandb logger
    wandb_logger = WandbLogger(project=project_name, name='-'.join([decoder, encoder, 'train']),
                               log_model="all", anonymous=False,
                               save_dir=os.path.join(os.path.dirname(__file__), '../logs'))
    wandb_logger.experiment.config.update({'epochs': epochs, 'train_bs': training_bs, 'val_bs': val_bs})

    # init checkpoint callback
    checkpoint_callback = ModelCheckpoint(monitor='valid_dataset_f1', mode='max', save_top_k=1)

    # init early stopping callback, stop training if no improvement for 10 epochs
    early_stop_callback = EarlyStopping(monitor="valid_dataset_f1", min_delta=0.00,
                                        patience=10, verbose=False, mode="max")

    # construct desired model
    model = SSMG(arch=decoder, encoder_name=encoder, in_channels=3, out_classes=out_class_num)
    trainer = pl.Trainer(gpus=1, max_epochs=epochs, logger=wandb_logger, precision=16,
                         callbacks=[checkpoint_callback, early_stop_callback])

    # start training
    print("Start training ...")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    print("Training finished!")
