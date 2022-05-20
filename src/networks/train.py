#!E:\anaconda/python

import argparse
import os
from torch.utils.data import DataLoader
from dataset import FluvialDataset
from pprint import pprint
from src.utils.custom_transforms import resize
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from SS_Model_Lit import SSModelGeneric as SSMG, check_encoder_existence, check_decoder_existence


if __name__ == '__main__':
    # define all accepted arguments
    parser = argparse.ArgumentParser(description="Pytorch Lightning Training")
    parser.add_argument('dataset_dir', metavar='DIR', help='Relative path to dataset')
    parser.add_argument('-a', '--augment', type=int, default=0, metavar='N',
                        help='Whether to load augmented training data')
    parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                        help='Input batch size for training (default: 12)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=1, metavar='N',
                        help='Validation batch size override (default: 1)')
    parser.add_argument('-e', '--encoder', type=str, default='resnet34', help='Encoder type')
    parser.add_argument('-d', '--decoder', type=str, default='Unet', help='Decoder type')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='Epochs number')
    parser.add_argument('-c', '--out-class-num', type=int, default=1, metavar='N', help='Output class number')

    # parse all arguments
    args = parser.parse_args()
    pprint(args.__dict__)

    # parse arguments as local variables
    dataset_dir = args.dataset_dir
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
    dataset_dir = os.path.join(os.path.dirname(__file__), dataset_dir)
    training_data = FluvialDataset(dataset_dir, train=True, use_augment=do_aug, transform=resize,
                                   target_transform=resize)
    test_data = FluvialDataset(dataset_dir, train=False, use_augment=False, transform=resize, target_transform=resize)
    print(f"Train num: {len(training_data)}")
    print(f"Image size: {training_data[0][0].shape}, mask size: {training_data[0][1].shape}")
    # print(f"Valid size: {len(valid_dataset)}")
    print(f"Test num: {len(test_data)}")
    train_dataloader = DataLoader(training_data, batch_size=training_bs, shuffle=True)
    # temporarily use test set as validation set
    val_dataloader = DataLoader(test_data, batch_size=val_bs, shuffle=False)

    # init wandb logger
    wandb_logger = WandbLogger(project=os.path.basename(dataset_dir), name='-'.join([decoder, encoder, 'train']),
                               log_model=False, anonymous=False,
                               save_dir=os.path.join(os.path.dirname(__file__), '../logs'))
    wandb_logger.experiment.config.update({'epochs': epochs, 'train_bs': training_bs, 'val_bs': val_bs})

    # construct desired model
    model = SSMG(arch=decoder, encoder_name=encoder, in_channels=3, out_classes=out_class_num)
    trainer = pl.Trainer(gpus=1, max_epochs=epochs, logger=wandb_logger)

    # start training
    print("Start training ...")
    trainer.fit(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    print("Training finished!")
