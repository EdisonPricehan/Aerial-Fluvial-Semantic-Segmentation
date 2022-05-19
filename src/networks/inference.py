#!E:\anaconda/python

import argparse
import os
from torch.utils.data import DataLoader
from dataset import FluvialDataset
from pprint import pprint
from src.utils.custom_transforms import resize
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from SS_Model_Lit import SSModelGeneric as ssmg, check_encoder_existence, check_decoder_existence


def inference(model, test_dataloader, logger):
    trainer = pl.Trainer(gpus=1, max_epochs=1, logger=logger)
    # start testing
    trainer.test(model, test_dataloader)


if __name__ == '__main__':
    # parse necessary arguments if run as script
    parser = argparse.ArgumentParser(description="Pytorch Lightning Testing")
    parser.add_argument('dataset_dir', metavar='DIR', help='Relative path to dataset')
    parser.add_argument('model_path', type=str, default='', help='Relative path of model to be loaded')
    parser.add_argument('-tb', '--test-batch-size', type=int, default=1, metavar='N',
                        help='Test batch size override (default: 1)')
    parser.add_argument('-e', '--encoder', type=str, default='resnet34', help='Encoder type')
    parser.add_argument('-d', '--decoder', type=str, default='Unet', help='Decoder type')
    parser.add_argument('-c', '--out-class-num', type=int, default=1, metavar='N', help='Output class number')

    # init args
    args = parser.parse_args()
    pprint(args.__dict__)

    # parse arguments as local variables
    dataset_dir = args.dataset_dir
    model_path = args.model_path
    test_bs = args.test_batch_size
    encoder = args.encoder
    decoder = args.decoder
    out_class_num = args.out_class_num

    # check encoder existence
    if not check_encoder_existence(encoder):
        exit(0)

    # check decoder existence
    if not check_decoder_existence(decoder):
        exit(0)

    # init dataset and dataloader
    dataset_dir = os.path.join(os.path.dirname(__file__), dataset_dir)
    test_data = FluvialDataset(dataset_dir, train=False, transform=resize, target_transform=resize)
    print(f"Test num: {len(test_data)}")
    print(f"Image size: {test_data[0][0].shape}, mask size: {test_data[0][1].shape}")
    test_dataloader = DataLoader(test_data, batch_size=test_bs, shuffle=False)

    # init logger, log model checkpoints at the end of training
    wandb_logger = WandbLogger(project=os.path.basename(dataset_dir), name='-'.join([decoder, encoder]),
                               log_model=False, anonymous=False,
                               save_dir=os.path.join(os.path.dirname(__file__), '../logs'))
    wandb_logger.experiment.config.update(args)

    checkpoint_path = os.path.join(os.path.dirname(__file__), model_path)
    print(f"{checkpoint_path=}")
    # construct desired model if you are sure your checkpoint has no hyper-parameters
    # model = ssmg(arch=decoder, encoder_name=encoder, in_channels=3, out_classes=out_class_num)
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint["state_dict"])

    # else load directly from checkpoint path
    model = ssmg.load_from_checkpoint(checkpoint_path=checkpoint_path)

    print(f"{model.hparams=}")

    # test
    print("Start inferencing ...")
    inference(model, test_dataloader, wandb_logger)
    print("Inference finished!")
