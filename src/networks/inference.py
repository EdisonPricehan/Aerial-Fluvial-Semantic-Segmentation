#!E:\anaconda/python

import argparse
import os
from torch.utils.data import DataLoader
from src.networks.dataset import FluvialDataset
from pprint import pprint
from src.utils.custom_transforms import resize
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from SS_Model_Lit import SSModelGeneric as ssmg, check_encoder_existence, check_decoder_existence


def inference(model, test_dataloader, logger=None):
    trainer = pl.Trainer(gpus=1, max_epochs=1, logger=(logger if logger is not None else False))
    # start testing
    trainer.test(model, test_dataloader, verbose=True)
    # return trainer.predict(model, dataloaders=test_dataloader)


def get_model(model_path):
    checkpoint_path = os.path.join(os.path.dirname(__file__), model_path)
    print(f"{checkpoint_path=}")
    m = ssmg.load_from_checkpoint(checkpoint_path=checkpoint_path)
    return m


def get_dataloader(ds_path):
    test_data = FluvialDataset(ds_path, use_augment=False, transform=resize, target_transform=resize)
    print(f"Test num: {len(test_data)}")
    print(f"Image size: {test_data[0][0].shape}, mask size: {test_data[0][1].shape}")
    test_dataloader = DataLoader(test_data, batch_size=test_bs, shuffle=False)
    return test_dataloader


def get_logger(encoder_name='', decoder_name='', project_name='', test_batch_size=None,
               trainset='pretraining', testset='wabash_wildcat'):
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
    # parse necessary arguments if run as script
    parser = argparse.ArgumentParser(description="Pytorch Lightning Testing")
    parser.add_argument('dataset_path', type=str, help='Relative path to dataset')
    parser.add_argument('model_path', type=str, default='', help='Relative path of model to be loaded')
    parser.add_argument('-p', '--project-name', type=str, default='', help='Project name')
    parser.add_argument('-tb', '--test-batch-size', type=int, default=1, metavar='N',
                        help='Test batch size override (default: 1)')
    parser.add_argument('-e', '--encoder', type=str, default='resnet34', help='Encoder type')
    parser.add_argument('-d', '--decoder', type=str, default='Unet', help='Decoder type')
    parser.add_argument('-c', '--out-class-num', type=int, default=1, metavar='N', help='Output class number')

    # init args
    args = parser.parse_args()
    pprint(args.__dict__)

    # parse arguments as local variables
    dataset_path = args.dataset_path
    model_path = args.model_path
    test_bs = args.test_batch_size
    encoder = args.encoder
    decoder = args.decoder
    out_class_num = args.out_class_num
    project_name = args.project_name

    # check encoder existence
    if not check_encoder_existence(encoder):
        exit(0)

    # check decoder existence
    if not check_decoder_existence(decoder):
        exit(0)

    # init dataloader
    test_dataloader = get_dataloader(dataset_path)

    # init logger
    logger = get_logger(encoder_name=encoder, decoder_name=decoder, project_name=project_name, test_batch_size=test_bs,
                        trainset='pretraining_wabash_wildcat', testset='wabash_wildcat')

    # construct desired model if you are sure your checkpoint has no hyper-parameters
    # model = ssmg(arch=decoder, encoder_name=encoder, in_channels=3, out_classes=out_class_num)
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint["state_dict"])

    # otherwise load model directly from checkpoint path, model hyper-params will be loaded automatically
    model = get_model(model_path)
    print(f"{model.hparams=}")

    # test
    print("Start inferencing ...")
    inference(model, test_dataloader, logger)
    print("Inference finished!")
