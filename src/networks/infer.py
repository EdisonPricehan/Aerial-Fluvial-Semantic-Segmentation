
import os.path as osp
import torch
import torchvision
import torchvision.transforms as TF
import pytorch_lightning as pl
from PIL import Image
import segmentation_models_pytorch as smp
from SS_Model_Lit import SSModelGeneric as ssmg


arch = 'Unet'  # PAN
encoder = 'mit_b1'

input_image_path = '/home/edison/Research/Aerial-Fluvial-Semantic-Segmentation/afid/WildcatCreekDataset/wildcat_dataset/images/wildcat_forward_0118.jpg'
output_mask_path = '/home/edison/Research/Aerial-Fluvial-Semantic-Segmentation/src/images/predictions/{}-{}.png'.format(arch, encoder)

input_checkpoint_path = '/home/edison/Research/Aerial-Fluvial-Semantic-Segmentation/src/models/{}-{}.ckpt'.format(arch, encoder)
output_pth_path = '/home/edison/Research/Aerial-Fluvial-Semantic-Segmentation/src/models/{}-{}.pth'.format(arch, encoder)
input_pth_path = output_pth_path

height, width = 320, 544

device = torch.device('cuda')


def print_config():
    print(f'{input_image_path=}')
    print(f'{output_mask_path}')
    print(f'{arch=}')
    print(f'{encoder=}')
    print(f'{input_checkpoint_path=}')
    print(f'{input_pth_path=}')
    print(f'{height=} {width=}')


if __name__ == '__main__':
    print_config()

    img = Image.open(input_image_path)

    transform = TF.Compose([
        TF.Resize(size=(height, width)),
        TF.PILToTensor(),
        TF.ConvertImageDtype(torch.float),
    ])

    img = transform(img).to('cuda')
    print('Loaded image size: {}'.format(img.shape))

    '''
    Load checkpoint
    '''
    # checkpoint = torch.load(input_checkpoint_path)
    # checkpoint = pl.LightningModule.load_from_checkpoint(input_checkpoint_path)
    # checkpoint = ssmg.load_from_checkpoint(input_checkpoint_path)
    # model = ssmg.load_from_checkpoint(input_checkpoint_path)
    # print('Checkpoint loaded!')

    '''
    Load .pth model
    '''
    model = ssmg(arch=arch, encoder_name=encoder, in_channels=3, out_classes=1)
    # model.load_state_dict(torch.load(input_pth_path, map_location='cuda:0'))
    model.load_state_dict(torch.load(input_pth_path))
    model.to(device)
    print('pth model loaded!')

    # print('Hyper parameters: {}'.format(checkpoint['hyper_parameters']))
    # print('Checkpoint keys: {}'.format(checkpoint.keys()))

    model.eval()
    logits_mask = model(img)
    prob_mask = logits_mask.sigmoid()
    pred_mask = ((prob_mask > 0.5).float() * 255).to(torch.uint8).squeeze(0).to('cpu')
    print('Pred mask shape: {}'.format(pred_mask.shape))
    print('Inference finished!')
    torchvision.io.write_png(pred_mask, output_mask_path)
    print('Inferred mask saved!')

    # torch.save(model.state_dict(), output_pth_path)
    # print('Pth model saved!')






