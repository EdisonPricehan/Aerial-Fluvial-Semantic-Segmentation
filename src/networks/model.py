import torch
import wandb
from typing import List, Tuple, Dict, Any
import lightning as L
import segmentation_models_pytorch as smp


# Define frequently used types
BATCH_OUTPUTS = Dict[str, torch.LongTensor]
EPOCH_OUTPUTS = List[BATCH_OUTPUTS]


# Generic model for semantic segmentation
class LitSegModel(L.LightningModule):
    def __init__(
            self,
            arch: str,
            encoder_name: str,
            in_channels: int,
            out_classes: int,
            lr: float = 1e-4,
            **kwargs: Dict[str, Any],
    ):
        super().__init__()

        # Define member variables
        self.lr: float = lr
        self.train_outputs: EPOCH_OUTPUTS = []
        self.valid_outputs: EPOCH_OUTPUTS = []
        self.test_outputs: EPOCH_OUTPUTS = []

        # Create model by architecture type and encoder type
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes, **kwargs
        )

        # Preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # For image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(
            mode='binary' if out_classes == 1 else 'multiclass',
            from_logits=True,
        )

        # Save module arguments to model hyperparameters
        self.save_hyperparameters()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass image to segmentation model, resulting in mask in logits format.

        Args:
            image:

        Returns:

        """
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> Dict[str, torch.LongTensor]:
        """
        Forward pass images, compare with masks, then return loss, true positive, false positive, true negative and
        false negative, etc., as metrics.

        Args:
            batch:
            stage:

        Returns:

        """
        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have the
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        # add one channel dimension if mask does not have it
        if mask.ndim == 3:
            mask = torch.unsqueeze(mask, 1)
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Compute metrics for some threshold
        pred_mask = self.logits_to_mask(logits_mask)

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(output=pred_mask.long(), target=mask.long(), mode="binary")

        if stage == 'test':
            return {'loss': loss, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'image': image, 'gt': mask, 'pred': pred_mask}
        else:
            return {'loss': loss, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def shared_epoch_end(self, outputs: List[Dict[str, torch.LongTensor]], stage: str) -> None:
        # Aggregate step metrics
        tp = torch.cat([x['tp'] for x in outputs])
        fp = torch.cat([x['fp'] for x in outputs])
        fn = torch.cat([x['fn'] for x in outputs])
        tn = torch.cat([x['tn'] for x in outputs])

        # Per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # Dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        accu = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

        # Log metrics
        metrics: Dict[str, float] = {
            f'{stage}_per_image_iou': per_image_iou.float().item(),
            f'{stage}_dataset_iou': dataset_iou.float().item(),
            f'{stage}_dataset_f1': f1_score.float().item(),
            f'{stage}_dataset_accuracy': accu.float().item(),
        }

        if self.logger:
            self.logger.log_metrics(metrics)

            if stage == "valid":
                self.log("valid_dataset_f1", f1_score, on_epoch=True)  # TODO is this redundant?

            # log images and masks as table for testing stage
            if stage == "test":
                columns = ['id', 'image', 'gt_mask', 'pred_mask', 'iou', 'f1', 'pa']
                data = [[idx,
                         wandb.Image(x['image'].cpu()),
                         wandb.Image(x['gt'].cpu()),
                         wandb.Image(x['pred'].cpu()),
                         smp.metrics.iou_score(x['tp'], x['fp'], x['fn'], x['tn'], reduction='micro-imagewise').item(),
                         smp.metrics.f1_score(x['tp'], x['fp'], x['fn'], x['tn'], reduction='micro').item(),
                         smp.metrics.accuracy(x['tp'], x['fp'], x['fn'], x['tn'], reduction='micro').item()]
                        for idx, x in enumerate(outputs)]
                self.logger.log_table(key='test results', columns=columns, data=data)

    def training_step(self, batch, batch_idx) -> BATCH_OUTPUTS:
        outputs: BATCH_OUTPUTS = self.shared_step(batch, 'train')
        self.train_outputs.append(outputs)
        return outputs

    def on_train_epoch_end(self) -> None:
        self.shared_epoch_end(self.train_outputs, 'train')
        self.train_outputs.clear()

    def validation_step(self, batch, batch_idx) -> BATCH_OUTPUTS:
        outputs: BATCH_OUTPUTS = self.shared_step(batch, 'valid')
        self.valid_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self) -> None:
        self.shared_epoch_end(self.valid_outputs, 'valid')
        self.valid_outputs.clear()

    def test_step(self, batch, batch_idx) -> BATCH_OUTPUTS:
        outputs: BATCH_OUTPUTS = self.shared_step(batch, 'test')
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self) -> None:
        self.shared_epoch_end(self.test_outputs, 'test')
        self.test_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        logits_mask = self(batch)
        return self.logits_to_mask(logits_mask, uint8=True)  # list([N x C x H x W])

    def logits_to_mask(self, logits_mask: torch.Tensor, uint8: bool = False) -> torch.Tensor:
        """
        First convert mask values to probabilities, then apply thresholding.

        Args:
            logits_mask:
            uint8:

        Returns:

        """
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        return (pred_mask * 255).to(torch.uint8) if uint8 else pred_mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def check_encoder_existence(encoder_name: str) -> bool:
    all_encoders = smp.encoders.get_encoder_names()
    # print(f"{all_encoders=}")
    if encoder_name not in all_encoders:
        print(f'{encoder_name} is not supported, available encoders: {all_encoders}!')
        return False
    return True


def check_decoder_existence(decoder_name: str) -> bool:
    all_decoders = ['DeepLabV3', 'DeepLabV3Plus', 'FPN', 'Linknet', 'MAnet', 'PAN', 'PSPNet', 'Unet', 'UnetPlusPlus']
    # print(f"{all_decoders=}")
    if decoder_name not in all_decoders:
        print(f'{decoder_name} is not supported, available decoders: {all_decoders}!')
        return False
    return True

