#!E:\anaconda/python

import os
import torch
from fluvial_cnn import FluvialCNN
from dataset import FluvialDataset
import torchvision.models as models
from src.utils.custom_transforms import resize


if __name__ == '__main__':
    load = True  # Specify whether you want to load model, if False, no need to care about model_name

    model_name = 'model-DeeplabV3_Resnet101-train-04-22-2022-07_20_33.pth'

    model_dir = os.path.join(os.path.dirname(__file__), '../models/WildcatCreek-Data')
    model_to_load = os.path.join(model_dir, model_name)

    if load:
        # Load in previous model weights
        net = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False, num_classes=2)
        net.load_state_dict(torch.load(model_to_load))
    else:
        # build the neural network from scratch
        net = models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False, num_classes=2)

    # define dataset directories we want to test our current model with
    dataset_dir1 = os.path.join(os.path.dirname(__file__), '../dataset/WildcatCreek-Data')
    dataset_dir2 = os.path.join(os.path.dirname(__file__), '../dataset/River-Segmentation-Data')

    # specify the target dataset to get the training and testing dataset objects
    training_data = FluvialDataset(dataset_dir1, train=True, transform=resize, target_transform=resize)
    test_data = FluvialDataset(dataset_dir1, train=False, transform=resize, target_transform=resize)

    # Specify whether you want to test your loaded model, if False, above 'load' must be True
    train = False
    if not train:
        assert load, "Must load model if you want to test the model!"

    if train:
        f = FluvialCNN(net, training_data, test_data, model_dir, epochs=10, learning_rate=1e-5, batch_size=4,
                       save_checkpoint=False, project_name='DeeplabV3_Resnet101-train')
        if not load:
            print("Start training ...")
            f.train(with_validation=False)
            f.save()
        else:
            print("Start testing ...")
            f.test()
    else:
        print("Start testing ...")
        f = FluvialCNN(net, training_data, test_data, model_dir, epochs=5, learning_rate=1e-5, batch_size=1,
                       save_checkpoint=False, project_name='DeeplabV3_Resnet101-test-batch-size-1')
        f.test()

    print("Done!")