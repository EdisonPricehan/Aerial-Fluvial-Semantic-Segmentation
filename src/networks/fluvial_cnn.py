#!E:\anaconda/python

import torch
from torch import nn
from torch.utils.data import DataLoader
import datetime
import tqdm
import os
import logging
import wandb
from src.utils.custom_metrics import dice_coeff, multiclass_dice_coeff
import torch.nn.functional as F


class FluvialCNN:
    """This class contains the dataset, dataloader, and all other respective initiated values for a fluvial CNN it holds
    everything except the network itself. In this way the same dataloader and dataset cna be utilized with several
    different trial networks that are defined elsewhere. The fluvial CNN class also contains prediction parameters, and
    will load in parameters."""

    def __init__(self, network, train_dataset, test_dataset, models_dir,
                 epochs: int = 5, learning_rate: float = 1e-5, batch_size: int = 4, save_checkpoint: bool = True,
                 project_name=''):
        """This __init__ function is where all of the hyperparameters are defined for the data inputs, and training of the
        Fluvial Convolutional Neural Network. the init function
        input:
            network - A structured pytorch neural netwrok class as per:
                https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
            train_dataset - dataset class defining the training dataset to be utilized
            test_dataset - dataset class defining the testing dataset to be utilized
            models_dir - the directory where saved models are kept
        output:
            None - Void function edits function parameters
        Note:
            A log of all training, results, models used, etc. is available in the Fluvial_CNn under SRC
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        # --- NETWORK--- #
        self.network = network
        self.network.to(self.device)

        # --- HYPERPARAMETERS--- #
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_checkpoint = save_checkpoint
        self.shuffle = True
        self.enable_amp = True  # whether enable automatic mixed precision

        # --- LOSS FUNCTION --- #
        # loss functions can be implimented from pytorch --> https://pytorch.org/docs/stable/nn.html#loss-functions
        # can also impliment custom functions --> https://discuss.pytorch.org/t/custom-loss-functions/29387
        # 2020 article about segmentation --> https://arxiv.org/pdf/2006.14822.pdf
        self.loss_fn = nn.CrossEntropyLoss()  # define the loss function that will be used - [pytorch class]

        # --- OPTIMIZER--- #
        # create the optimizer to use for training the network in back propogations - [python class]
        # List of PyTorch Loss Functions --> https://pytorch.org/docs/stable/optim.html
        # Adam parameters specifically --> https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # --- Load and Save info ---#
        self.save_dir = models_dir  # save the directory of stored models - [string]
        self.project_name = project_name if project_name != '' else 'network'
        self.model_name = 'model-' + self.project_name + '-' + \
                          datetime.datetime.now().strftime("%m-%d-%Y-%H_%M_%S") + '.pth'

        # define the classes that are being predicted for each pixel - list of strings
        self.classes = ["water", "non-water"]

        # --- DATASETS / DATALOADERS ---#
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = None
        self.test_dataloader = None
        self.dataloader_init()  # init dataloader and print out info

    def dataloader_init(self):
        """This function iniitalizes the dataloader for the dataset in the class
        Input:
            None - Class vars
        Output:
            Void- Dataloaders that are callable
            Also Prints out the shape of the variables to glean insight into data"""

        # Create data loaders.
        if self.train_dataset:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            print("Train dataset is None!")

        if self.test_dataset:
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            print("Test dataset is None!")

        # Print out the dataloaders to vizualize the size of the data
        for X, y in self.test_dataloader:
            print("Shape of X [N, C, H, W]: ", X.shape, X.dtype)
            print("Shape of y: ", y.shape, y.dtype)
            break

    def print_info(self):
        training_size = len(self.train_dataset)
        test_size = len(self.test_dataset)

        logging.info(f'''Starting training:
                Epochs:          {self.epochs}
                Batch size:      {self.batch_size}
                Learning rate:   {self.learning_rate}
                Training size:   {training_size}
                Test size:       {test_size}
                Checkpoints:     {self.save_checkpoint}
                Device:          {self.device}
                Mixed Precision: {self.enable_amp}
                Model params:    {self.count_params()}
                Model size (Mb): {self.get_model_size()}
            ''')

    def train(self, with_validation=False):
        """This function is utilized for training neural networks with data from the dataloader specified
        Training is done by forward predicting the test data, obtaining the loss using the error and the defined loss
        function, using the lass the error is back propogated through the model, then the optimizer uses this to step
        the model weights forward to try and improve the model relative to the learning rate.
        Input:
            self: this passed the wrapper class which contains the optimizer, loss function, learning rate, modle, etc.
            Dataloader: this is a tirch dataloader for the specific data transitioned into a tensor for training
                the size and data-type are varing
        Output: Void()
            Updates Neural Network weights and outputs the loss and current state of training
            """
        assert self.train_dataloader, "Train dataloader is None!"

        self.print_info()

        # Init logging
        experiment = wandb.init(project=self.project_name, resume='allow', anonymous='must')
        experiment.config.update(dict(epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
                                      amp=self.enable_amp))

        # Creates once at the beginning of training
        scaler = torch.cuda.amp.GradScaler()

        # Start training
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            self.network.train()
            epoch_loss = 0

            for batch, (X, y) in enumerate(tqdm.tqdm(self.train_dataloader)):
                print(f"Batch: {batch}")
                X, y = X.to(device=self.device, dtype=torch.float32), y.to(self.device, dtype=torch.long)

                # Casts operations to mixed precision
                with torch.cuda.amp.autocast(enabled=self.enable_amp):
                    pred = self.network(X)
                    loss = self.loss_fn(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(loss).backward()
                # Unscales gradients and calls or skips optimizer.step()
                scaler.step(self.optimizer)
                scaler.update()

                batch_loss, current = loss.item(), batch * len(X)
                epoch_loss += batch_loss

                # log loss for this batch and 1st image of this batch
                experiment.log({'train batch loss': batch_loss,
                                'train epoch loss': epoch_loss,
                                'global step': current,
                                'images': {'original': wandb.Image(X[0].cpu()),
                                           'mask_true': wandb.Image(y[0].float().cpu()),
                                           'mask_pred': wandb.Image(
                                               torch.softmax(pred, dim=1).argmax(dim=1)[0].float().cpu())}
                                })

                # use test dataset for validation
                if with_validation:
                    self.test(experiment=experiment)

                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            if self.save_checkpoint:
                os.mkdir(self.save_dir)
                checkpoint_path = os.path.join(self.save_dir, 'checkpoint_epoch{}.pth'.format(epoch))
                torch.save(self.network.state_dict(), checkpoint_path)
                print(f"Checkpoint {epoch} saved!")

    def test(self, test_dataloader=None, experiment=None):
        """This function takes in data from a dataloader and a model with associated weights and computers the accuracy
        of the network on a test dataset, and then reports out
        Input:
            self: this passes the wrapper class into the function whcih includes the loss function
            Dataloader: this is a tirch dataloader for the specific data transitioned into a tensor for training
                the size and data-type are varing
            model: this should be the entire class which defines the architecture and layers of the neural network
                and associated weights

        Output:Void()
            outputs the accuracy of the training batch"""
        dataloader = test_dataloader if test_dataloader is not None else self.test_dataloader
        assert dataloader, "Test dataloader is None!"

        num_batches = len(dataloader)
        n_classes = len(self.classes)
        print(f"Testing, batches num: {num_batches}, class num: {n_classes}")

        if not experiment:
            experiment = wandb.init(project=self.project_name, resume='allow', anonymous='must')

        # create a wandb Artifact for each meaningful step
        # use the unique wandb run id to organize my artifacts
        test_data_at = wandb.Artifact("test_" + str(experiment.id), type="predictions")
        columns = ["batch_id", "original", "mask_true", "mask_pred", "loss", "dice_score"]
        test_table = wandb.Table(columns=columns)

        # fix model weights/biases
        self.network.eval()

        test_loss, dice_score = 0, 0
        with torch.no_grad():
            for batch_id, (X, y) in enumerate(dataloader):
                X, y = X.to(device=self.device, dtype=torch.float32), y.to(self.device, dtype=torch.long)
                mask_true = F.one_hot(y, n_classes).permute(0, 3, 1, 2).float()

                mask_pred = self.network(X)
                # print("mask_pred: ")
                # print(mask_pred.shape)
                # print(mask_pred[0])
                batch_loss = self.loss_fn(mask_pred, y).item()
                test_loss += batch_loss

                # convert to one-hot format
                if len(self.classes) == 1:
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    dice_coefficient = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

                else:
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_coefficient = multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                        reduce_batch_first=False)
                dice_score += dice_coefficient

                # construct images list to upload to wandb
                for i in range(len(X)):
                    test_table.add_data(batch_id, wandb.Image(X[i].cpu()), wandb.Image(y[i].float().cpu()),
                                        wandb.Image(torch.softmax(mask_pred, dim=1).argmax(dim=1)[i].float().cpu()),
                                        batch_loss, dice_coefficient)

                # log predictions table to wandb, giving it a name
        test_data_at.add(test_table, "predictions")
        experiment.log_artifact(test_data_at)

        test_loss /= num_batches
        dice_score /= num_batches
        print(f"Test Score: \n Avg dice score: {dice_score:>0.1f}%, Avg loss: {test_loss:>8f} \n")

        self.network.train()  # unfix network gradients after evaluation

    def predict(self, data):
        """This function takes in data and creates a prediction based on the current state of teh neural network model
        This is done by forward computing the network and picking the highest value in the resulting logit with argmax
        Input:
            self: This contains the model, the loss function, the optimizer, etc.
            data: the data that the prediction shoul dbe done on
        output: void()
            returns text comparing the predition and the actual value """
        self.network.eval()  # set up the model to evaluate
        x, y = data[0][0], data[0][1]  # pull in the raw data as x and the real classification as y
        with torch.no_grad():
            pred = self.network(x)  # make the prediction and take in the logits
            predicted, actual = self.classes[pred[0].argmax(0)], self.classes[y]  # classify the highest logit
            print(f'Predicted: "{predicted}", Actual: "{actual}"')  # output the prediction

        return predicted

    def save(self):
        # save the model weights
        model_path = os.path.join(self.save_dir, self.model_name)
        torch.save(self.network.state_dict(), model_path)
        print(f"Saved PyTorch Model State to {model_path}!")

    def count_params(self):
        return sum(map(lambda p: p.data.numel(), self.network.parameters()))

    def get_model_size(self):
        param_size = 0
        for param in self.network.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.network.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('model size: {:.3f}MB'.format(size_all_mb))


if __name__ == '__main__':
    d = datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    print(d)
