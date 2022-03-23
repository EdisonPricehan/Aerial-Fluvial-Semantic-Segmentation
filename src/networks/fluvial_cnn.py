#!E:\anaconda/python

import torch
from torch import nn
from torch.utils.data import DataLoader
import datetime


class FluvialCNN:
    """This class contains the dataset, dataloader, and all other respective initiated values for a fluvial CNN it holds
    everything except the network itself. In this way the same dataloader and dataset cna be utilized with several
    different trial networks that are defined elsewhere. The fluvial CNN class also contains prediction parameters, and
    will load in parameters."""
    def __init__(self, network, train_dataset, test_dataset, models_dir):
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
        self.network = network  # this class takes in network class as a subclass to do forward and backward propagation
        self.network.to(self.device)

        # --- HYPERPARAMETERS--- #
        self.batch_size = 32  # batch size for training of the CNN network - [int]
        self.learning_rate = 1e-5  # Define the learning rate of the network - [float64]
        self.epochs = 1000  # Define the number of epochs to train for - [int]
        self.shuffle = True  # should the test-train dataset be re-shuffled??

        # --- LOSS FUNCTION --- #
        # loss functions can be implimented from pytorch --> https://pytorch.org/docs/stable/nn.html#loss-functions
        # can also impliment custom functions --> https://discuss.pytorch.org/t/custom-loss-functions/29387
        # 2020 article about segmentation --> https://arxiv.org/pdf/2006.14822.pdf
        self.loss_fn = nn.CrossEntropyLoss  # define the loss function that will be used - [pytorch class]

        # --- OPTIMIZER--- #
        # create the optimizer to use for training the network in back propogations - [python class]
        # List of PyTorch Loss Functions --> https://pytorch.org/docs/stable/optim.html
        # Adam parameters specifically --> https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # --- Load and Save info ---#
        self.save_dir = models_dir  # save the directory of stored models - [string]
        self.model_name = "prelim_training" + str(datetime.date)  # current training model name for save - [string]

        # define the classes that are being predicted for each pixel - list of strings
        self.classes = ["water", "non-water"]

        # --- DATASETS ---#
        self.train_dataset = train_dataset  # Pass in the training dataset
        self.test_dataset = test_dataset  # pass in the testing dataset
        self.train_dataloader = None  # initialize the training dataloader to be filled later
        self.test_dataloader = None  # initiatie the test dataloader to be filled later
        self.dataloader_init()  # create datasets and print out info

    def dataloader_init(self):
        """This function iniitalizes the dataloader for the dataset in the class
        Input:
            None - Class vars
        Output:
            Void- Dataloaders that are callable
            Also Prints out the shape of the variables to glean insight into data"""

        # Create data loaders.
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        # Print out the dataloaders to vizualize the size of the data
        for X, y in self.test_dataloader:
            print("Shape of X [N, C, H, W]: ", X.shape)
            print("Shape of y: ", y.shape, y.dtype)
            break

    def train(self):
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
        size = len(self.train_dataloader)  # calculate the size of the dataset.
        # for the values in the batch and do for the input (x) and output(y)
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            X = X.float()

            # Compute prediction error
            pred = self.network(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self):
        """This function takes in data from a dataloader and a model with associated wieights and computers the accuracy
        of the network on a test dataset, and then reports out
        Input:
            self: this passes the wrapper class into the function whcih includes the loss function
            Dataloader: this is a tirch dataloader for the specific data transitioned into a tensor for training
                the size and data-type are varing
            model: this should be the entire class which defines the architecture and layers of the neural network
                and associated weights

        Output:Void()
            outputs the accuracy of the training batch"""
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.network.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                X = X.float()
                pred = self.network(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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
        torch.save(self.network.state_dict(), self.save_dir)
        print("Saved PyTorch Model State to model.pth")

