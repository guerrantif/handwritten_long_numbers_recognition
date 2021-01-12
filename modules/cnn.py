"""
Copyright 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re   # regular expressions
import datetime
from .dataset import MNIST
import os
from tqdm import tqdm


class CNN(nn.Module):
    """
    Convolutional Neural Network class for MNIST dataset.
    """

    def __init__(
          self
        , data_augmentation: bool=True
        , device: str='cpu'
        ) -> None:
        """
        CNN class constructor.

        Args:
            data_augmentation (bool): (default=True) whether to perform data augmentation
            device             (str): device to be used {"cpu", "cuda:0", "cuda:1", ...}
        """

        super(CNN, self).__init__()

        self.num_outputs = 10       # for MNIST dataset: 10-class classification problem
        self.data_augmentation = data_augmentation

        # device setup
        # ----------------------
        if torch.cuda.is_available() and bool(re.findall("cuda:[\d]+$", device)): 
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
        # ----------------------
            
        # model of the CNN
        # ----------------------
        self.net = nn.Sequential(
                # [batch_size, 1, 28, 28]
                nn.Conv2d(
                      in_channels=1
                    , out_channels=12
                    , kernel_size=5
                    , stride=1
                    , padding=0
                )
                # [batch_size, 12, 24, 24]
                , nn.ReLU(inplace=True)
                # [batch_size, 12, 24, 24]
                , nn.MaxPool2d(
                      kernel_size=2
                    , stride=2
                    , padding=0
                )
                # [batch_size, 12, 12, 12]
                , nn.Conv2d(
                      in_channels=12
                    , out_channels=24
                    , kernel_size=5
                    , stride=1
                    , padding=0
                )
                # [batch_size, 24, 8, 8]
                , nn.ReLU(inplace=True)
                # [batch_size, 24, 8, 8]
                , nn.MaxPool2d(
                      kernel_size=2
                    , stride=2
                    , padding=0
                )
                # [batch_size, 24, 4, 4]
                , nn.Flatten()
                # [batch_size, 384]
                , nn.Linear(
                      in_features=4*4*24
                    , out_features=784
                )
                # [batch_size, 784]
                , nn.ReLU(inplace=True)
                # [batch_size, 784]
                , nn.Dropout()
                # [batch_size, 784]
                , nn.Linear(
                      in_features=784
                    , out_features=10
                )
                # [batch_size, self.num_outputs]
        )
        # ----------------------

        # moving network to the correct device memory
        # ----------------------
        self.net.to(self.device)
        # ----------------------

        # the data augmentation consists in rotating the image of a random angle between -30° to +30°
        # ----------------------
        if self.data_augmentation:
            # supposing that input is tensor as provided by dataset class
            self.preprocess = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(30),
                torchvision.transforms.RandomResizedCrop(28, scale=(0.7,1.1)),
            ])
        
        else:
            self.preprocess = None
        # ----------------------


    def save(
          self
        , path: str
        ) -> None:
        """
        Save the classifier.
        All the useful parameters of the network are saved to memory.
        More info here: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        
        Args:
            path   (str): path of the saved file, must have .pth extension
        """  
        torch.save(self.net.state_dict(), path)


    def load(
          self
        , path: str
        ) -> None:
        """
        Load the classifier.
        All the useful parameters of the network are loaded from memory.
        More info here: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        map-location indicates the location where all tensors should be loaded
        
        Args:
            path   (str): name of the saved file, must have .pth extension
        """

        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.to(self.device)


    def forward(
          self
        , x: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute the output of the network.
        After the forward phase over the layers of the network we obtain the so called "logits",
        which (in this case) is a tensor of raw (non-normalized) predictions that the classification 
        model generates, and which is then passed to a normalization function (softmax).
        
        Args:
            x       (torch.Tensor): 4D input Tensor of the CNN [batch_size, 1, 28, 28] 
        
        Returns:
            outputs (torch.Tensor): 2D output Tensor of the net (after softmax act.) [batch_size, 10] 
            logits  (torch.Tensor): 2D output Tensor of the net (before softmax act.) [batch_size, 10]
        """

        logits = self.net(x)
        outputs = F.softmax(logits, dim=1)

        return outputs, logits  # logits returned to compute loss


    @staticmethod
    def __decision(
        outputs: torch.Tensor
        ) -> torch.Tensor:
        """
        Given the tensor with the net outputs, compute the final decision of the classifier (class label).
        The decision is the winning class, aka the class identified by the neuron with greatest value.

        Args:
            outputs     (torch.Tensor): 2D output Tensor of the net (after softmax act.) [batch_size, 10]

        Returns:
            decisions   (torch.Tensor): 1D output Tensor with the decided class IDs [batch_size].
        """

        # decision -> winning class
        # ----------------------
        decisions = torch.argmax(outputs, dim=1)
        # ----------------------

        return decisions


    @staticmethod
    def __loss(
          logits: torch.Tensor
        , labels: torch.Tensor
        , weights: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute the loss function of the cnn.

        Args:
            logits  (torch.Tensor): 2D output Tensor of the net (before softmax act.) [batch_size, 10]
            labels  (torch.Tensor): 1D label Tensor of the outputs [batch_size]
            weights (torch.Tensor): 1D weight Tensor of the classes, usefull in unbalanced datasets. 
                                    For the MNIST dataset the weights are computed dividing the overall
                                    number of examples by the frequency of each class. The classes 
                                    having less samples will be weighted more wrt the ones with more samples

        Returns:
            tot_loss      (float): value of the loss function
        """
        tot_loss = F.cross_entropy(
                              input=logits
                            , target=labels
                            , weight=weights
                            , reduction='mean'
                            )
        return tot_loss


    @staticmethod
    def __performance(
          outputs: torch.Tensor
        , labels: torch.Tensor
        ) -> float:
        """
        Compute the accuracy in predicting the main classes.

        Args:
            outputs     (torch.Tensor): 2D output Tensor of the net (after softmax act.) [batch_size, 10]
            labels      (torch.Tensor): 1D label Tensor of the outputs [batch_size]

        Returns:
            accuracy           (float): accuracy in predicting the main classes
        """

        # taking a decision
        # ----------------------
        decisions = CNN.__decision(outputs)
        # ----------------------

        # computing the accuracy on main classes
        # ----------------------
        correct_decisions = torch.eq(decisions, labels)     # element-wise equality
        accuracy = torch.mean(correct_decisions.to(torch.float) * 100.0).item()
        # ----------------------

        return accuracy


    def train_cnn(
          self
        , training_set: MNIST
        , validation_set: MNIST
        , batch_size: int=64
        , lr: float=0.001
        , epochs: int=10
        , num_workers: int=3
        , model_path: str='./models/'
        ) -> None:
        """
        CNN training procedure.

        Args:
            training_set    (dataset.MNIST): training set
            validation_set  (dataset.MNIST): validation set
            batch_size              (float): batch size
            lr                      (float): learning rate
            epochs                    (int): number of training epochs
            num_workers               (int): number of workers
            model_path              (float): folder in which the trained model will be saved
        """

        # set network in training mode (affect on dropout module)
        # ----------------------
        self.net.train()
        # ----------------------

        # optimizer https://pytorch.org/docs/stable/optim.html
        # ----------------------
        self.optimizer = torch.optim.Adam(
                                  params=filter(    # filter on parameters that require gradient
                                            lambda p: p.requires_grad, 
                                            self.net.parameters()
                                            )
                                , lr=lr
                                )
        # ----------------------

        best_validation_accuracy = -1.                  # best accuracy on the validation data
        best_epoch = -1                                 # epoch in which best accuracy was computed
        self.epochs_validation_accuracy_list = list()   # list of epoch accuracies on validation set
        self.epochs_training_accuracy_list = list()     # list of epoch accuracies on training set

        first_batch_flag = True                         # flag for first mini-batch
    
        # model name for saving
        # ----------------------
        aug = '-a' if self.data_augmentation else ''
        self.model_name = "CNN-batch_size{}-lr{}-epochs{}{}".format(
                        batch_size, lr, epochs, aug)
        filepath = '{}.pth'.format(os.path.join(model_path, self.model_name))
        # ----------------------

        # model folder creation
        # ----------------------
        if not os.path.exists(model_path):   
            os.makedirs(model_path)       
        # ----------------------



        # getting data loaders of datasets
        # ----------------------
        train_loader = training_set.get_loader(
                                  batch_size=batch_size
                                , num_workers=num_workers
                                , shuffle=True
                                )
        
        val_loader = validation_set.get_loader(
                                  batch_size=batch_size
                                , num_workers=num_workers
                                , shuffle=False
                                )
        # ----------------------
        

        # start train phase (looping on epochs)
        # ----------------------
        for e in range(0, epochs):

            # printing epoch number
            print("Epoch {}/{}".format(e + 1,epochs))

            epoch_training_accuracy = 0.        # accuracy of current epoch over training set
            epoch_training_loss = 0.            # loss of current epoch over training set
            epoch_num_training_examples = 0     # accumulated number of training examples for current epoch

            # looping on batches
            # ----------------------
            for X, Y in tqdm(train_loader):
                
                if first_batch_flag:
                    batch_size = X.shape[0]     # take user batch-size
                    first_batch_flag = False    # reset flag

                # generally == batch_size, != in last batch if len(training_set) % batch_size != 0
                batch_num_training_examples = X.shape[0] 
                epoch_num_training_examples += batch_num_training_examples 

                # moving data to correct device (speed process up)
                # ----------------------
                X = X.to(self.device)
                Y = Y.to(self.device)
                # ----------------------

                # forwarding network
                # ----------------------
                outputs, logits = self.forward(X)
                # ----------------------

                # computing loss of network
                # ----------------------
                loss = CNN.__loss(logits, Y, training_set.classes_distribution())
                # ----------------------

                # computing gradients and updating network weights
                # ----------------------
                self.optimizer.zero_grad()      # put all gradients to zero before computing backward phase
                loss.backward()                 # computing gradients (for parameters with requires_grad=True)
                self.optimizer.step()           # updating parameters according to optimizer
                # ----------------------

                # evaluating performances on mini-batches
                # ----------------------
                with torch.no_grad():       # keeping off the autograd engine
                    
                    # setting network out of training mode (affects dropout layer)
                    # ----------------------
                    self.net.eval()         
                    # ----------------------

                    # evaluating performance of current mini-batch
                    # ----------------------
                    batch_training_accuracy = CNN.__performance(outputs, Y)
                    # ----------------------

                    # accumulating accuracy of all mini-batches for current epoch (batches normalized)
                    # ----------------------
                    epoch_training_accuracy += batch_training_accuracy * batch_num_training_examples
                    # ----------------------

                    # accumulating loss of all mini-batches for current epoch (batches normalized)
                    # ----------------------
                    epoch_training_loss += loss.item() * batch_num_training_examples       # loss.item() to access value
                    # ----------------------

                    # switching to train mode
                    # ----------------------
                    self.net.train() 
                    # ----------------------
            # ----------------------   
            # end of mini-batches scope

            # epoch scope
            # ----------------------

            # network evaluation on validation set (end of each epoch)
            # ----------------------
            validation_accuracy = self.eval_cnn(
                                          dataset=validation_set
                                        , batch_size=batch_size
                                        , num_workers=num_workers)
            # ----------------------

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch = e + 1
                
                # saving the best model so far
                # ----------------------
                self.save(path=filepath)
                # ----------------------

            # appending epoch validation accuracy to list
            # ----------------------
            self.epochs_validation_accuracy_list.append(validation_accuracy)
            # ----------------------

            # normalizing epoch accuracy and appending to list
            # ----------------------
            epoch_training_accuracy /= epoch_num_training_examples
            self.epochs_training_accuracy_list.append(epoch_training_accuracy)
            # ----------------------

            # epoch loss computation
            # ----------------------
            epoch_training_loss /= epoch_num_training_examples
            # ----------------------

            # printing (epoch related) stats on screen
            # ----------------------
            print(("loss: {:.4f} - acc: {:.4f} - val_acc: {:.4f}" + (" - BEST!" if best_epoch == e + 1 else ""))
            .format(epoch_training_loss, epoch_training_accuracy, validation_accuracy))
            # ----------------------
        
        # end of epoch scope
        # ----------------------

        # plotting 
        # ----------------------
        self.__plot()
        # ----------------------
        

    def eval_cnn(
          self
        , dataset: MNIST
        , batch_size: int=64
        , num_workers: int=3
        ) -> float:
        """
        CNN evaluation procedure.

        Args:
            dataset (dataset.MNIST): dataset to be evaluated
            batch_size      (float): batch size
            num_workers       (int): number of workers

        Returns:
            accuracy    (float): accuracy of the network on input dataset
        """
        
        # checking if network is in 'eval' or 'train' mode
        # ----------------------
        training_mode_originally_on = self.net.training
        if training_mode_originally_on:
            self.net.eval()         # switch to eval mode
        # ----------------------

        batch_outputs = []          # network outputs
        batch_labels = []           # labels for outputs


        with torch.no_grad():       # keeping off autograd engine

            # getting data loaders of dataset
            # ----------------------
            dataset_loader = dataset.get_loader(
                                      batch_size=batch_size
                                    , num_workers=num_workers
                                    , shuffle=False
                                    )
            # ----------------------
            
            # loop over mini-batches
            # ----------------------
            for X, Y in tqdm(dataset_loader):
                X = X.to(self.device)

                outputs, _ = self.forward(X)
                batch_outputs.append(outputs.cpu())     # append operation forced to be computed in cpu
                batch_labels.append(Y)
            # ----------------------
            
            # computing network performances on validation set
            # ----------------------
            accuracy = CNN.__performance(torch.cat(batch_outputs, dim=0), torch.cat(batch_labels, dim=0))
            # ----------------------

        if training_mode_originally_on:
            self.net.train()    # restoring training state
        
        return accuracy


    def classify(
          self
        , input: torch.tensor
        ) -> torch.tensor:
        """
        CNN classification procedure.

        Args:
            input   (torch.tensor): tensor containing images to be classified

        Returns:
            output  (torch.tensor): tensor containing the classification decision
        """
        
        # checking if network is in 'eval' or 'train' mode
        # ----------------------
        training_mode_originally_on = self.net.training
        if training_mode_originally_on:
            self.net.eval()         # switch to eval mode
        # ----------------------

        batch_outputs = []          # network outputs

        with torch.no_grad():       # keeping off autograd engine
            
            # loop over mini-batches
            # ----------------------
            input.to(self.device)
            outputs, _ = self.forward(input)
            batch_outputs.append(outputs.cpu())     # append operation forced to be computed in cpu
            # ----------------------

        if training_mode_originally_on:
            self.net.train()    # restoring training state
        
        return CNN.__decision(torch.FloatTensor(outputs))


    def __plot(self) -> None:
        """
        Plots validation and testing accuracy over the epochs.
        """
        # retrieve batch_size, lr and epochs from model name
        fields = self.model_name.split('-')
        batch_size = int(fields[1][10:])
        lr = float(fields[2][2:])
        epochs = int(fields[3][6:])

        x = list(range(1, epochs + 1))
        plt.plot(x, self.epochs_training_accuracy_list, label='Training')
        plt.plot(x, self.epochs_validation_accuracy_list, label='Validation')
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy %')
        title = 'data_augmentation={}, batch_size={}, lr={}, epochs={}'.format(self.data_augmentation, batch_size, lr, epochs)
        plt.title(title)
        plt.legend()

        basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder = os.path.join(basedir, 'results/')
        if not os.path.exists(folder):   
            os.makedirs(folder)       
        filepath = os.path.join(folder, self.model_name)
        plt.savefig("{}.png".format(filepath), dpi=1000)