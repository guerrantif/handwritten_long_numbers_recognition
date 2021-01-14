""" modules.cnn.py
Summary
-------
This module contains the convolutional neural network class which will be used as classifier. 

Classes
-------
CNN
    implements the Convolutional Neural Network used as classifier


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

from .dataset import MNIST

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import re 
import os
from tqdm import tqdm


class CNN(nn.Module):
    """ Convolutional Neural Network class for MNIST dataset.

    Derived from torch.nn.Module.

    Provides operation for:
    * save and load the model
    * train the model over a training set
    * evaluate the model over a validation set
    * classify an input (or batch of input)
    """

    def __init__(
        self, 
        data_augmentation: bool=True, 
        device: str='cpu'
        ) -> None:
        """ CNN class constructor.

        Parameters
        ----------
        data_augmentation: bool (default=True) 
            (if True) applies data augmentation techniques: random rotation, random crop and resize
            (if False) trains the network without data augmentation

        device: str (default: 'cpu')
            represents the device {cpu, cuda:0, ...} in which the computation is performed
        """

        super(CNN, self).__init__()

        self.num_outputs = 10       # for MNIST dataset: 10-class classification problem
        self.data_augmentation = data_augmentation

        # device setup
        if torch.cuda.is_available() and bool(re.findall("cuda:[\d]+$", device)): 
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            
        #region - CNN model
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
                    , out_features=self.num_outputs
                )
                # [batch_size, self.num_outputs]
        )
        #endregion

        self.net.to(self.device)

        #region - data augmentation
        if self.data_augmentation:
            # assume that input to CNN is a tensor
            self.preprocess = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(30),
                torchvision.transforms.RandomResizedCrop(28, scale=(0.9,1.1)),
            ])
        
        else:
            self.preprocess = None
        #endregion


    def save(
        self,
        path: str
        ) -> None:
        """ Saves the classifier.

        All the parameters of the model are saved to memory.
        
        Parameters
        ----------
        path: str
            path of the saved file, must have `.pth` extension

        References
        ----------
        [Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        """  
        torch.save(self.net.state_dict(), path)


    def load(
        self,
        path: str
        ) -> None:
        """ Load the classifier.

        All the parameters of the model are loaded from memory.
        
        Parameters
        ----------
        path: str
            path of the saved file, must have `.pth` extension

        References
        ----------
        [Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        """

        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.to(self.device)


    def forward(
        self, 
        x: torch.Tensor
        ) -> (torch.Tensor, torch.Tensor):
        """ Computes the output of the network.

        The forward phase over the layers of the network generates a tensor of non-normalized predictions (logits)
        which is then passed to a normalization function (softmax).
        
        Parameters
        ----------
        x: torch.Tensor)
            4D input tensor of the CNN [batch_size, 1, 28, 28] 
        
        Returns
        -------
        outputs: torch.Tensor
            2D output Tensor of the net (after softmax act.) [batch_size, 10] 

        logits: torch.Tensor
            2D output Tensor of the net (before softmax act.) [batch_size, 10]
        """

        logits = self.net(x)
        outputs = F.softmax(logits, dim=1)

        return outputs, logits


    @staticmethod
    def __decision(outputs: torch.Tensor) -> torch.Tensor:
        """ Computes the decision of the classifier (which class the input belongs to).

        Given the tensor with the net outputs, computes the final decision of the classifier (class label).
        The decision is the winning class, a.k.a. the class identified by the neuron with greatest value.

        Parameters
        ----------
        outputs: torch.Tensor
            2D output Tensor of the net (after softmax act.) [batch_size, 10]

        Returns
        -------
        decisions: torch.Tensor
            1D output Tensor with the decided class IDs [batch_size].
        """
        decisions = torch.argmax(outputs, dim=1)

        return decisions


    @staticmethod
    def __loss(
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        weights: torch.Tensor
        ) -> torch.Tensor:
        """ Compute the loss function of the CNN

        The logits (non-normalized outputs) are used to compute the Cross-Entropy loss.  
        The user can also provide the weights which are used to weight the class unbalanced.  
        In case of MNIST dataset, `weights` is a tensor of dimensions [batch_size, 10] 
        which store the classes distribution.

        Parameters
        ----------
        logits: torch.Tensor
            2D output Tensor of the net (before softmax act.) [batch_size, 10]
        
        labels: torch.Tensor
            1D label Tensor of the outputs [batch_size]
        
        weights: torch.Tensor
            1D weight Tensor of the classes distribution, usefull in unbalanced datasets. 
            For the MNIST dataset the weights are computed dividing the overall number 
            of examples by the frequency of each class. 
            The classes having less samples will be weighted more w.r.t. the more populated ones.

        Returns
        -------
        tot_loss: float
            value of the loss function
        """
        tot_loss = F.cross_entropy(input=logits, target=labels, weight=weights, reduction='mean')
        return tot_loss


    @staticmethod
    def __performance(
        outputs: torch.Tensor, 
        labels: torch.Tensor
        ) -> float:
        """ Computes the prediction accuracy.

        The accuracy is computed as
            # correct predictions / # total predictions

        Parameters
        ----------
        outputs: torch.Tensor
            2D output Tensor of the net (after softmax act.) [batch_size, 10]

        labels: torch.Tensor
            1D label Tensor of the outputs [batch_size]

        Returns
        -------
        accuracy: float
            prediction accuracy
        """
        decisions = CNN.__decision(outputs)
        
        correct_decisions = torch.eq(decisions, labels)     # element-wise equality
        accuracy = torch.mean(correct_decisions.to(torch.float) * 100.0).item()

        return accuracy


    def train_cnn(
        self, 
        training_set: MNIST, 
        validation_set: MNIST, 
        batch_size: int=64, 
        lr: float=0.001, 
        epochs: int=10, 
        num_workers: int=3, 
        model_path: str='./models/'
        ) -> None:
        """ CNN training procedure.

        Exploits the `MNIST.get_loader()` method to get the `DataLoaders` of training and validation set.  
        Iterates over the epochs and applies the `forward()` procedure to each mini-batch.  
        Computes the loss and backpropagates it using the `backward()` method (after zeroed them with zero_grad()).  
        Uses Adam optimizer's `step()` method to update all the (learnable) parameters.  
        Evaluates the performances on the current mini-batch and accumulates the accuracies and the losses.  
        Saves the best model found.

        Parameters
        ----------
        training_set: dataset.MNIST
            training set

        validation_set: dataset.MNIST)
            validation set

        batch_size: float
            batch size

        lr: float
            learning rate

        epochs: int
            number of training epochs

        num_workers: int
            number of workers

        model_path: float
            folder in which the trained model will be saved
        """

        #region - initialization
        self.net.train()

        self.optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr)

        best_validation_accuracy = -1.                  # best accuracy on the validation data
        best_epoch = -1                                 # epoch in which best accuracy was computed
        self.epochs_validation_accuracy_list = list()   # list of epoch accuracies on validation set
        self.epochs_training_accuracy_list = list()     # list of epoch accuracies on training set

        first_batch_flag = True                         # flag for first mini-batch
    
        # model name for saving
        aug = '-a' if self.data_augmentation else ''
        self.model_name = "CNN-{}b-{}e-{}l{}".format(
                        batch_size, epochs, lr, aug)
        filepath = '{}.pth'.format(os.path.join(model_path, self.model_name))

        if not os.path.exists(model_path):   
            os.makedirs(model_path)    

        # getting data loaders of datasets
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
        #endregion

        #region - training over epochs
        for e in range(0, epochs):

            print("Epoch {}/{}".format(e + 1,epochs))

            epoch_training_accuracy = 0.        # accuracy of current epoch over training set
            epoch_training_loss = 0.            # loss of current epoch over training set
            epoch_num_training_examples = 0     # accumulated number of training examples for current epoch

            #region - looping on batches
            for X, Y in tqdm(train_loader):

                # if len(training_set) % batch_size != 0, then batch_size on last iteration is != from user batch_size
                batch_num_training_examples = X.shape[0] 
                epoch_num_training_examples += batch_num_training_examples 

                X = X.to(self.device)
                Y = Y.to(self.device)

                outputs, logits = self.forward(X)

                loss = CNN.__loss(logits, Y, training_set.classes_distribution())
                
                self.optimizer.zero_grad()      # put all gradients to zero before computing backward phase
                loss.backward()                 # computing gradients (for parameters with requires_grad=True)
                self.optimizer.step()           # updating parameters according to optimizer
                

                #region - mini-batch evaluation
                with torch.no_grad():       # keeping off the autograd engine
                    
                    self.net.eval()         
                    
                    batch_training_accuracy = CNN.__performance(outputs, Y)

                    epoch_training_accuracy += batch_training_accuracy * batch_num_training_examples
                    
                    epoch_training_loss += loss.item() * batch_num_training_examples

                    self.net.train() 
                #endregion
            #endregion
            

            #region - validation set evaluation
            validation_accuracy = self.eval_cnn(dataset=validation_set, batch_size=batch_size, num_workers=num_workers)
            #endregion

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch = e + 1
                
                # saving the best model so far
                self.save(path=filepath)

            self.epochs_validation_accuracy_list.append(validation_accuracy)

            epoch_training_accuracy /= epoch_num_training_examples

            self.epochs_training_accuracy_list.append(epoch_training_accuracy)

            epoch_training_loss /= epoch_num_training_examples
            

            print(("loss: {:.4f} - acc: {:.4f} - val_acc: {:.4f}" + (" - BEST!" if best_epoch == e + 1 else ""))
            .format(epoch_training_loss, epoch_training_accuracy, validation_accuracy))
            
        
        #endregion

        self.__plot()
        

    def eval_cnn(
        self, 
        dataset: MNIST, 
        batch_size: int=64, 
        num_workers: int=3
        ) -> float:
        """ CNN evaluation procedure.

        Evaluates the accuracy over the provided dataset by forwarding it (batch by batch) through the model 
        and accumulating the accuracies on each mini-batch.

        Parameters
        ----------
        dataset: dataset.MNIST
            dataset to be evaluated

        num_workers: int
            number of workers

        Returns
        -------
        accuracy: float
            accuracy of the network on input dataset
        """
        
        training_mode_originally_on = self.net.training
        if training_mode_originally_on:
            self.net.eval()

        batch_outputs = []
        batch_labels = []


        with torch.no_grad():       # keeping off autograd engine

            dataset_loader = dataset.get_loader(batch_size=batch_size, num_workers=num_workers, shuffle=False)
            
            #region - mini-batch evaluation
            for X, Y in tqdm(dataset_loader):
                X = X.to(self.device)

                outputs, _ = self.forward(X)
                batch_outputs.append(outputs.cpu())     # append operation forced to be computed in cpu
                batch_labels.append(Y)
            #endregion
                
            accuracy = CNN.__performance(torch.cat(batch_outputs, dim=0), torch.cat(batch_labels, dim=0))

        if training_mode_originally_on:
            self.net.train()
        
        return accuracy


    def classify(
        self, 
        input: torch.tensor
        ) -> torch.tensor:
        """ CNN classification procedure.

        Forwards an input sample (or batch of samples) through the model and makes a decision.

        Parameters
        ----------
        input: torch.tensor
            tensor containing images to be classified

        Returns
        -------
        output: torch.tensor
            tensor containing the classification decision
        """
        
        training_mode_originally_on = self.net.training
        if training_mode_originally_on:
            self.net.eval()

        batch_outputs = []

        with torch.no_grad():       # keeping off autograd engine

            input.to(self.device)
            outputs, _ = self.forward(input)
            batch_outputs.append(outputs.cpu())     # append operation forced to be computed in cpu

        if training_mode_originally_on:
            self.net.train()
        
        return CNN.__decision(torch.FloatTensor(outputs))


    def __plot(self) -> None:
        """ Plots validation and training accuracy over the epochs. """

        #region - parameters retrieval
        fields = self.model_name.split('-')
        batch_size = int(fields[1][:-1])
        epochs = int(fields[2][:-1])
        lr = float(fields[3][:-1])
        #endregion

        #region - plotting
        x = list(range(1, epochs + 1))
        plt.plot(x, self.epochs_training_accuracy_list, label='Training')
        plt.plot(x, self.epochs_validation_accuracy_list, label='Validation')
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy %')
        title = 'data_augmentation={}, batch_size={}, epochs={}, lr={}'.format(self.data_augmentation, batch_size, epochs, lr)
        plt.title(title)
        plt.legend()
        #endregion

        #region - save plot
        basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder = os.path.join(basedir, 'results/')
        if not os.path.exists(folder):   
            os.makedirs(folder)       
        filepath = os.path.join(folder, self.model_name)
        plt.savefig("{}.png".format(filepath), dpi=1000)
        #endregion