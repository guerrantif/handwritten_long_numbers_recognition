"""
Copyright December 2020 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

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
import torch.nn as nn
import torch.nn.functional as F
import re   # regular expressions
from typing import Any, Callable, Optional, Sequence



class CNN(nn.Module):

    num_outputs: int
    name: str
    device: str
    net: nn.Sequential


    def __init__(
          self
        , model: Optional[str]='model1'
        , device: Optional[str]='cpu'
        ) -> None:
        """
        CNN class constructor.

        Args:
            model   (str): {'model1' (default), 'model2'} several models are provided for this example.
            device  (str): {'cpu}
        """

        super(CNN, self).__init__()

        self.num_outputs = 10       # for MNIST dataset: 10-class classification problem
        self.name = "CNN-{0}".format(model)

        # device setup
        if torch.cuda.is_available() and bool(re.findall("cuda:[\d]+$", device)): 
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")


        if model == "model1":

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

        else:
            raise ValueError('{0}: undefined model'.format(model))

        # moving network to the correct device memory
        self.net.to(self.device)


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


    def save_all(
          self
        , epoch: int
        , loss: torch.Tensor
        , path: str
        ) -> None:
        """
        Save the classifier and the other hyperparameters.
        All the useful parameters of the network are saved to memory, plus other informations
        such as the number of epochs and the optimizer parameters.
        More info here: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        
        Args:
            epochs        (int): number of epochs computed till the current moment
            loss (torch.Tensor): loss of the network till the current moment
            path          (str): path of the saved file, must have .pth extension
        """

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_state_dict': loss.state_dict()
                    }, path)


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


    def load_all(
          self
        , path: str
        ) -> Sequence:
        """
        Load the classifier and the other hyperparameters.
        All the useful parameters of the network are loaded from memory, plus other informations
        such as the number of epochs and the optimizer parameters.
        More info here: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        map-location indicates the location where all tensors should be loaded
        
        Args:
            path          (str): path of the saved file, must have .pth extension

        Returns:
            epochs        (int): number of epochs computed till the saved moment
            loss (torch.Tensor): loss of the network till the saved moment
        """

        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'], map_location=self.device)
        self.net.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], map_location=self.device)
        self.optimizer.to(self.device)
        loss.load_state_dict(checkpoint['loss_state_dict'], map_location=self.device)
        loss.to(self.device)
        epoch = checkpoint['epoch']

        return epoch, loss


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
        decisions = torch.argmax(outputs, dim=1)

        return decisions


    @staticmethod
    def __loss(
          logits: torch.Tensor
        , labels: torch.Tensor
        , weights: Optional[torch.Tensor]=torch.Tensor([10.1300,  8.8994, 10.0705,  9.7863, 10.2705, 11.0681, 10.1386,  9.5770, 10.2547, 10.0857])
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
          self
        , outputs: torch.Tensor
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
        decisions = CNN.__decision(outputs)

        # computing the accuracy on main classes
        right_decisions = torch.eq(decisions, labels)     # element-wise equality
        accuracy = torch.mean(right_decisions.to(torch.float) * 100.0).item()

        return accuracy


    def train_cnn(
          self
        , training_set: torch.utils.data.DataLoader
        , validation_set: torch.utils.data.DataLoader
        , batch_size: Optional[int]=64
        , optimizer_mode: Optional[str]="adam"
        , lr: Optional[float]=0.001
        , epochs: Optional[int]=10
        , momentum: Optional[float]=0.5
        ) -> None:
        """
        CNN training procedure.

        Args:
            training_set    (DataLoader): DataLoader of the training set
            validation_set  (DataLoader): DataLoader of the validation set
            batch_size             (int): number of samples for each mini-batch
            optimizer_mode         (str): {"adam", "sgd"}, type of optimizer
            lr                   (float): learning rate
            epochs                 (int): number of training epochs
            momentum             (float): momentum for the SGD optimizer
        """
        
        best_validation_accuracy = -1.  # best accuracy on the validation data
        best_epoch = -1                 # epoch in which best accuracy was computed

        # set network in training mode (affect on dropout module)
        self.net.train()

        # optimizer https://pytorch.org/docs/stable/optim.html
        if optimizer_mode == "adam":
            self.optimizer = torch.optim.Adam(
                                  params=filter(lambda p:   # filter on parameters that require gradient
                                                    p.requires_grad, 
                                                    self.net.parameters()
                                                )
                                , lr=lr
                                )
            self.optimizer.to(self.device)

        elif optimizer_mode == "sgd":
            self.optimizer = torch.optim.SGD(
                                  params=filter(lambda p:   # filter on parameters that require gradient
                                                    p.requires_grad, 
                                                    self.net.parameters()
                                                )
                                , lr=lr
                                , momentum=momentum
                                )
            self.optimizer.to(self.device)

        else:
            raise ValueError("Invalid optimizer {}: \'adam\' or \'sgd\' must be provided")



        # start train phase (looping on epochs)
        # ----------------------
        for e in range(0, epochs):

            epoch_train_accuracy = 0.           # accuracy of current epoch over training set
            epoch_train_loss = 0.               # loss of current epoch over training set
            epoch_num_training_examples = 0     # accumulated number of training examples for current epoch

            # looping on batches
            # ----------------------
            for X, Y in training_set:
                
                # generally == batch_size, != in last batch if len(training_set) % batch_size != 0
                batch_num_training_examples = X.shape[0] 
                epoch_num_training_examples += batch_num_training_examples 

                # moving data to correct device (speed process up)
                X = X.to(self.device)
                Y = Y.to(self.device)

                # forwarding network
                outputs, logits = self.forward(X)

                # computing loss of network
                loss = CNN.__loss(logits, Y)

                # computing gradients and updating network weights
                optimizer.zero_grad()       # put all gradients to zero before computing backward phase
                loss.backward()             # computing gradients (for parameters with requires_grad=True)
                optimizer.step()            # updating parameters according to optimizer

                # evaluating performances on mini-batches
                with torch.no_grad():       # keeping off the autograd engine

                    self.net.eval()         # setting network out of training mode (affects dropout layer)

                    # evaluating performance of current mini-batch
                    batch_train_accuracy = self.__performance(outputs, Y)

                    # accumulating accuracy of all mini-batches for current epoch (batches normalized)
                    epoch_train_accuracy += batch_train_accuracy * batch_num_training_examples

                    # accumulating loss of all mini-batches for current epoch (batches normalized)
                    epoch_train_loss += loss.item() * batch_num_training_examples       # loss.item() to access value

                    # printing (mini-batch related) stats on screen
                    print("  mini-batch:\tloss={0:.4f}, tr_acc={1:.2f}".format(loss.item(), batch_train_accuracy))

                    # switching to train mode
                    self.net.train() 

            # ----------------------   
            # end of mini-batches scope

            # epoch scope
            # ----------------------

            # netwrok evaluation on validation set (end of each epoch)
            validation_accuracy = self.eval_cnn(validation_set)

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch = e + 1
                self.save_all(epoch=e, loss=loss, path="./models/CNN-{0}.pth".format(self.name))

            # epoch loss computation
            epoch_train_loss /= epoch_num_training_examples

            # printing (epoch related) stats on screen
            print(("epoch={0}/{1}:\tloss={2:.4f}, tr_acc={3:.2f}, val_acc={4:.2f}"
                   + (", BEST!" if best_epoch == e + 1 else ""))
                  .format(e + 1, epochs, epoch_train_loss,
                          epoch_train_acc / epoch_num_train_examples, val_acc))


    def eval_cnn(
          self
        , data_set: torch.utils.data.DataLoader
        ) -> float:
        """
        CNN evaluation procedure.

        Args:
            data_set    (DataLoader): DataLoader of the validation set

        Returns:
            accuracy         (float): accuracy of the network on the validation set
        """
        
        # checking if network is in 'eval' or 'train' mode
        training_mode_originally_on = self.net.training
        if training_mode_originally_on:
            self.net.eval()         # switch to eval mode

        batch_outputs = []          # network outputs
        batch_labels = []           # labels for outputs


        with torch.no_grad():       # keeping off autograd engine
            
            # loop over mini-batches
            for X, Y in data_set:
                X = X.to(self.device)

                outputs, _ = self.forward(X)
                batch_outputs.append(outputs.cpu())     # append operation forced to be computed in cpu
                batch_labels.append(Y)
            
            # computing network performances on validation set
            accuracy = CNN.__performance(torch.cat(batch_outputs, dim=0), torch.cat(batch_labels, dim=0))

        if training_mode_originally_on:
            self.net.train()    # restoring training state
        
        return accuracy