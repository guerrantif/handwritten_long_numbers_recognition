import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms, datasets
from PIL import Image
import os
import argparse




class MNIST_dataset(datasets.MNIST):

    def __init__(
        self, 
        root_dir: str, 
        train: bool = True, 
        download: bool = True
        ) -> None:

        super(MNIST_dataset, self).__init__(root=root_dir, 
                                            train=train, 
                                            download=download)
        if train:
            self.labels = self.train_labels
            self.images = self.train_data
        else:
            self.labels = self.test_labels
            self.images = self.test_data
        
        self.len = len(self.labels)


    def __len__(self) -> int:

        return self.len


    def __getitem__(
        self, 
        index: int
        ) -> tuple:

        super(MNIST_dataset, self).__getitem__(index)

    
    # def create_splits(
    #     self, 
    #     proportions: list
    #     ) -> list:

    #     # checking argument
    #     if sum(proportions) != 1. or any(p <= 0. for p in proportions):
    #         raise ValueError("Invalid split proportions: they must sum up to 1 and be greater than 0.")

    #     # we want to make balanced splits
    #     indices_per_class = [[i for i in range(self.len) if int(self.labels[i].numpy()) == j] for j in set(self.labels.numpy())]
        
    #     # indices = torch.randperm(len(self.labels))
    #     for _class in indices_per_class:

    #     return indices_per_class



class CNN():
    """Convolutional Neural Network that makes predictions on handwritten digits."""

    def __init__(self, device="cpu"):
        """Create an untrained classifier.

        Args:
            device: the string ("cpu", "cuda:0", "cuda:1", ...) that indicates the device to use.
        """

        # class attributes
        self.num_outputs = 10  # in the handwritten digits classification problem we have 10 classes
        self.device = torch.device(device)  # the device on which data will be moved

        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),

            nn.Conv2d(in_channels=12, out_channels=24, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Flatten(),
            nn.Linear(24 * 4 * 4, 784),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(784, self.num_outputs)
        )

        # moving the network to the right device memory
        self.net.to(self.device)

    def save(self, file_name):
        """Save the classifier (network and data mean and std)."""

        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        """Load the classifier (network and data mean and std)."""

        # since our classifier is a nn.Module, we can load it using pytorch facilities (mapping it to the right device)
        self.net.load_state_dict(torch.load(file_name, map_location=self.device))
        self.net.to(self.device)

    def forward(self, x):
        """Compute the output of the network."""

        logits = self.net(x)  # outputs before applying the activation function
        outputs = F.softmax(logits, dim=1)

        # we also return the logits (useful in order to more precisely compute the loss function)
        return outputs, logits

    @staticmethod
    def decision(outputs):
        """Given the tensor with the net outputs, compute the final decision of the classifier (class label).

        Args:
            outputs: the 2D tensor with the outputs of the net (each row is about an example).

        Returns:
            1D tensor with the main class IDs (for each example).
        """

        # the decision on main classes is given by the winning class (since they are mutually exclusive)
        main_class_ids = torch.argmax(outputs, dim=1)

        return main_class_ids

    def train_classifier(self, train_set, validation_set, batch_size, lr, epochs):

        # initializing some elements
        best_val_acc = -1.  # the best accuracy computed on the validation data (main classes)
        best_epoch = -1  # the epoch in which the best accuracy above was computed

        # ensuring the classifier is in 'train' mode (pytorch)
        self.net.train()

        # creating the optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr)

        # loop on epochs!
        for e in range(0, epochs):

            # epoch-level stats (computed by accumulating mini-batch stats)
            # accuracy is computed on main classes
            epoch_train_acc = 0.
            epoch_train_loss = 0.
            epoch_num_train_examples = 0

            for X, Y in train_set:
                batch_num_train_examples = X.shape[0]  # mini-batch size (it might be different from 'batch_size')
                epoch_num_train_examples += batch_num_train_examples

                X = X.to(self.device)
                Y = Y.to(self.device)

                # computing the network output on the current mini-batch
                outputs, logits = self.forward(X)

                # computing the loss function
                loss = CNN.__loss(logits, Y)

                # computing gradients and updating the network weights
                optimizer.zero_grad()  # zeroing the memory areas that were storing previously computed gradients
                loss.backward()  # computing gradients
                optimizer.step()  # updating weights

                # computing the performance of the net on the current training mini-batch
                with torch.no_grad():  # keeping these operations out of those for which we will compute the gradient
                    self.net.eval()  # switching to eval mode

                    # computing performance
                    batch_train_acc = self.__performance(outputs, Y)

                    # accumulating performance measures to get a final estimate on the whole training set
                    epoch_train_acc += batch_train_acc * batch_num_train_examples

                    # accumulating other stats
                    epoch_train_loss += loss.item() * batch_num_train_examples

                    self.net.train()  # going back to train mode

                    # printing (mini-batch related) stats on screen
                    print("  mini-batch:\tloss={0:.4f}, tr_acc={1:.2f}".format(loss.item(), batch_train_acc))

            val_acc = self.eval_classifier(validation_set)

            # saving the model if the validation accuracy increases
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e + 1
                self.save("classifier.pth")

            epoch_train_loss /= epoch_num_train_examples

            # printing (epoch related) stats on screen
            print(("epoch={0}/{1}:\tloss={2:.4f}, tr_acc={3:.2f}, val_acc={4:.2f}"
                   + (", BEST!" if best_epoch == e + 1 else ""))
                  .format(e + 1, epochs, epoch_train_loss,
                          epoch_train_acc / epoch_num_train_examples, val_acc))

    def eval_classifier(self, data_set):
        """Evaluate the classifier on the given data set."""

        # checking if the classifier is in 'eval' or 'train' mode (in the latter case, we have to switch state)
        training_mode_originally_on = self.net.training
        if training_mode_originally_on:
            self.net.eval()  # enforcing evaluation mode

        # lists on which the mini-batch network outputs (and the targets) will be accumulated
        cpu_batch_outputs = []
        cpu_batch_labels = []

        with torch.no_grad():  # keeping off the autograd engine

            # loop on mini-batches to accumulate the network outputs
            for _, (X, Y) in enumerate(data_set):
                X = X.to(self.device)

                # computing the network output on the current mini-batch
                outputs, _ = self.forward(X)
                cpu_batch_outputs.append(outputs.cpu())
                cpu_batch_labels.append(Y)

            # computing the performance of the net on the whole dataset
            acc = self.__performance(torch.cat(cpu_batch_outputs, dim=0), torch.cat(cpu_batch_labels, dim=0))

        if training_mode_originally_on:
            self.net.train()  # restoring the training state, if needed

        return acc

    @staticmethod
    def __loss(logits, labels):
        """Compute the loss function of the classifier.

        Args:
            logits: the (partial) outcome of the forward operation.
            labels: 1D tensor with the class labels.

        Returns:
            The value of the loss function.
        """

        tot_loss = F.cross_entropy(logits, labels, reduction="mean")
        return tot_loss

    def __performance(self, outputs, labels):
        """Compute the accuracy in predicting the main classes.

        Args:
            outputs: the 2D tensor with the network outputs for a batch of samples (one example per row).
            labels: the 1D tensor with the expected labels.

        Returns:
            The accuracy in predicting the main classes.
        """

        # taking a decision
        main_class_ids = self.decision(outputs)

        # computing the accuracy on main classes
        right_predictions_on_main_classes = torch.eq(main_class_ids, labels)
        acc_main_classes = torch.mean(right_predictions_on_main_classes.to(torch.float) * 100.0).item()

        return acc_main_classes