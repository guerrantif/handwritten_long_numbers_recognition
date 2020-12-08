{
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5-final"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "python38564bitf5e98526d62c4af1bdda9906588672b8",
   "display_name": "Python 3.8.5 64-bit"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2,
 "cells": [
  {
   "source": [
    "# Handwritten long number recognition\n",
    "---\n",
    "This notebook is a step by step tutorial for the implementation of a **Convolutional Neural Network (CNN)** capable of recognizing a long sequence of handwritten digits.\n",
    "\n",
    "\n",
    "Here the outline:\n",
    "\n",
    " * Dataset Analysis\n"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "source": [
    "## Dataset Analysis\n",
    "\n",
    "For this project we are using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). This dataset is composed by handwritten digits and it is divided as follows:\n",
    "```\n",
    "* training set: 60'000 examples\n",
    "* test set:     10'000 examples\n",
    "```\n",
    "It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.\n",
    "\n",
    "\n",
    "To complete the task of loading the dataset we exploit the `torchvision.datasets` class.\n",
    "\n",
    "\n",
    "From the [documentation](https://pytorch.org/docs/stable/torchvision/datasets.html):\n",
    "\n",
    "\"All datasets are subclasses of `torch.utils.data.Dataset` i.e, they have `__getitem__` and `__len__` methods implemented. Hence, they can all be passed to a `torch.utils.data.DataLoader` which can load multiple samples parallelly using torch.multiprocessing workers.\""
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torchvision import datasets\n",
    "from PIL import Image\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "mnist_dataset_train = datasets.MNIST(root='data/', train=True, download=True)\n",
    "mnist_dataset_test = datasets.MNIST(root='data/', train=False)"
   ]
  },
  {
   "source": [
    "The `torchvision.datsets.MNIST(...)` module has several parameters.\n",
    "\n",
    "* root (string): root directory of dataset\n",
    "* train (bool, optional): if True, creates dataset from `training.pt`, otherwise from `test.py` (default=True)\n",
    "* download (bool, optional): if True, downloads the dataset from the internet and puts it in root.\n",
    "* transform (callable, optional): a function/transform which takes in a PIL image and returns a transformed version \n",
    "\n",
    "For the moment, we focus on the download phase of the dataset.\n",
    "\n",
    "Let's see some **statistics**:"
   ],
   "cell_type": "markdown",
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Number of training samples:\t60000\nNumber of test samples:\t\t10000\nShape of each sample:\t\t(28, 28)\nNumber of classes:\t\t10\nClasses:\t\t\t{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}\n\n\nEXAMPLES: [5, 0, 4, 1, 9, 2]\n"
     ]
    },
    {
     "output_type": "display_data",
     "data": {
      "text/plain": "<Figure size 432x288 with 6 Axes>",
      "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"62.228571pt\" version=\"1.1\" viewBox=\"0 0 349.2 62.228571\" width=\"349.2pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <metadata>\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2020-12-07T00:03:03.475987</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.3.3, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 62.228571 \nL 349.2 62.228571 \nL 349.2 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g clip-path=\"url(#p880f39211e)\">\n    <image height=\"48\" id=\"image73f6642a61\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"7.2\" xlink:href=\"data:image/png;base64,\n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\" y=\"-7.028571\"/>\n   </g>\n  </g>\n  <g id=\"axes_2\">\n   <g clip-path=\"url(#pe1b4673519)\">\n    <image height=\"48\" id=\"image3d53dc7623\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"64.594286\" xlink:href=\"data:image/png;base64,\n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\" y=\"-7.028571\"/>\n   </g>\n  </g>\n  <g id=\"axes_3\">\n   <g clip-path=\"url(#p4e8036fd2f)\">\n    <image height=\"48\" id=\"image41388b4446\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"121.988571\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAFlklEQVR4nO2Zu28TSxSHvx2v7V3ba29sR4Q4JpYwkCaIVwMBRIVEQ0vLn8N/ghAFBR0SDQ8XFIgkPELAgjyUBcfr9SPr53r3FlexiIhzHTtgrsTXuJlzdn4z55w5M5YAj/8xYtwTGJW/AsaNPI6PCiEQQiDLMp7n4boujuPgeYdPx98uwOfzsbCwwNzcHHfv3mVtbY2lpSUePnzI58+fD+3vtwuQJIlUKsWpU6e4dOkSkUiESqVCKBQayt9vzwEhBPPz88zPzyOEQJKk0fwd0bwOhc/nw+fzAQwV9z8ytirked7Ik4c/oIxalsXHjx/Z2dkZyn6sAiRJotlsUiqV6HQ6Q/kYm4Afk3eUcBp7CLXbbWq1Gt1udyj7sSexbdsYhkGz2RzKz9h3YFTGLkCSJHw+39AH2tgFTE9Pc+3aNeLx+FD2fXshWZaJx+P4/X6CwWBfB67rYts2juPQbrfpdDq02+0DP2rbNvV6HYBYLEYmkxm6F+orIJFIcOfOHdLpNLOzs30ddLtdcrkc379/Z21tDcMwWF9f7zvedV3ev3+PqqrcunWLmZkZbty4wYMHD45GgBCC8+fPk81muX79OrquE4vFaDQaOI6zZ2w4HCYYDBIIBKhWq5imiWVZbG9vU6vVeodUo9GgUqlQLpep1+t8+fIFTdOo1+sIIdB1Hb/ffzQCZFnm5s2bXLhwgdu3b+M4Do1Gg+3tbRqNxp6x09PTJJNJzp49i+d5dLvdXihtbGxgmibv3r2jUCiQz+f59OkTm5ubfPjwAYBqtYqu60xMTBAIBI5GAIDf78fv9yNJEsvLyzx58oSlpSWKxeKecalUimQyiSRJKIpCNptFURRUVaXdbiOE4PLly8C/cV+tVqnVauTzeaLRKKFQCFmWEUKQSCSYmpqiUCjguu5oAnavfACFQoE3b96Qy+UwDGPPuKmpKeLxOEIIIpEIpmkSjUaJRqPIskwoFGJubg5VVYnFYhw/fhzXdUkkEr1F2r0TaJqGrusUi8XRBfxIu92mUqns22wVCgVM0wT+redv375FkqSeeEmSuHfvHn6/H1VVOXnyJJlMhitXrpBOp/H7/ciyjCRJpNNpstksX79+/SnXRhIQCATQNG3fJHNdd89q9SufPp+vF+O2bRMMBjFNk1QqRTweZ2JighMnTjA3N8fLly/pdDoD90b/KUDXdU6fPs3y8vJADvej2+3SaDRYXV1ldXWVZ8+ekclkiEajzM/Pc/HiRa5evUoqleLRo0e0Wq3eOXFoAZ7nYVkWlmXheR7hcJipqakDD7NhqFQqPH78mE6nw5kzZ4hEIkxOTqKqKoFAYGABP7USnudRLpcpl8t4noeiKL2kO0pqtRovXrxgZWWFnZ0dFEVB13UURUGWB38s+Wmk67osLi4ihKDZbB4qoQ6D4zi9Q88wDFRVRQhBPB6nVCr9VLL7se8OlEolTNOk1WoNfdEYBMdxevHuui5CCCYmJtA0bWAf++bAxsYGuq5jmubAsTgsnU6n1wwGg0HOnTtHt9tlcXFxIPt92+nd98rd31/JxsYGT58+pVgs9nYgEokMbH/gfeBXhs8uKysr3L9/n83NTSRJIpFIEIvFBrbvm+61Wo3nz5+jqmrvFe1X0Gq1KJVKvHr1CkmSOHbsGLOzs8zMzGBZFrZtH2jfdwdarRb5fJ5v377RbDaP5BVtPxzHodlssrW1xfr6OoqioGkakUhkoA617w7Yts3r16/RNI1wODxwWRsG13UxTZOtrS0mJyd7h+ggi9ZXQKfTwTAMLMtCUZRfXo2q1SqWZREKhVBVdWC7vgKazSYrKytHMrlBMAwDTdNIp9OUy+WB7ST+kL9Zk8kkuq6zsLBAsVgkl8tRr9dptVoH2v0xAoZl7O9Co/JXwLj5K2Dc/BUwbv4B9iuEWFRdFbwAAAAASUVORK5CYII=\" y=\"-7.028571\"/>\n   </g>\n  </g>\n  <g id=\"axes_4\">\n   <g clip-path=\"url(#p1df646a61f)\">\n    <image height=\"48\" id=\"imagec285a9cfb2\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"179.382857\" xlink:href=\"data:image/png;base64,\niVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAFJElEQVR4nO2ZzW8S6xfHPzPDEKDTUCgtxVokUGuw6oKIxEQTE0PiQpdudKELd678G/wL3HSrbly7aXRDjN2VJsat1Ixtsa3yZtuZQnmZee7ipuT293LTwkDvTfwkT0hI5pzz5eGcOed5JEDwL0Y+7QD65beA08bV18MuF4qiIEkSAEIILMui0+k4EtyxYujlIbfbzejoKPfu3ePGjRucP38el8tFpVLh7du3vHz50uk4/y89CXC5XGiaRiKRIJVKdQVsb2+Tz+edjvFv6SkHvF4v586dY25ujvn5eXw+n9NxHZueBEiShCzLyLKMoihOx3Qieq5CQogjn4ccJvSw6ElAp9Nhf38fwzDY3d3FsixcLheBQICrV6/y6NEjYrEYLldfRe5Y9CTAsiwajQaGYWAYBrZtI8syXq+XaDRKJpMhFAqhqurAd6QnAaZpUigU+Pz5MysrK9TrdWRZRlVVLl68yIMHD0gmk4yNjf0zBQghaLfb6LrOysoK+/v73VxQVRVN0wgEAgSDQWR5sC/7vqx//PiRhYUFSqUSQgiEENi2jRCCs2fPMjs7O/A86PvnsW2ber3OwcHBkYqUTCbJZDJomjZQEY5YPqxIwWCw+59PJBIA+P1+Dg4OME3TCVf/E9HPUhRFZLNZ8ezZM6HruqjVaqLdbgvDMMTW1pZYWFgQjx8/7svH362+d8C2bQqFQreZU1WV0dFRPB4Psixz6dIlqtUqHo+HdruNZVn9ujxC3wKEEGxsbCBJEsvLy6RSKaampoA/u9ZMJoNpmkSjUUqlEjs7O/26PIIjNU4IgWmafPr0qSvmcCmKwvj4OOl0msnJSSfcHcGxIr23t0cul6NQKBwRIEkS4XCYbDZLNBp1yl0Xx+pbp9OhXC7z4cMH3G43d+/eJR6P43a7cbvdhMNhxsfH8fv9mKbpWC44tgO2bdNoNPj27RtLS0usra2xu7uLbduoqsrY2BihUIiJiQnHeyRHy5qiKMLr9Yr79++L58+fi1qtJlqtlmg0GmJ5eVm8fv1azM/PC5/P988oo//JYadaLBbRNI2dnR1UVcXn8zE1NYVlWSQSCdrtNqurq/81T5yUgb3jdV2n1Wqh6zpCCGKxGNPT00xMTHDnzh0mJyfRdb3vE4yBtYqGYbC5ucmrV694//5993tFUUilUly/fp25uTlCoVBffga2A41Gg2azyeLiIgBPnjzpztEXLlxAURTi8TiWZVGpVHr2M9Bm3bZt9vb2+Pr1K4uLi6yvryPLMiMjI0SjUR4+fEg6ne7Lx8CPFi3LwjAMCoUC1WqVTqeDLMv4fD5mZ2eZmZkhGAyiqmpP9gc/dQM/f/7k3bt3hMNhZmZmCAQCeDwerly5wo8fP1hbW2NpaYnNzc0T2x7K4W69XqdYLJLP58nlctRqNdrtNrIsE4lEuHbtGpFIpDv8nGQMHZoAXdfJ5XK8efOG7e1tms0mANFolNu3bxOPxwkEArjd7hMdlkkM8YZG0zT8fj9Pnz7l8uXLZLNZbNum2Wzy5csX1tfXefHiBd+/f2djY+NYNod6P2CaJqVSCV3XKRaL3T5J0zTS6TS3bt0iHA6jadqxbQ79gkMIQaVSoVwu02q1+u5Khy7Atm22trbQdZ3V1VXK5XK3M+2lQx1KGf0rtm2Tz+cpFotEIhFu3rzJmTNnkGW5JyFDTeK/MjIyQjKZZHp6mlgshiRJNBoNcrkcv379olqtHsvOqQlwin/9LeVvAafNbwGnzW8Bp80fcN9hKbAFfrcAAAAASUVORK5CYII=\" y=\"-7.028571\"/>\n   </g>\n  </g>\n  <g id=\"axes_5\">\n   <g clip-path=\"url(#pdfbafd0d3e)\">\n    <image height=\"48\" id=\"image8d9cb8356d\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"236.777143\" xlink:href=\"data:image/png;base64,\n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\" y=\"-7.028571\"/>\n   </g>\n  </g>\n  <g id=\"axes_6\">\n   <g clip-path=\"url(#pd951f93370)\">\n    <image height=\"48\" id=\"imagee01303e9ae\" transform=\"scale(1 -1)translate(0 -48)\" width=\"48\" x=\"294.171429\" xlink:href=\"data:image/png;base64,\n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\" y=\"-7.028571\"/>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p880f39211e\">\n   <rect height=\"47.828571\" width=\"47.828571\" x=\"7.2\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"pe1b4673519\">\n   <rect height=\"47.828571\" width=\"47.828571\" x=\"64.594286\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"p4e8036fd2f\">\n   <rect height=\"47.828571\" width=\"47.828571\" x=\"121.988571\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"p1df646a61f\">\n   <rect height=\"47.828571\" width=\"47.828571\" x=\"179.382857\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"pdfbafd0d3e\">\n   <rect height=\"47.828571\" width=\"47.828571\" x=\"236.777143\" y=\"7.2\"/>\n  </clipPath>\n  <clipPath id=\"pd951f93370\">\n   <rect height=\"47.828571\" width=\"47.828571\" x=\"294.171429\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n</svg>\n",
      "image/png": "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\n"
     },
     "metadata": {
      "needs_background": "light"
     }
    }
   ],
   "source": [
    "n_train = mnist_dataset_train.__len__()\n",
    "n_test = mnist_dataset_test.__len__()\n",
    "ex_shape = (mnist_dataset_train.data.shape[1], mnist_dataset_train.data.shape[2])\n",
    "classes = set(mnist_dataset_train.targets.numpy())\n",
    "n_classes = len(classes)\n",
    "images, targets = [], []\n",
    "for i in range(6):\n",
    "    img, trg = mnist_dataset_train.__getitem__(i)\n",
    "    images.append(img)\n",
    "    targets.append(trg)\n",
    "\n",
    "print('Number of training samples:\\t{}'.format(n_train))\n",
    "print('Number of test samples:\\t\\t{}'.format(n_test))\n",
    "print('Shape of each sample:\\t\\t{}'.format(ex_shape))\n",
    "print('Number of classes:\\t\\t{}'.format(n_classes))\n",
    "print('Classes:\\t\\t\\t{}'.format(classes))\n",
    "print('\\n\\nEXAMPLES: {}'.format(targets))\n",
    "\n",
    "fig = plt.figure()\n",
    "columns, rows = 6, 1\n",
    "for i in range(1, columns*rows +1):\n",
    "    img = images[i-1]\n",
    "    fig.add_subplot(rows, columns, i)\n",
    "    plt.axis('off')\n",
    "    plt.imshow(img, cmap='gray', )\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from __utils__ import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "cnn = CNN()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = [[]]*10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "[[], [], [], [], [], [], [], [], [], []]"
      ]
     },
     "metadata": {},
     "execution_count": 6
    }
   ],
   "source": [
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = MNIST_dataset(root_dir='data/', train=True, download=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "tensor([0.4907, 0.3432, 0.4342, 0.9406, 0.6997, 0.1193, 0.7470, 0.2540, 0.5452,\n        0.3822])\ntensor([8, 2, 9, 1, 4, 0, 6, 3, 7, 5])\ntensor([0.5452, 0.4342, 0.3822, 0.3432, 0.6997, 0.4907, 0.7470, 0.9406, 0.2540,\n        0.1193])\n"
     ]
    }
   ],
   "source": [
    "import random\n",
    "import numpy as np\n",
    "\n",
    "a = np.arange(10)\n",
    "t = torch.rand(10)\n",
    "print(t)\n",
    "r = torch.randperm(10)\n",
    "print(r)\n",
    "t = t[r]\n",
    "print(t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = [0,0,0,1,0,2,2,1,2,0,3,1,2,1,3,3,2,1,3,3]\n",
    "b = [[i for i in range(len(a)) if a[i] == j] for j in set(a)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "[[0, 1, 2, 4, 9], [3, 7, 11, 13, 17], [5, 6, 8, 12, 16], [10, 14, 15, 18, 19]]"
      ]
     },
     "metadata": {},
     "execution_count": 10
    }
   ],
   "source": [
    "b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "tensor([5, 0, 4,  ..., 5, 6, 8])"
      ]
     },
     "metadata": {},
     "execution_count": 11
    }
   ],
   "source": [
    "dataset.labels"
   ]
  }
 ]
}