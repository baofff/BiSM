from torch.utils.data import Subset
from torchvision import datasets
import torchvision.transforms as transforms
from .dataset_factory import DatasetFactory
from .utils import *
import os
from utils.config import config


class Mnist(DatasetFactory):
    r""" train: 50,000
         val:   10,000
         test:  10,000
         shape: 1 * 28 * 28

         train mean: 0.1309
         train biased std: 0.3085
    """

    """  some gauss_noise is good
    """

    def __init__(self):
        super(Mnist, self).__init__()
        self.binarized = config.get("data", "binarized", default=False)
        self.gauss_noise = config.get("data", "gauss_noise", default=True)
        self.noise_std = config.get("data", "noise_std", default=0.01)
        self.preprocess = config.get("data", "preprocess", default=None)
        self.flattened = config.get("data", "flattened", default=True)
        self.padding = config.get("data", "padding", default=False)

        if self.binarized:
            assert not self.gauss_noise
            assert self.preprocess is None
        assert self.preprocess == "standardize" or self.preprocess == "subtract_mean" or self.preprocess is None

        self.data_path = os.path.join("workspace", "datasets", "mnist")

        # Get datasets
        im_transformer = transforms.Compose([transforms.ToTensor()])
        self.train_val = datasets.MNIST(self.data_path, train=True, transform=im_transformer, download=True)
        self.train = Subset(self.train_val, list(range(50000)))
        self.val = Subset(self.train_val, list(range(50000, 60000)))
        self.test = datasets.MNIST(self.data_path, train=False, transform=im_transformer, download=True)

        # digits = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        # self.train = my_filter(self.train, digits)
        # self.val = my_filter(self.val, digits)
        # self.test = my_filter(self.test, digits)

        self.train_mean = 0.1309
        self.train_std = 0.3085

    def augment(self, dataset):
        if self.binarized:
            dataset = BinarizedDataset(dataset)
        if self.gauss_noise:
            dataset = GaussNoiseDataset(dataset, std=self.noise_std)
        return dataset

    def affine_transform(self, dataset):
        if self.padding:
            dataset = PaddedDataset(dataset, pad=2)
        if self.preprocess == "standardize":
            dataset = StandardizedDataset(dataset, mean=self.train_mean, std=self.train_std)
        elif self.preprocess == "subtract_mean":
            dataset = TranslatedDataset(dataset, delta=-self.train_mean)
        if self.flattened:
            dataset = FlattenedDataset(dataset)
        return dataset

    def unpreprocess(self, v):
        if self.padding:
            v = v.view(len(v), 1, 32, 32)
        else:
            v = v.view(len(v), 1, 28, 28)
        if self.preprocess == "standardize":
            v *= self.train_std
            v += self.train_mean
        if self.preprocess == "subtract_mean":
            v += self.train_mean
        if self.padding:
            v = v[..., 2:-2, 2:-2]
        v.clamp_(0., 1.)
        return v
