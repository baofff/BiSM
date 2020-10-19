from torch.utils.data import Subset
from torchvision import datasets
import torchvision.transforms as transforms
from .dataset_factory import DatasetFactory
from .utils import *
import os
from utils.config import config


class Cifar10(DatasetFactory):
    r""" train: 40,000
         val:   10,000
         test:  10,000
         shape: 3 * 32 * 32
    """

    def __init__(self):
        super(Cifar10, self).__init__()
        self.binarized = config.get("data", "binarized", default=False)
        self.gauss_noise = config.get("data", "gauss_noise", default=True)
        self.noise_std = config.get("data", "noise_std", default=0.01)
        self.flattened = config.get("data", "flattened", default=True)

        if self.binarized:
            assert not self.gauss_noise

        self.data_path = os.path.join("workspace", "datasets", "cifar10")

        # Get datasets
        im_transformer = transforms.Compose([transforms.ToTensor()])
        self.train_val = datasets.CIFAR10(self.data_path, train=True, transform=im_transformer, download=True)
        self.train = Subset(self.train_val, list(range(40000)))
        self.val = Subset(self.train_val, list(range(40000, 50000)))
        self.test = datasets.CIFAR10(self.data_path, train=False, transform=im_transformer, download=True)

    def augment(self, dataset):
        if self.binarized:
            dataset = BinarizedDataset(dataset)
        if self.gauss_noise:
            dataset = GaussNoiseDataset(dataset, std=self.noise_std)
        return dataset

    def affine_transform(self, dataset):
        if self.flattened:
            dataset = FlattenedDataset(dataset)
        return dataset

    def unpreprocess(self, v):
        v = v.view(len(v), 3, 32, 32)
        v.clamp_(0., 1.)
        return v
