from torchvision import datasets
import torchvision.transforms as transforms
from .dataset_factory import DatasetFactory
from .utils import *
import os
from utils.config import config


class CelebA(DatasetFactory):
    r""" train: 162,770
         val:   19,867
         test:  19,962
         shape: 3 * 32 * 32
    """

    def __init__(self):
        super(CelebA, self).__init__()
        self.binarized = config.get("data", "binarized", default=False)
        self.gauss_noise = config.get("data", "gauss_noise", default=False)
        self.noise_std = config.get("data", "noise_std", default=0.01)
        self.flattened = config.get("data", "flattened", default=False)

        if self.binarized:
            assert not self.gauss_noise

        self.data_path = os.path.join("workspace", "datasets", "celeba")

        # Get datasets
        im_transformer = transforms.Compose([transforms.CenterCrop(140), transforms.Resize(32),
                                             transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])
        self.train = datasets.CelebA(self.data_path, split="train", target_type=[],
                                     transform=im_transformer, download=True)
        self.val = datasets.CelebA(self.data_path, split="valid", target_type=[],
                                   transform=im_transformer, download=True)
        self.test = datasets.CelebA(self.data_path, split="test", target_type=[],
                                    transform=im_transformer, download=True)

        self.train = UnlabeledDataset(self.train)
        self.test = UnlabeledDataset(self.test)
        self.val = UnlabeledDataset(self.val)

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
