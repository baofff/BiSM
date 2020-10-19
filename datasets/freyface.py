from .dataset_factory import DatasetFactory
from .utils import *
import os
import urllib.request as request
import scipy.io as sio
import numpy as np
import torch
from utils.config import config


class FreyFace(DatasetFactory):
    r""" train: 1,400
         val:   300
         test:  265
         shape: 28 * 20

         train mean: 0.6049
         train biased std: 0.1763
    """

    def __init__(self):
        super(FreyFace, self).__init__()
        self.binarized = config.get("data", "binarized", default=False)
        self.gauss_noise = config.get("data", "gauss_noise", default=True)
        self.noise_std = config.get("data", "noise_std", default=0.01)
        self.preprocess = config.get("data", "preprocess", default=None)
        self.flattened = config.get("data", "flattened", default=True)

        if self.binarized:
            assert not self.gauss_noise
            assert self.preprocess is None
        assert self.preprocess == "standardize" or self.preprocess == "subtract_mean" or self.preprocess is None

        self.data_root = os.path.join("workspace", "datasets", "freyface")
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        self.data_path = os.path.join(self.data_root, "frey_rawface.mat")
        if not os.path.exists(self.data_path):
            request.urlretrieve("https://cs.nyu.edu/~roweis/data/frey_rawface.mat", self.data_path)

        data = sio.loadmat(self.data_path)['ff'].transpose().astype(np.float32) / 255.
        data = torch.tensor(data).view(-1, 1, 28, 20)
        train_data = data[: 1400]
        val_data = data[1400: 1700]
        test_data = data[1700:]

        # Get datasets
        self.train = QuickDataset(train_data)
        self.val = QuickDataset(val_data)
        self.test = QuickDataset(test_data)
        self.uniform = QuickDataset(torch.rand_like(train_data))

        # Calculate the train mean and std
        if self.preprocess == "standardize":
            standardize_mode = config.get("data", "standardize_mode", default="pixel")
            if standardize_mode == "pixel":
                self.train_mean = 0.6049
                self.train_std = 0.1763
            elif standardize_mode == "vector":
                self.train_mean = train_data.mean(dim=0)
                self.train_std = train_data.std(dim=0, unbiased=False) + 1e-3
            else:
                raise NotImplementedError

    def augment(self, dataset):
        if self.binarized:
            dataset = BinarizedDataset(dataset)
        if self.gauss_noise:
            dataset = GaussNoiseDataset(dataset, std=self.noise_std)
        return dataset

    def affine_transform(self, dataset):
        if self.preprocess == "standardize":
            dataset = StandardizedDataset(dataset, mean=self.train_mean, std=self.train_std)
        elif self.preprocess == "subtract_mean":
            dataset = TranslatedDataset(dataset, delta=-self.train_mean)
        if self.flattened:
            dataset = FlattenedDataset(dataset)
        return dataset

    def unpreprocess(self, v):
        v = v.view(len(v), 1, 28, 20)
        if self.preprocess == "standardize":
            v *= self.train_std.to(v.device)
            v += self.train_mean.to(v.device)
        if self.preprocess == "subtract_mean":
            v += self.train_mean.to(v.device)
        v.clamp_(0., 1.)
        return v
