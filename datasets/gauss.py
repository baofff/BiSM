from .dataset_factory import DatasetFactory
from .utils import *
from torch.distributions import MultivariateNormal


class GaussDataset(Dataset):
    def __init__(self, n, mean, cov):
        self.n = n
        self.dist = MultivariateNormal(loc=mean, covariance_matrix=cov)

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.dist.sample()


class Gauss(DatasetFactory):
    r""" train: 50,000
         val:   10,000
         test:  10,000
    """

    def __init__(self, mean, cov):
        super(Gauss, self).__init__()
        self.train = GaussDataset(50000, mean, cov)
        self.val = GaussDataset(10000, mean, cov)
        self.test = GaussDataset(10000, mean, cov)

    def get_train_data(self, labelled=False):
        return self.train

    def get_val_data(self, labelled=False):
        return self.val

    def get_test_data(self, labelled=False):
        return self.val
