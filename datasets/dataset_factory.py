from .utils import is_labelled, UnlabeledDataset
from torch.utils.data import ConcatDataset


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.val = None
        self.test = None
        self.uniform = None

    def allow_labelled(self):
        return is_labelled(self.train)

    def get_data(self, dataset, augment, labelled):
        assert not (not is_labelled(dataset) and labelled)
        if is_labelled(dataset) and not labelled:
            dataset = UnlabeledDataset(dataset)
        if augment:
            dataset = self.augment(dataset)
        return self.affine_transform(dataset)

    def get_train_data(self, labelled=False):
        return self.get_data(self.train, augment=True, labelled=labelled)

    def get_val_data(self, labelled=False):
        return self.get_data(self.val, augment=True, labelled=labelled)

    def get_train_val_data(self, labelled=False):
        train_val = ConcatDataset([self.train, self.val])
        return self.get_data(train_val, augment=True, labelled=labelled)

    def get_test_data(self, labelled=False):
        return self.get_data(self.test, augment=True, labelled=labelled)

    def augment(self, dataset):
        return dataset

    def affine_transform(self, dataset):
        return dataset

    def get_uniform_data(self):  # for initialize gibbs sampling
        return self.get_data(self.uniform, augment=False, labelled=False)
