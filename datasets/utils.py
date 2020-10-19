import random
import torch
from torch.utils.data import Dataset, TensorDataset
import torch.nn.functional as F


def my_filter(dataset, digits):
    data = []
    labels = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y in digits:
            data.append(x.unsqueeze(dim=0))
            labels.append(y)
    data = torch.cat(data, dim=0)
    labels = torch.tensor(labels)
    return TensorDataset(data, labels)


def is_labelled(dataset):
    labelled = False
    if isinstance(dataset[0], tuple) and len(dataset[0]) == 2:
        labelled = True
    return labelled


class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return self.transform(x), y
        else:
            x = self.dataset[item]
            return self.transform(x)


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return x


class FlattenedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return x.view(-1), y
        else:
            x = self.dataset[item]
            return x.view(-1)


class BinarizedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return x.bernoulli(), y
        else:
            x = self.dataset[item]
            return x.bernoulli()


class TranslatedDataset(Dataset):
    def __init__(self, dataset, delta):
        self.dataset = dataset
        self.delta = delta
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return x + self.delta, y
        else:
            x = self.dataset[item]
            return x + self.delta


class StandardizedDataset(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return (x - self.mean) / self.std, y
        else:
            x = self.dataset[item]
            return (x - self.mean) / self.std


class PaddedDataset(Dataset):
    def __init__(self, dataset, pad):
        self.dataset = dataset
        self.pad = [pad] * 4
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return F.pad(x, self.pad), y
        else:
            x = self.dataset[item]
            return F.pad(x, self.pad)


class GaussNoiseDataset(Dataset):
    def __init__(self, dataset, std):
        self.dataset = dataset
        self.std = std
        self.labelled = is_labelled(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.labelled:
            x, y = self.dataset[item]
            return x + self.std * torch.rand_like(x).to(x.device), y
        else:
            x = self.dataset[item]
            return x + self.std * torch.rand_like(x).to(x.device)


class QuickDataset(Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, item):
        return self.array[item]


class InfiniteRandomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 2147483647

    def __getitem__(self, item):
        return random.choice(self.dataset)
