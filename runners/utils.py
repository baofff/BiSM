from torch.utils.data import DataLoader


def infinite_loader(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    while True:
        for data in loader:
            yield data


