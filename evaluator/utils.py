import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import os
from utils import get_device


def get_sample(label, dataset):
    for i in range(len(dataset)):
        v, y_ = dataset[i]
        if y_ == label:
            return v


def get_samples(label, n_samples, dataset):
    samples = []
    for i in range(len(dataset)):
        v, y_ = dataset[i]
        if y_ == label:
            samples.append(v.unsqueeze(dim=0))
            if len(samples) >= n_samples:
                break
    return torch.cat(samples, dim=0)


def extract_labelled_feature(q, dataset, batch_size=100):
    features = []
    ys = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for x, y in loader:
        x = x.to(get_device(q))
        feature, _ = q.moments(x)
        feature = feature.detach().cpu().numpy()
        features.append(feature)
        ys.append(y)
    features = np.concatenate(features, axis=0)
    ys = np.concatenate(ys, axis=0)
    return features, ys


def linear_interpolate(a, b, steps):
    a_shape = a.shape
    a = a.detach().cpu().view(-1)
    b = b.detach().cpu().view(-1)
    res = []
    for aa, bb in zip(a, b):
        res.append(torch.linspace(aa, bb, steps=steps).unsqueeze(dim=1))
    res = torch.cat(res, dim=1)
    res = res.view(len(res), *a_shape)
    return res


def rect_interpolate(a, b, c, steps):
    a = a.detach().cpu()
    b = b.detach().cpu()
    c = c.detach().cpu()
    ab = linear_interpolate(a, b, steps) - a
    ac = linear_interpolate(a, c, steps) - a
    res = []
    for st in ac:
        res.append(ab + st)
    res = torch.cat(res, dim=0) + a
    return res


def dimension_reduction(arr):
    model = TSNE(n_components=2, random_state=0)
    features = model.fit_transform(arr)
    return features


def classify_features(tr_features, tr_ys, te_features, te_ys, classifier):
    if classifier == "kn":
        clf = KNeighborsClassifier(n_neighbors=30)
    elif classifier == "svm":
        clf = SVC()
    elif classifier == "lsvm":
        clf = LinearSVC()
    elif classifier == "logistic":
        clf = LogisticRegression()
    else:
        raise NotImplementedError
    clf.fit(tr_features, tr_ys)
    pred = clf.predict(te_features)
    acc = (pred == te_ys).mean()
    return clf, pred, acc


def sample_from_ckpt(runner, ckpt, n_samples, sample_path):
    ckpt_path = os.path.join(runner.model_root, ckpt)
    runner.load_model(path=ckpt_path)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    print("sample to {}".format(sample_path))
    runner.evaluator.sample2dir(sample_path, n_samples)


def valid_sample(runner):
    runner.to(runner.device)
    valid_path = os.path.join(runner.workspace_root, "valid_samples")
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    ckpts = list(filter(lambda x: x.split(".")[-1].isdigit(), os.listdir(runner.model_root)))
    ckpts = sorted(ckpts, key=lambda x: int(x.split(".")[-1]))[-20:]
    for ckpt in ckpts:
        sample_path = os.path.join(valid_path, ckpt)
        sample_from_ckpt(runner, ckpt, 1000, sample_path)


def evaluate(dataset, runner, fn, **kwargs):
    runner.eval()
    batch_size = kwargs.get("batch_size", runner.batch_size)
    total_loss = 0.
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for v in dataloader:
        v = v.to(runner.device)
        loss = fn(v)
        total_loss += loss.sum().detach()
    mean_loss = total_loss / len(dataset)
    return mean_loss
