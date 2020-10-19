import os
import argparse


def get_device(model):
    return list(model.parameters())[0].device


def my_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def get_free_gpus(black_list=()):
    print("get_free_gpus")
    s = os.popen("gpustat")
    gpus = []
    for idx, line in enumerate(s):
        if idx == 0:
            continue
        line = line.strip()
        gpu = int(line[1])
        if line[-1] == "|" and gpu not in black_list:
            gpus.append(gpu)
    print("gpus:", gpus)
    return gpus


def _assign_gpu(n_tasks, gpus):
    res = []
    for item in amortize(n_tasks, len(gpus)):
        res += gpus[:item]
    return res


def assign_gpu(n_tasks, black_list=()):
    gpus = get_free_gpus(black_list)
    return _assign_gpu(n_tasks, gpus)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def grad_norm(model):
    s = 0.
    for p in model.parameters():
        if p.grad is not None:
            s += (p.grad.data ** 2).sum().item()
    return s ** 0.5


def d2(lst1, lst2):
    d = 0.
    for a, b in zip(lst1, lst2):
        d += ((a - b) ** 2).sum().item()
    return d ** 0.5


def l2(lst):
    s = 0.
    for a in lst:
        s += (a ** 2).sum().item()
    return s ** 0.5
