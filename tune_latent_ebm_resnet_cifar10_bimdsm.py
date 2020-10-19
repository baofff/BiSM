import os
from utils import Tuner

template = os.path.join("latent_ebm_resnet", "cifar10", "latent_ebm_resnet_cifar10_bimdsm.yml")

mode = "train"
seed = 1234

common = {"others/seed": seed, "training/n_ckpts": 60}

if mode == "train":
    states = [
        {"model/h_dim": 20, **common},
        {"model/h_dim": 50, **common},
        {"model/h_dim": 100, **common},
    ]
elif mode == "valid" or mode == "test":
    states = [
        {"model/h_dim": 20, "training/batch_size": 1000, **common},
        {"model/h_dim": 50, "training/batch_size": 1000, **common},
        {"model/h_dim": 100, "training/batch_size": 1000, **common},
    ]
elif mode == "test_ckpt":
    states = [
        {"model/h_dim": 20, "others/tested_ckpt": "latent_ebm_resnet_cifar10_bimdsm_0.ckpt.pth.???",
         "training/batch_size": 1000, **common},
        {"model/h_dim": 50, "others/tested_ckpt": "latent_ebm_resnet_cifar10_bimdsm_1.ckpt.pth.???",
         "training/batch_size": 1000, **common},
        {"model/h_dim": 100, "others/tested_ckpt": "latent_ebm_resnet_cifar10_bimdsm_2.ckpt.pth.???",
         "training/batch_size": 1000, **common},
    ]
else:
    raise NotImplementedError


if __name__ == "__main__":
    tuner = Tuner(template)
    tuner.task_assign(states, mode=mode)
