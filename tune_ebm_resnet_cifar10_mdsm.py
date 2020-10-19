import os
from utils import Tuner

template = os.path.join("ebm_resnet", "cifar10", "ebm_resnet_cifar10_mdsm.yml")

mode = "train"
seed = 1234

common = {"others/seed": seed, "training/n_ckpts": 60}

if mode == "train":
    states = [
        {"model/scalar_net": "LinearAFSquare", **common},
    ]
elif mode == "valid" or mode == "test":
    states = [
        {"model/scalar_net": "LinearAFSquare", "training/batch_size": 1000, **common},
    ]
elif mode == "test_ckpt":
    states = [
        {"model/scalar_net": "LinearAFSquare", "others/tested_ckpt": "ebm_resnet_cifar10_mdsm_0.ckpt.pth.???",
         "training/batch_size": 1000, **common},
    ]
else:
    raise NotImplementedError


if __name__ == "__main__":
    tuner = Tuner(template)
    tuner.task_assign(states, mode=mode)
