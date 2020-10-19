import os
from utils import Tuner

template = os.path.join("latent_ebm_resnet", "mnist", "latent_ebm_resnet_mnist_bimdsm.yml")

mode = "train"

states = [
    {"model/h_dim": 20},
    {"model/h_dim": 50},
    {"model/h_dim": 100},
]


if __name__ == "__main__":
    tuner = Tuner(template)
    tuner.task_assign(states, mode=mode)
