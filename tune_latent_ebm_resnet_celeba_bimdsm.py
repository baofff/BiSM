import os
from utils import Tuner

template = os.path.join("latent_ebm_resnet", "celeba", "latent_ebm_resnet_celeba_bimdsm.yml")

mode = "train"

states = [
    {"model/h_dim": 20, "model/scalar_net": "LinearAFSquare"},
    {"model/h_dim": 50, "model/scalar_net": "LinearAFSquare"},
    {"model/h_dim": 100, "model/scalar_net": "LinearAFSquare"},
]


if __name__ == "__main__":
    tuner = Tuner(template)
    tuner.task_assign(states, mode=mode)
