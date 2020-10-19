import os
from utils import Tuner

template = os.path.join("ebm_resnet", "mnist", "ebm_resnet_mnist_mdsm.yml")


states = [
    {"model/scalar_net": "LinearAFSquare"},
]


if __name__ == "__main__":
    tuner = Tuner(template)
    tuner.task_assign(states)
