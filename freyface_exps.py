import os
from utils import Tuner

mode = "train"

common = {}

states = [
    {**common}
]


states_bism = [
    {"update/n_unroll": 0, **common}, {"update/n_unroll": 1, **common},
    {"update/n_unroll": 5, **common}, {"update/n_unroll": 10, **common},
    {"update/n_inner_loops": 0, **common}, {"update/n_inner_loops": 1, **common},
    {"update/n_inner_loops": 5, **common}, {"update/n_inner_loops": 10, **common},
]

if __name__ == "__main__":
    template = os.path.join("grbm", "freyface", "grbm_freyface_bidsm.yml")
    tuner = Tuner(template)
    tuner.task_assign(states_bism, mode=mode)
