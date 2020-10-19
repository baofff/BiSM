import os
from utils import Tuner

mode = "train"

seed = 1234

states = [
    {"others/seed": seed}
]

states_cd = [
    {"loss/k": 1, "others/seed": seed},
    {"loss/k": 5, "others/seed": seed},
    {"loss/k": 10, "others/seed": seed},
]

states_bism = [
    {"update/n_unroll": 0, "others/seed": seed},
    {"update/n_unroll": 5, "others/seed": seed},
    {"update/n_unroll": 10, "others/seed": seed},
]


if __name__ == "__main__":
    # cd
    template = os.path.join("grbm", "toy", "grbm_toy_cd.yml")
    tuner = Tuner(template)
    tuner.task_assign(states_cd, devices=[0, 0, 0], mode=mode)

    # pcd
    template = os.path.join("grbm", "toy", "grbm_toy_pcd.yml")
    tuner = Tuner(template)
    tuner.task_assign(states_cd, devices=[0], mode=mode)

    # ssm
    template = os.path.join("grbm", "toy", "grbm_toy_ssm.yml")
    tuner = Tuner(template)
    tuner.task_assign(states, devices=[0], mode=mode)

    # bissm
    template = os.path.join("grbm", "toy", "grbm_toy_bissm.yml")
    tuner = Tuner(template)
    tuner.task_assign(states_bism, devices=[0, 1, 2], mode=mode)

    # dsm
    template = os.path.join("grbm", "toy", "grbm_toy_dsm.yml")
    tuner = Tuner(template)
    tuner.task_assign(states, devices=[0], mode=mode)

    # bidsm
    template = os.path.join("grbm", "toy", "grbm_toy_bidsm.yml")
    tuner = Tuner(template)
    tuner.task_assign(states_bism, devices=[0, 3, 3], mode=mode)

    # nce
    template = os.path.join("grbm", "toy", "grbm_toy_nce.yml")
    tuner = Tuner(template)
    tuner.task_assign(states, devices=[4], mode=mode)

    # vnce
    template = os.path.join("grbm", "toy", "grbm_toy_vnce.yml")
    tuner = Tuner(template)
    tuner.task_assign(states, devices=[4], mode=mode)
