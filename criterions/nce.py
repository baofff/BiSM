from .base import Criterion
import evaluator.utils
from utils.config import config
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import utils.func


def _init_optim_nce(criterion):
    optimizer = config.get("optim", "optimizer")
    scheduler = config.get("optim", "scheduler")
    lr = config.get("optim", "lr")
    weight_decay = config.get("optim", "weight_decay", default=0.)
    params = []
    for name, model in criterion.models.items():
        params += model.parameters()
    params += [criterion.c]  # nce parameter
    # init optimizers
    if optimizer == "Adam":
        betas = tuple(config.get("optim", "betas", default=(0.9, 0.95)))
        criterion.opts["all"] = optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer == "RMSProp":
        criterion.opts["all"] = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGD":
        momentum = config.get("optim", "momentum", default=0.01)
        criterion.opts["all"] = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    # init schedulers
    n_its = config.get("training", "n_its")
    if scheduler == "step":
        criterion.schs["all"] = optim.lr_scheduler.StepLR(criterion.opts["all"], n_its // 6)
    elif scheduler == "cosine":
        criterion.schs["all"] = optim.lr_scheduler.CosineAnnealingLR(criterion.opts["all"], n_its, eta_min=1e-6)
    elif scheduler == "const":
        criterion.schs["all"] = optim.lr_scheduler.StepLR(criterion.opts["all"], n_its)
    else:
        raise NotImplementedError


class NCE(Criterion):
    def __init__(self, models):
        self.model = models["model"]
        self.nu = config.get("loss", "nu", default=1.)
        self.c = nn.Parameter(torch.scalar_tensor(0., device=config.device))
        super(NCE, self).__init__(models)

    def sample_from_noise(self, v):
        return torch.randn_like(v).to(v.device)

    def log_p_noise(self, y):
        return -utils.func.sos(y) * 0.5 - y.size(1) * np.log(2 * np.pi) * 0.5

    def loss(self, v):
        y = self.sample_from_noise(v)
        log_h_v = F.logsigmoid(-self.model.free_energy_net(v) - self.c - np.log(self.nu) - self.log_p_noise(v))
        log_mh_y = F.logsigmoid(self.model.free_energy_net(y) + self.c + np.log(self.nu) + self.log_p_noise(y))
        return -(log_h_v + self.nu * log_mh_y)

    def init_optim(self):
        return _init_optim_nce(self)
