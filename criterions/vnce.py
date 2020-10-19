from .base import Criterion
import evaluator.utils
from utils.config import config
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import utils.func
from .iwae import iwae


def _init_optim_vnce(criterion):
    optimizer = config.get("optim", "optimizer")
    scheduler = config.get("optim", "scheduler")
    lr = config.get("optim", "lr")
    lr_q = config.get("optim", "lr_q", default=lr)
    weight_decay = config.get("optim", "weight_decay", default=0.)
    model, q = criterion.models["model"], criterion.models["q"]
    model_params = list(model.parameters()) + [criterion.c]
    # init optimizers
    if optimizer == "Adam":
        betas = tuple(config.get("optim", "betas", default=(0.9, 0.95)))
        criterion.opts["model"] = optim.Adam(model_params, lr=lr, weight_decay=weight_decay, betas=betas)
        criterion.opts["q"] = optim.Adam(q.parameters(), lr=lr_q, weight_decay=weight_decay, betas=betas)
    elif optimizer == "RMSProp":
        criterion.opts["model"] = optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
        criterion.opts["q"] = optim.RMSprop(q.parameters(), lr=lr_q, weight_decay=weight_decay)
    elif optimizer == "SGD":
        momentum = config.get("optim", "momentum", default=0.01)
        criterion.opts["model"] = optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion.opts["q"] = optim.SGD(q.parameters(), lr=lr_q, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    # init schedulers
    n_its = config.get("training", "n_its")
    if scheduler == "step":
        criterion.schs["model"] = optim.lr_scheduler.StepLR(criterion.opts["model"], n_its // 6)
        criterion.schs["q"] = optim.lr_scheduler.StepLR(criterion.opts["q"], n_its // 6)
    elif scheduler == "cosine":
        criterion.schs["model"] = optim.lr_scheduler.CosineAnnealingLR(criterion.opts["model"], n_its, eta_min=1e-6)
        criterion.schs["q"] = optim.lr_scheduler.CosineAnnealingLR(criterion.opts["q"], n_its, eta_min=1e-6)
    elif scheduler == "const":
        criterion.schs["model"] = optim.lr_scheduler.StepLR(criterion.opts["model"], n_its)
        criterion.schs["q"] = optim.lr_scheduler.StepLR(criterion.opts["q"], n_its)
    else:
        raise NotImplementedError


def _update(v, runner, criterion):
    opt_model, opt_q = criterion.opts["model"], criterion.opts["q"]
    sch_model, sch_q = criterion.schs["model"], criterion.schs["q"]

    # update q
    runner.model.requires_grad_(False)
    runner.q.requires_grad_(True)
    for i in range(config.get("update", "n_inner_loops")):
        opt_q.zero_grad()
        criterion.mid_vals["inner_loss"] = inner_loss = criterion.inner_loss(v).mean()
        inner_loss.backward()  # backward
        opt_q.step()  # step
    sch_q.step()

    # update model
    runner.model.requires_grad_(True)
    runner.q.requires_grad_(False)
    criterion.loss_val = loss = criterion.loss(v).mean()
    opt_model.zero_grad()
    loss.backward()
    opt_model.step()
    sch_model.step()


class VNCE(Criterion):
    def __init__(self, models):
        self.model = models["model"]
        self.q = models["q"]
        self.nu = config.get("loss", "nu", default=1.)
        self.n_particles = config.get("loss", "n_particles", default=5)
        self.c = nn.Parameter(torch.scalar_tensor(0., device=config.device))
        self.k = config.get("loss", "k", default=20)  # for iwae inner loss
        self.use_true_post = config.get("debug", "use_true_post", default=False)
        super(VNCE, self).__init__(models)

    def sample_from_noise(self, v):
        return torch.randn_like(v).to(v.device)

    def log_p_noise(self, y):
        return -utils.func.sos(y) * 0.5 - y.size(1) * np.log(2 * np.pi) * 0.5

    def implicit_net_log_q(self, v, n_particles=1):
        with torch.no_grad():
            if self.use_true_post:
                h = torch.stack([self.model.csample_h(v) for _ in range(n_particles)])
                log_q = self.model.log_cph(h, v)
            else:
                h, log_q = self.q.implicit_net_log_q(v, n_particles)
        return h, log_q

    def loss(self, v):
        hv, log_qv = self.implicit_net_log_q(v)
        hv = hv.squeeze(dim=0)
        log_qv = log_qv.squeeze(dim=0)
        y = self.sample_from_noise(v)
        hy, log_qy = self.implicit_net_log_q(y, n_particles=self.n_particles)
        log_py = -self.model.energy_net(y, hy)
        log_wy = log_py - log_qy
        iwae = utils.func.mylogsumexp(log_wy, dim=0) - np.log(self.n_particles)

        log_h_v = F.logsigmoid(-self.model.energy_net(v, hv) - log_qv - self.c - np.log(self.nu) - self.log_p_noise(v))
        log_mh_y = F.logsigmoid(-iwae + self.c + np.log(self.nu) + self.log_p_noise(y))
        return -(log_h_v + self.nu * log_mh_y)

    def init_optim(self):
        return _init_optim_vnce(self)

    def update(self, v, runner):
        _update(v, runner, self)

    def inner_loss(self, v):
        return -iwae(v, self.model, self.q, usage="sm", k=self.k)
