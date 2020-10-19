import torch
import torch.autograd as autograd
import numpy as np
from .base import Criterion
from utils.config import config


def ssm(v, model, noise_type='radermacher'):
    v.requires_grad_(True)

    u = torch.randn_like(v).to(v.device)
    if noise_type == 'radermacher':  # better
        u = u.sign()
    elif noise_type == 'gaussian':
        pass
    else:
        raise NotImplementedError

    log_p = -model.free_energy_net(v)
    score = autograd.grad(log_p.sum(), v, create_graph=True)[0]

    loss1 = (score ** 2).sum(dim=-1) * 0.5

    grad_mul_u = score * u
    hvp = autograd.grad(grad_mul_u.sum(), v, create_graph=True)[0]
    loss2 = (hvp * u).sum(dim=-1)

    return loss1 + loss2


class SSM(Criterion):
    def __init__(self, models):
        super(SSM, self).__init__(models)
        self.model = models["model"]
        self.noise_type = config.get("loss", "noise_type", default='radermacher')

    def loss(self, v):
        return ssm(v, self.model, self.noise_type)


def dsm(v, model, noise_std, eps=None):
    if eps is None:
        eps = torch.randn_like(v).to(v.device)
    grad_log_q = -eps / noise_std

    v_ = v + noise_std * eps
    v_.requires_grad_(True)

    log_p = -model.free_energy_net(v_)
    score = autograd.grad(log_p.sum(), v_, create_graph=True)[0]

    return ((score - grad_log_q) ** 2).sum(dim=-1) * 0.5


class DSM(Criterion):
    def __init__(self, models):
        super(DSM, self).__init__(models)
        self.model = models["model"]
        self.noise_std = config.get("loss", "noise_std")

    def loss(self, v):
        return dsm(v, self.model, self.noise_std)


def mdsm(v, model, sigma0, sigma_begin, sigma_end, dist, eps=None):
    if dist == "linear":
        used_sigmas = torch.linspace(sigma_begin, sigma_end, len(v))
    elif dist == "geometrical":
        used_sigmas = torch.logspace(np.log10(sigma_begin), np.log10(sigma_end), len(v))
    else:
        raise NotImplementedError

    used_sigmas = used_sigmas.view(len(v), *([1] * len(v.shape[1:]))).to(v.device)
    if eps is None:
        eps = torch.randn_like(v).to(v.device)

    v_ = v + used_sigmas * eps
    v_.requires_grad_(True)

    log_p = -model.free_energy_net(v_)
    score = autograd.grad(log_p.sum(), v_, create_graph=True)[0]

    return ((score / used_sigmas + eps / sigma0 ** 2) ** 2).view(len(v), -1).sum(dim=-1) * 0.5


class MDSM(Criterion):
    def __init__(self, models):
        super(MDSM, self).__init__(models)
        self.model = models["model"]
        self.sigma0 = config.get("loss", "sigma0")
        self.sigma_begin = config.get("loss", "sigma_begin")
        self.sigma_end = config.get("loss", "sigma_end")
        self.dist = config.get("loss", "dist")

    def loss(self, v):
        return mdsm(v, self.model, self.sigma0, self.sigma_begin, self.sigma_end, self.dist)
