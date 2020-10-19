import torch
import torch.autograd as autograd
import numpy as np
from .iwae import iwae
from .base import Criterion
from torch.nn.utils import clip_grad_norm_
from utils import grad_norm
import higher
from utils.config import config
from .base import _init_optim_bilevel


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
        if config.get("update", "gradient_clip", default=False):
            clip_grad_norm_(runner.q.parameters(), 0.5)
        opt_q.step()  # step
    sch_q.step()

    # update model
    runner.model.requires_grad_(True)
    runner.q.requires_grad_(True)
    backup_q_state_dict = runner.q.state_dict()
    with higher.innerloop_ctx(runner.q, opt_q) as (fq, diffopt_q):
        criterion.q = fq
        for i in range(config.get("update", "n_unroll")):
            inner_loss = criterion.inner_loss(v).mean()
            diffopt_q.step(inner_loss)
        criterion.loss_val = loss = criterion.loss(v).mean()
        opt_model.zero_grad()
        runner.model.requires_grad_(True)
        runner.q.requires_grad_(False)
        loss.backward()
        if config.get("update", "gradient_clip", default=False):
            clip_grad_norm_(runner.model.parameters(), 0.5)
        opt_model.step()
        sch_model.step()
    runner.q.load_state_dict(backup_q_state_dict)
    criterion.q = runner.q

    criterion.mid_vals["grad_model"] = grad_norm(runner.model)
    criterion.mid_vals["grad_q"] = grad_norm(runner.q)


def cond_fisher(v, model, q):
    v = v.clone().detach()
    h = q.implicit_net(v).squeeze(dim=0)
    log_p = -model.energy_net(v, h)
    log_q = q.log_q(h, v)
    log_w = log_p - log_q
    grad_log_w_h = autograd.grad(log_w.sum(), h, create_graph=True)[0]
    loss_h = (grad_log_w_h ** 2).flatten(1).sum(dim=-1) * 0.5
    return loss_h


class BiSM(Criterion):
    def __init__(self, models):
        super(BiSM, self).__init__(models)
        self.model = models["model"]
        self.q = models["q"]
        self.inner_loss_type = config.get("loss", "inner_loss_type")
        self.k = config.get("loss", "k", default=20)  # for iwae inner loss

    def _sample(self, v):
        raise NotImplementedError

    def inner_loss(self, v):
        v_ = self._sample(v)[-1]
        if self.inner_loss_type == "iwae":
            return -iwae(v_, self.model, self.q, usage="sm", k=self.k)
        elif self.inner_loss_type == "cond_fisher":
            return cond_fisher(v_, self.model, self.q)
        else:
            raise NotImplementedError

    def update(self, v, runner):
        _update(v, runner, self)

    def init_optim(self):
        _init_optim_bilevel(self)


class BiMDSM(BiSM):
    def __init__(self, models):
        super(BiMDSM, self).__init__(models)
        self.sigma0 = config.get("loss", "sigma0")
        self.sigma_begin = config.get("loss", "sigma_begin")
        self.sigma_end = config.get("loss", "sigma_end")
        self.dist = config.get("loss", "dist")
        self.inner_loss_div_sigmas = config.get("loss", "inner_loss_div_sigmas", default=True)

    def _sample_sigmas(self, v):
        if self.dist == "linear":
            sigmas = torch.linspace(self.sigma_begin, self.sigma_end, len(v))
        elif self.dist == "geometrical":
            sigmas = torch.logspace(np.log10(self.sigma_begin), np.log10(self.sigma_end), len(v))
        else:
            raise NotImplementedError
        return sigmas.to(v.device)

    def _sample(self, v):
        sigmas = self._sample_sigmas(v)
        sigmas4v = sigmas.view(len(v), *([1] * len(v.shape[1:])))
        eps = torch.randn_like(v).to(v.device)
        v_ = v + sigmas4v * eps
        return sigmas, eps, v_

    def loss(self, v):
        sigmas, eps, v_ = self._sample(v)
        sigmas4v = sigmas.view(len(v), *([1] * len(v.shape[1:])))
        h = self.q.implicit_net(v_.clone().detach()).squeeze(dim=0)
        v_.requires_grad_(True)
        log_p = -self.model.energy_net(v_, h)
        log_q = self.q.log_q(h, v_)
        log_w = log_p - log_q
        grad_log_w_v_ = autograd.grad(log_w.sum(), v_, create_graph=True)[0]
        return ((grad_log_w_v_ / sigmas4v + eps / self.sigma0 ** 2) ** 2).view(len(v), -1).sum(dim=-1) * 0.5

    def inner_loss(self, v):
        inner_loss_ = super(BiMDSM, self).inner_loss(v)
        if self.inner_loss_div_sigmas:
            sigmas = self._sample_sigmas(v)
            inner_loss_ = inner_loss_ / sigmas
        return inner_loss_


class BiDSM(BiSM):
    def __init__(self, models):
        super(BiDSM, self).__init__(models)
        self.noise_std = config.get("loss", "noise_std")

    def _sample(self, v):
        eps = torch.randn_like(v).to(v.device)
        v_ = v + self.noise_std * eps
        return eps, v_

    def loss(self, v):
        eps, v_ = self._sample(v)
        grad_log_q = -eps / self.noise_std
        h = self.q.implicit_net(v_.clone().detach()).squeeze(dim=0)
        v_.requires_grad_(True)
        log_p = -self.model.energy_net(v_, h)
        log_q = self.q.log_q(h, v_)
        log_w = log_p - log_q
        score = autograd.grad(log_w.sum(), v_, create_graph=True)[0]
        return ((score - grad_log_q) ** 2).view(len(v), -1).sum(dim=-1) * 0.5


class BiSSM(BiSM):
    def __init__(self, models):
        super(BiSSM, self).__init__(models)
        self.noise_type = config.get("loss", "noise_type", default='radermacher')

    def _sample(self, v):
        u = torch.randn_like(v).to(v.device)
        if self.noise_type == 'radermacher':  # better
            u = u.sign()
        elif self.noise_type == 'gaussian':
            pass
        else:
            raise NotImplementedError
        return u, v.clone().detach()

    def loss(self, v):
        u, v_ = self._sample(v)
        v_.requires_grad_(True)
        h = self.q.implicit_net(v_.clone().detach()).squeeze(dim=0)
        log_p = -self.model.energy_net(v_, h)
        log_q = self.q.log_q(h, v_)
        log_w = log_p - log_q
        score = autograd.grad(log_w.sum(), v_, create_graph=True)[0]
        loss1 = (score ** 2).sum(dim=-1) * 0.5
        grad_mul_u = score * u
        hvp = autograd.grad(grad_mul_u.sum(), v_, create_graph=True)[0]
        loss2 = (hvp * u).sum(dim=-1)
        return loss1 + loss2
