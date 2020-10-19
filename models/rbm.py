# Use RBM to debug CD algorithm
import torch.nn as nn
import torch
import math
from .latent_model import LatentModel
from .utils import log_p_normal
import torch.nn.functional as F
from utils.config import config
import numpy as np
from utils.func import mylogsumexp
from tqdm import tqdm


class _GRBM(LatentModel):
    def __init__(self):
        super(_GRBM, self).__init__()
        self.device = config.device
        self.log_std = None
        self.b_h = None
        self.b_v = None
        self.W = None

    def cmean_h(self, v):  # E(h|v)
        return (self.b_h + v @ self.W).sigmoid()

    def cmean_v(self, h):  # E(v|h)
        # h: (n_particles *) batch_size * h_dim
        return self.b_v + (h @ self.W.t()) * (self.log_std * 2).exp()

    def csample_h(self, v):
        return self.cmean_h(v).bernoulli()

    def csample_v(self, h, **kwargs):
        # h: (n_particles *) batch_size * h_dim
        eps = torch.randn(*h.shape[:-1], self.v_dim).to(h.device)
        mean = self.cmean_v(h)
        v = mean + self.log_std.exp() * eps
        return v

    def log_cpv(self, v, h):
        # h: (n_particles *) batch_size * h_dim
        mean = self.cmean_v(h)
        return log_p_normal(v, mean, self.log_std)

    def sample_h(self, n_samples, **kwargs):
        v = kwargs.get("v0", torch.rand(n_samples, self.v_dim).to(self.device))
        h = self.cmean_h(v).detach()
        for i in range(100):
            v = self.csample_v(h).detach()
            h = self.csample_h(v).detach()
        return h

    def sample(self, n_samples, **kwargs):
        random = kwargs.get("random", False)
        record_gibbs = kwargs.get("record_gibbs", False)
        v0 = v = kwargs.get("v0", torch.rand(n_samples, self.v_dim).to(self.device))
        h = self.cmean_h(v).detach()

        if record_gibbs:
            rec_steps = [1, 20, 500, 1000]
            samples = []
            for i in range(rec_steps[-1] + 1):
                if i in rec_steps:
                    samples.append((i, self.csample_v(h).detach() if random else self.cmean_v(h).detach()))
                v = self.csample_v(h).detach()
                h = self.csample_h(v).detach()
            if kwargs.get("ret_v0", False):
                return samples, v0
            else:
                return samples
        else:
            for i in range(1000):
                v = self.csample_v(h).detach()
                h = self.csample_h(v).detach()
            v = self.csample_v(h).detach() if random else self.cmean_v(h).detach()
            if kwargs.get("ret_v0", False):
                return v, v0
            else:
                return v

    def energy_net(self, v, h):
        # h: (n_particles *) batch_size * h_dim
        v_part = ((v - self.b_v) ** 2 * (-2. * self.log_std).exp()).sum(dim=-1) * 0.5
        h_part = - h @ self.b_h
        vh_part = - ((v @ self.W) * h).sum(dim=-1)  # n_particles * batch_size
        return v_part + h_part + vh_part

    def free_energy_net(self, v):
        a = ((v - self.b_v) ** 2 * (-2. * self.log_std).exp()).sum(dim=-1) * 0.5
        b = F.softplus(self.b_h + v @ self.W).sum(dim=-1)
        return a - b

    def log_partition(self):
        raise NotImplementedError


class _BRGRBM(_GRBM):
    r""" for ais
         p(v) = N(v; b_v, std^2)
    """
    def __init__(self, log_std, b_v):
        super(_BRGRBM, self).__init__()
        self.log_std = log_std.data.clone()
        self.b_v = b_v.data.clone()
        self.b_h = torch.zeros(self.h_dim).to(self.device)
        self.W = torch.zeros(self.v_dim, self.h_dim).to(self.device)
        self.log_partition_ = self.log_std.sum() + 0.5 * np.log(2 * np.pi) * self.v_dim + np.log(2.) * self.h_dim

    def log_partition(self):
        return self.log_partition_

    def sample(self, n_samples, **kwargs):
        eps = torch.randn(n_samples, self.v_dim).to(self.device)
        return self.b_v + eps * self.log_std.exp()


class _MIXGRBM(_GRBM):  # for ais
    def __init__(self, log_std, b_v, b_h, W):
        super(_MIXGRBM, self).__init__()
        self.log_std = log_std.data.clone()
        self.b_v = b_v.data.clone()
        self.b_h = None
        self.W = None
        self.b_h_target = b_h.data.clone()
        self.W_target = W.data.clone()

    def set_status(self, weight):
        self.b_h = weight * self.b_h_target
        self.W = weight * self.W_target

    def vhv(self, v):
        return self.csample_v(self.csample_h(v))


class GRBM(_GRBM):
    def __init__(self):
        super(GRBM, self).__init__()
        self.fix_std = config.get("model", "fix_std", default=False)
        if self.fix_std:
            std = config.get("model", "std")
            self.log_std = torch.ones(self.v_dim).to(self.device) * np.log(std)
        else:
            self.log_std = nn.Parameter(torch.zeros(self.v_dim), requires_grad=True)
        self.b_h = nn.Parameter(torch.zeros(self.h_dim), requires_grad=True)
        self.b_v = nn.Parameter(torch.zeros(self.v_dim), requires_grad=True)
        self.W = nn.Parameter(torch.zeros(self.v_dim, self.h_dim), requires_grad=True)
        self.init_parameters()
        self.log_partition_ = None

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b_h, -bound, bound)
        nn.init.uniform_(self.b_v, -bound, bound)

    def update_log_partition(self):  # ais
        m0 = _BRGRBM(self.log_std, self.b_v)
        weights = torch.linspace(0., 1., 2000)
        n_samples = 2000
        v = m0.sample(n_samples)
        mk = _MIXGRBM(self.log_std, self.b_v, self.b_h, self.W)
        log_w = m0.free_energy_net(v)
        for weight in tqdm(weights[:-1], desc="ais"):
            mk.set_status(weight)
            log_w -= mk.free_energy_net(v)
            v = mk.vhv(v)
            log_w += mk.free_energy_net(v)
        mk.set_status(weights[-1])
        log_w -= mk.free_energy_net(v)
        self.log_partition_ = mylogsumexp(log_w, dim=0) - np.log(n_samples) + m0.log_partition()

    def log_partition(self):
        return self.log_partition_

    def log_likelihood(self, v, **kwargs):
        if kwargs.get("ais", False):
            return -self.free_energy_net(v) - self.log_partition()
        else:  # brute force
            return super(GRBM, self).log_likelihood(v)
