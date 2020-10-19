import torch.nn as nn
import numpy as np
from utils.func import mylogsumexp
from utils.config import config


class LatentModel(nn.Module):
    def __init__(self):
        super(LatentModel, self).__init__()
        self.device = config.device
        self.h_dim = config.h_dim
        self.v_dim = config.v_dim
        self.v_shape = config.v_shape

    def cmean_h(self, v):  # E[h|v]
        raise NotImplementedError

    def cmean_v(self, h):  # E[v|h]
        # h: (n_particles *) batch_size * h_dim
        raise NotImplementedError

    def csample_h(self, v):  # sample from p(h|v)
        raise NotImplementedError

    def csample_v(self, h, **kwargs):  # sample from p(v|h)
        # h: (n_particles *) batch_size * h_dim
        raise NotImplementedError

    def sample(self, n_samples, **kwargs):  # sample from p(v)
        raise NotImplementedError

    def sample_h(self, n_samples):  # sample from p(h)
        raise NotImplementedError

    def energy_net(self, v, h):  # E(v, h)
        # h: (n_particles *) batch_size * h_dim
        raise NotImplementedError

    def free_energy_net(self, v):  # F(v)
        raise NotImplementedError

    def free_energy_net_affine(self, v, a, b):
        # the free energy of p_affine (v), where p_affine is determined by y = a x + b, x ~ p_model (x)
        return self.free_energy_net((v - b) / a)

    def log_joint(self, v, h):  # log p(v, h)
        # h: (n_particles *) batch_size * h_dim
        raise NotImplementedError

    def log_cpv(self, v, h):  # log p(v|h)
        # h: (n_particles *) batch_size * h_dim
        raise NotImplementedError

    def log_likelihood(self, v, **kwargs):  # log p(v)
        # A simple estimator of log-likelihood when h is low dimensional
        n_particles = 100
        h = self.sample_h(n_particles * len(v))
        h = h.to(v.device).view(n_particles, len(v), -1)
        log_p = self.log_cpv(v, h)
        log_likelihood = mylogsumexp(log_p, dim=0) - np.log(n_particles)
        return log_likelihood

    def reconstruct(self, v, **kwargs):
        random = kwargs.get("random", False)
        h = self.csample_h(v)
        v_reconstruct = self.csample_v(h, random=random)
        return v_reconstruct

    def reconstruction_loss(self, v):
        v_reconstruct = self.reconstruct(v)
        return ((v - v_reconstruct) ** 2).view(len(v), -1).sum(dim=1)

    def bayes_denoise(self, v_noise):
        raise NotImplementedError
