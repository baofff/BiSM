from .utils import *
from .latent_model import LatentModel
import numpy as np
from utils.func import mylogsumexp, duplicate
from utils.config import config
from .modules import ResidualNet6, ResidualNet9, LinearAFSquare, Quadratic, AFQuadratic


class EBM(LatentModel):
    def __init__(self):
        super(EBM, self).__init__()
        self.device = config.device
        # for sampling
        n_sampling = 2000
        Tmax, Tmin = 100, 1
        Ts = Tmax * np.exp(-np.linspace(0, n_sampling - 1, n_sampling) * (np.log(Tmax / Tmin) / n_sampling))
        Ts = np.concatenate((Tmax * np.ones((500,)), Ts), axis=0)
        self.Ts = np.concatenate((Ts, Tmin * np.linspace(1, 0, 200)), axis=0)
        self.sigma = 0.02
        self.sample_every = 100
        self.denoise = True
        self.sigma0 = 0.1

    def sample_joint(self, n_samples):
        mode = self.training
        self.eval()
        init_v = 0.5 + torch.randn(n_samples, *self.v_shape).to(self.device)
        init_h = 0.5 + torch.randn(n_samples, self.h_dim).to(self.device)
        init = [init_v, init_h]
        samples, es = annealed_langevin_dynamic(self, init, self.sigma, self.Ts, self.sample_every,
                                                mode="joint", denoise=self.denoise, sigma0=self.sigma0)
        self.train(mode)
        return samples[-1]

    def sample_marginal(self, n_samples):
        mode = self.training
        self.eval()
        init_v = 0.5 + torch.randn(n_samples, *self.v_shape).to(self.device)
        init = [init_v]
        samples, es = annealed_langevin_dynamic(self, init, self.sigma, self.Ts, self.sample_every,
                                                mode="marginal", denoise=self.denoise, sigma0=self.sigma0)
        self.train(mode)
        return samples[-1][0]

    def sample(self, n_samples, **kwargs):
        if self.h_dim == 0:
            return self.sample_marginal(n_samples)
        else:
            return self.sample_joint(n_samples)[0]

    def sample_h(self, n_samples):
        return self.sample_joint(n_samples)[1]

    def csample_v(self, h, **kwargs):
        assert h.dim() == 2
        mode = self.training
        self.eval()
        init_v = kwargs.get("init_v", 0.5 + torch.randn(len(h), *self.v_shape).to(self.device))
        init = [init_v]
        samples, es = annealed_langevin_dynamic(self, init, self.sigma, self.Ts, self.sample_every,
                                                mode="csample_v", h=h, denoise=self.denoise, sigma0=self.sigma0)
        self.train(mode)
        return samples[-1][0]

    def csample_h(self, v):
        mode = self.training
        self.eval()
        init_h = 0.5 + torch.randn(len(v), self.h_dim).to(self.device)
        init = [init_h]
        samples, es = annealed_langevin_dynamic(self, init, self.sigma, self.Ts, self.sample_every,
                                                mode="csample_h", v=v, denoise=False)
        self.train(mode)
        return samples[-1][0]

    def log_likelihood(self, v, **kwargs):  # un-normalized
        n_particles = 200
        sigma = 5.0
        h = torch.randn(n_particles, len(v), self.h_dim).to(self.device)
        log_q = -(h ** 2).sum(dim=-1) / (2. * sigma ** 2)
        log_p = - self.energy_net(v, h)
        log_w = log_p - log_q
        return mylogsumexp(log_w, dim=0)


class LatentEBMResNet(EBM):
    def __init__(self):
        super(LatentEBMResNet, self).__init__()
        self.channels = config.get("model", "channels")
        self.feature_net = config.get("model", "feature_net")
        self.scalar_net = config.get("model", "scalar_net")
        self.hw = hw = self.v_shape[-1]
        in_channels = self.v_shape[0]

        self.af = nn.ELU()
        self.v2f = nn.DataParallel(eval(self.feature_net)(in_channels, self.channels))
        self.f2h = nn.Linear((hw // 8) ** 2 * 8 * self.channels, self.h_dim)
        self.cp2e = eval(self.scalar_net)(in_features=self.h_dim * 2,
                                          features=(hw // 8) ** 2 * 4 * self.channels)

    def energy_net(self, v, h):
        # h: (n_particles *) batch_size * h_dim
        fh = self.f2h(self.v2f(v))
        if h.dim() == 2:
            cp = torch.cat([fh + h, h], dim=-1)
            return self.cp2e(cp)
        elif h.dim() == 3:
            assert v.size(0) == h.size(1)
            n_particles = h.size(0)
            batch_size = h.size(1)
            fh = duplicate(fh, n_particles).flatten(0, 1)
            h = h.flatten(0, 1)
            cp = torch.cat([fh + h, h], dim=-1)
            return self.cp2e(cp).view(n_particles, batch_size)
        else:
            raise ValueError


class EBMResNet(EBM):
    def __init__(self):
        super(EBMResNet, self).__init__()
        self.channels = config.get("model", "channels")
        self.feature_net = config.get("model", "feature_net")
        self.scalar_net = config.get("model", "scalar_net")
        hw = self.v_shape[-1]
        in_channels = self.v_shape[0]

        self.af = nn.ELU()
        self.v2f = nn.DataParallel(eval(self.feature_net)(in_channels, self.channels))
        self.f2e = eval(self.scalar_net)(in_features=(hw // 8) ** 2 * 8 * self.channels,
                                         features=(hw // 8) ** 2 * 4 * self.channels)

    def free_energy_net(self, v):
        f = self.v2f(v)
        return self.f2e(f)
