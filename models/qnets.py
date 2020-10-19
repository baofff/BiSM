# The approximate posteriors
from utils.config import config
from .utils import *
import torch.nn.functional as F
from .modules import *
from utils.func import duplicate


class GaussQ(nn.Module):
    def __init__(self):
        super(GaussQ, self).__init__()
        self.h_dim = config.h_dim
        self.v_dim = config.v_dim
        self.v_shape = config.v_shape

    def moments(self, v):
        raise NotImplementedError

    def cmean_h(self, v):
        return self.moments(v)[0]

    def log_q(self, h, v):  # log q(h|v)
        # h: (n_particles *) batch_size * h_dim
        mean, log_std = self.moments(v)
        e = ((h - mean) ** 2 * torch.exp(-2. * log_std)).sum(dim=-1) * -0.5
        reg = -log_std.sum(dim=-1)
        c = np.log(2 * np.pi) * self.h_dim * -0.5
        return e + reg + c

    def implicit_net(self, v, n_particles=1):  # sample from q(h|v) as an implicit model
        eps = torch.randn(n_particles, len(v), self.h_dim).to(v.device)
        mean, log_std = self.moments(v)
        h = mean + log_std.exp() * eps
        return h

    def implicit_net_log_q(self, v, n_particles=1):
        mean, log_std = self.moments(v)
        eps = torch.randn(n_particles, len(v), self.h_dim).to(v.device)
        h = mean + log_std.exp() * eps
        # log q(h|v)
        e = ((h - mean).pow(2) * torch.exp(-2. * log_std)).sum(dim=-1) * -0.5
        reg = -log_std.sum(dim=-1)
        c = np.log(2 * np.pi) * self.h_dim * -0.5
        log_q = e + reg + c
        return h, log_q

    def csample_h(self, v, n_particles=1):  # sample from q(h|v)
        return self.implicit_net(v, n_particles=n_particles)


class MLPGaussQ(GaussQ):  # the standard q used in iwae
    def __init__(self):
        super(MLPGaussQ, self).__init__()
        self.main = nn.Sequential(MLP2(self.v_dim), nn.Tanh())
        self.mean = nn.Linear(200, self.h_dim)
        self.log_std = nn.Linear(200, self.h_dim)

    def forward(self, v):
        m = self.main(v.flatten(1))
        mean = self.mean(m)
        log_std = self.log_std(m)
        return mean, log_std

    def moments(self, v):
        return self.forward(v)


class Conv3GaussQ(GaussQ):
    def __init__(self):
        super(Conv3GaussQ, self).__init__()
        in_channels = self.v_shape[0]
        self.hw = hw = self.v_shape[-1]
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.mean = nn.Linear((hw // 8) ** 2 * 256, self.h_dim)
        self.log_std = nn.Linear((hw // 8) ** 2 * 256, self.h_dim)

    def forward(self, v):
        output = self.conv1(v)
        output = self.lrelu(output)
        output = self.conv2(output)
        output = self.lrelu(output)
        output = self.conv3(output)
        output = self.lrelu(output)
        output = output.view(-1, (self.hw // 8) ** 2 * 256)
        mean = self.mean(output)
        log_std = self.log_std(output)
        return mean, log_std

    def moments(self, v):
        return self.forward(v)


class Conv5GaussQ(GaussQ):
    def __init__(self):
        super(Conv5GaussQ, self).__init__()
        in_channels = self.v_shape[0]
        self.hw = hw = self.v_shape[-1]
        self.lrelu = nn.LeakyReLU(0.2)
        self.feature_net = nn.Sequential(nn.Conv2d(in_channels, 64, 3, stride=2, padding=1), self.lrelu,
                                         nn.Conv2d(64, 64, 3, stride=1, padding=1), self.lrelu,
                                         nn.Conv2d(64, 128, 3, stride=2, padding=1), self.lrelu,
                                         nn.Conv2d(128, 128, 3, stride=1, padding=1), self.lrelu,
                                         nn.Conv2d(128, 256, 3, stride=2, padding=1))
        self.mean = nn.Linear((hw // 8) ** 2 * 256, self.h_dim)
        self.log_std = nn.Linear((hw // 8) ** 2 * 256, self.h_dim)

    def forward(self, v):
        output = self.feature_net(v)
        output = self.lrelu(output)
        output = output.view(-1, (self.hw // 8) ** 2 * 256)
        mean = self.mean(output)
        log_std = self.log_std(output)
        return mean, log_std

    def moments(self, v):
        return self.forward(v)


class BernoulliQ(nn.Module):
    def __init__(self):
        super(BernoulliQ, self).__init__()
        self.h_dim = config.h_dim
        self.v_dim = config.v_dim
        self.v_shape = config.v_shape
        self.temperature = config.get("q", "temperature")

    def logits(self, v):
        raise NotImplementedError

    def cmean_h(self, v):
        return self.logits(v).sigmoid()

    def log_q(self, h, v):  # log q(h|v)
        # h: (n_particles *) batch_size * h_dim
        n_particles = h.size(0) if h.dim() == 3 else 1
        if h.dim() == 2:
            h = h.unsqueeze(dim=0)
        logits = self.logits(v)
        dup_logits = duplicate(logits, n_particles)
        res = -F.binary_cross_entropy_with_logits(dup_logits, h, reduction="none").sum(dim=-1)
        if h.dim() == 2:
            res.squeeze(dim=0)
        return res

    def implicit_net(self, v, n_particles=1):  # sample from q(h|v) as an implicit model
        logits = self.logits(v).unsqueeze(dim=-1)
        paddings = torch.zeros_like(logits)
        logits = torch.cat([logits, paddings], dim=-1)
        log_pi = logits.log_softmax(dim=-1)  # the parameter of the distribution
        dup_log_pi = duplicate(log_pi, n_particles)
        y = gumbel_softmax(dup_log_pi, temperature=self.temperature)
        h = y.select(-1, 0)
        return h  # n_particles * batch_size * h_dim

    def implicit_net_log_q(self, v, n_particles=1):
        logits = self.logits(v)
        logits_ = logits.unsqueeze(dim=-1)
        paddings = torch.zeros_like(logits_)
        logits_ = torch.cat([logits_, paddings], dim=-1)
        log_pi = logits_.log_softmax(dim=-1)  # the parameter of the distribution
        dup_log_pi = duplicate(log_pi, n_particles)
        y = gumbel_softmax(dup_log_pi, temperature=self.temperature)
        h = y.select(-1, 0)
        # log q(h|v)
        dup_logits = duplicate(logits, len(h))
        log_q = -F.binary_cross_entropy_with_logits(dup_logits, h, reduction="none").sum(dim=-1)
        return h, log_q

    def csample_h(self, v, n_particles=1):  # sample from q(h|v)
        return self.implicit_net(v, n_particles=n_particles)


class LinearBernoulliQ(BernoulliQ):
    def __init__(self):
        super(LinearBernoulliQ, self).__init__()
        self.main = nn.Linear(self.v_dim, self.h_dim)

    def forward(self, v):
        return self.main(v)

    def logits(self, v):
        return self.forward(v)


class MLPBernoulliQ(BernoulliQ):
    def __init__(self):
        super(MLPBernoulliQ, self).__init__()
        self.main = nn.Sequential(nn.Linear(self.v_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Softplus(),
                                  nn.Linear(self.h_dim, self.h_dim))

    def forward(self, v):
        return self.main(v)

    def logits(self, v):
        return self.forward(v)
