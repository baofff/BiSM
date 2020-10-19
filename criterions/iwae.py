import numpy as np
from utils.func import mylogsumexp
from .base import Criterion
from utils.config import config


def iwae(v, model, q, k, usage="mle"):
    h, log_q = q.implicit_net_log_q(v, n_particles=k)
    if usage == "mle":
        log_p = model.log_joint(v, h)
    elif usage == "sm":
        log_p = -model.energy_net(v, h)
    else:
        raise NotImplementedError
    log_w = log_p - log_q
    iwae = mylogsumexp(log_w, dim=0) - np.log(k)
    return iwae


class IWAE(Criterion):
    def __init__(self, models):
        super(IWAE, self).__init__(models)
        self.model = models["model"]
        self.q = models["q"]
        self.k = config.get("loss", "k")

    def train_loss_name(self):
        return "niwae%d" % self.k

    def iwae(self, v):
        return iwae(v, self.model, self.q, self.k, usage="mle")

    def loss(self, v):
        return -self.iwae(v)
