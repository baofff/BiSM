from .base import Criterion


def elbo(v, model, q, usage="mle"):
    h, log_q = q.implicit_net_log_q(v)
    h = h.squeeze(dim=0)
    log_q = log_q.squeeze(dim=0)
    if usage == "mle":
        log_p = model.log_joint(v, h)  # log p(v, h)
    elif usage == "sm":
        log_p = -model.energy_net(v, h)
    else:
        raise NotImplementedError
    return log_p - log_q


class ELBO(Criterion):
    def __init__(self, models):
        super(ELBO, self).__init__(models)
        self.model = models["model"]
        self.q = models["q"]

    def train_loss_name(self):
        return "nelbo"

    def elbo(self, v):
        return elbo(v, self.model, self.q)

    def loss(self, v):
        return -self.elbo(v)
