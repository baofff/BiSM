from .base import Criterion
import evaluator.utils
from utils.config import config


class CD(Criterion):
    def __init__(self, models):
        super(CD, self).__init__(models)
        self.model = models["model"]
        self.k = config.get("loss", "k", default=1)

    def train_loss_name(self):
        return "cd%d" % self.k

    def evaluate_loss_name(self):
        return "reconstruction"

    def loss(self, v):
        # k step contrastive divergence
        # The gradient of the cd loss w.r.t. parameter is an estimation of the gradient of the nll
        # *BUT* the value of the cd loss is not an estimation of the nll !!
        h = self.model.csample_h(v).detach()
        e1 = self.model.free_energy_net(v)
        h_sample = h
        for _ in range(self.k):
            v_sample = self.model.csample_v(h_sample).detach()
            h_sample = self.model.csample_h(v_sample).detach()
        e2 = self.model.free_energy_net(v_sample)
        return (e1 - e2).mean()

    def evaluate(self, dataset, runner):
        return evaluator.utils.evaluate(dataset, runner, runner.model.reconstruction_loss)


class PCD(Criterion):  # can't stop
    def __init__(self, models):
        super(PCD, self).__init__(models)
        self.model = models["model"]
        self.k = config.get("loss", "k", default=1)
        self.persistent = None

    def train_loss_name(self):
        return "pcd%d" % self.k

    def evaluate_loss_name(self):
        return "reconstruction"

    def loss(self, v):
        # k step contrastive divergence
        # The gradient of the cd loss w.r.t. parameter is an estimation of the gradient of the nll
        # *BUT* the value of the cd loss is not an estimation of the nll !!
        if self.persistent is None:
            self.persistent = self.model.csample_h(v).detach()
        e1 = self.model.free_energy_net(v)
        h_sample = self.persistent
        for _ in range(self.k):
            v_sample = self.model.csample_v(h_sample).detach()
            h_sample = self.model.csample_h(v_sample).detach()
        e2 = self.model.free_energy_net(v_sample)
        self.persistent = h_sample
        return (e1 - e2).mean()

    def evaluate(self, dataset, runner):
        return evaluator.utils.evaluate(dataset, runner, runner.model.reconstruction_loss)
