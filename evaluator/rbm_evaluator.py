from .base_evaluator import BaseEvaluator
import criterions
from datasets import *
import logging
from torchvision.utils import make_grid, save_image
import os
from utils.writer import writer
from .utils import evaluate
from utils.config import config


class RBMEvaluator(BaseEvaluator):
    def __init__(self, runner, evaluator_root):
        super(RBMEvaluator, self).__init__(runner, evaluator_root)
        self.v0 = runner.data_factory.get_uniform_data()
        data = []
        for i in range(len(self.v0)):
            data.append(self.v0[i].numpy())
        self.v0 = torch.tensor(data)

    def sample_v0(self, n_samples):
        idxes = (torch.rand(n_samples) * len(self.v0)).int().clamp(0, len(self.v0) - 1)
        idxes = list(idxes.numpy())
        return self.v0[idxes]

    def log_likelihood(self, it, **kwargs):
        ais = config.get("others", "ais", default=False)
        if ais:
            self.model.update_log_partition()
        ll = evaluate(self.te, self.runner, lambda v: self.model.log_likelihood(v, ais=ais),
                      batch_size=min(10000, len(self.te)))
        logging.info("[log_likelihood] [it: {}] [log_likelihood: {}]".format(it, ll))
        writer.add_scalar("log_likelihood", ll, global_step=it)

    def fisher(self, it, **kwargs):
        fisher_ = evaluate(self.te, self.runner, lambda v: criterions.ssm(v, self.model),
                           batch_size=min(10000, len(self.te)))
        logging.info("[fisher] [it: {}] [fisher: {}]".format(it, fisher_))
        writer.add_scalar("fisher", fisher_, global_step=it)

    def sample(self, name, **kwargs):
        fname = self._prepare("sample", name, "png")
        if config.get("others", "record_gibbs", default=False):
            name, ext = os.path.splitext(fname)
            samples, v0 = self.model.sample(50, ret_v0=True, record_gibbs=True, v0=self.sample_v0(50))
            v0 = self.data_factory.unpreprocess(v0)
            grid_v0 = make_grid(v0, 5)
            for gibbs_step, v in samples:
                v = self.data_factory.unpreprocess(v)
                grid_v = make_grid(v, 5)
                grid = make_grid([grid_v0, grid_v])
                save_image(grid, "{}_gibbs_{}{}".format(name, gibbs_step, ext))
        else:
            super(RBMEvaluator, self).sample(name, **kwargs)
