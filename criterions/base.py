import evaluator.utils
from utils.config import config
import torch.optim as optim


def _init_optim_default(criterion):
    optimizer = config.get("optim", "optimizer")
    scheduler = config.get("optim", "scheduler")
    lr = config.get("optim", "lr")
    weight_decay = config.get("optim", "weight_decay", default=0.)
    params = []
    for name, model in criterion.models.items():
        params += model.parameters()
    # init optimizers
    if optimizer == "Adam":
        betas = tuple(config.get("optim", "betas", default=(0.9, 0.95)))
        criterion.opts["all"] = optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer == "RMSProp":
        criterion.opts["all"] = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGD":
        momentum = config.get("optim", "momentum", default=0.01)
        criterion.opts["all"] = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    # init schedulers
    n_its = config.get("training", "n_its")
    if scheduler == "step":
        criterion.schs["all"] = optim.lr_scheduler.StepLR(criterion.opts["all"], n_its // 6)
    elif scheduler == "cosine":
        criterion.schs["all"] = optim.lr_scheduler.CosineAnnealingLR(criterion.opts["all"], n_its, eta_min=1e-6)
    elif scheduler == "const":
        criterion.schs["all"] = optim.lr_scheduler.StepLR(criterion.opts["all"], n_its)
    else:
        raise NotImplementedError


def _init_optim_bilevel(criterion):
    optimizer = config.get("optim", "optimizer")
    scheduler = config.get("optim", "scheduler")
    lr = config.get("optim", "lr")
    lr_q = config.get("optim", "lr_q", default=lr)
    weight_decay = config.get("optim", "weight_decay", default=0.)
    model, q = criterion.models["model"], criterion.models["q"]
    # init optimizers
    if optimizer == "Adam":
        betas = tuple(config.get("optim", "betas", default=(0.9, 0.95)))
        criterion.opts["model"] = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        criterion.opts["q"] = optim.Adam(q.parameters(), lr=lr_q, weight_decay=weight_decay, betas=betas)
    elif optimizer == "RMSProp":
        criterion.opts["model"] = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion.opts["q"] = optim.RMSprop(q.parameters(), lr=lr_q, weight_decay=weight_decay)
    elif optimizer == "SGD":
        momentum = config.get("optim", "momentum", default=0.01)
        criterion.opts["model"] = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion.opts["q"] = optim.SGD(q.parameters(), lr=lr_q, momentum=momentum, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    # init schedulers
    n_its = config.get("training", "n_its")
    if scheduler == "step":
        criterion.schs["model"] = optim.lr_scheduler.StepLR(criterion.opts["model"], n_its // 6)
        criterion.schs["q"] = optim.lr_scheduler.StepLR(criterion.opts["q"], n_its // 6)
    elif scheduler == "cosine":
        criterion.schs["model"] = optim.lr_scheduler.CosineAnnealingLR(criterion.opts["model"], n_its, eta_min=1e-6)
        criterion.schs["q"] = optim.lr_scheduler.CosineAnnealingLR(criterion.opts["q"], n_its, eta_min=1e-6)
    elif scheduler == "const":
        criterion.schs["model"] = optim.lr_scheduler.StepLR(criterion.opts["model"], n_its)
        criterion.schs["q"] = optim.lr_scheduler.StepLR(criterion.opts["q"], n_its)
    else:
        raise NotImplementedError


class Criterion(object):
    r""" criterion does:
         1. training
             a. calculating loss
             b. calculating gradient
             c. updating parameters
         2. evaluate
    """

    def __init__(self, models):
        self.models = models
        self.loss_val = None
        self.mid_vals = {}
        self.opts = {}  # optimizers
        self.schs = {}  # sch
        self.init_optim()

    def loss(self, v):
        raise NotImplementedError

    def update(self, v, runner):
        self.loss_val = loss = self.loss(v).mean()
        self.opts["all"].zero_grad()
        loss.backward()
        self.opts["all"].step()
        self.schs["all"].step()

    def evaluate(self, dataset, runner):
        return evaluator.utils.evaluate(dataset, runner, self.loss)

    def train_loss_name(self):
        return self.__class__.__name__.lower()

    def evaluate_loss_name(self):
        return self.train_loss_name()

    def init_optim(self):
        _init_optim_default(self)

    def load_optim_ckpt(self, opts_states, schs_states):
        for k, state in opts_states.items():
            self.opts[k].load_state_dict(state)
        for k, state in schs_states.items():
            self.schs[k].load_state_dict(state)

    def get_optim_ckpt(self):
        opts_states, schs_states = {}, {}
        for k, opt in self.opts.items():
            opts_states[k] = opt.state_dict()
        for k, sch in self.schs.items():
            schs_states[k] = sch.state_dict()
        return opts_states, schs_states
