import logging
import os
from .utils import *
from datasets import *
from criterions import *
from evaluator import *
from utils.writer import writer
from models import *
from utils.config import config


class BaseRunner(nn.Module):
    r""" The abstract interface for running a training procedure
    """
    def __init__(self):
        super(BaseRunner, self).__init__()
        self.batch_size = config.get("training", "batch_size")
        self.n_its = config.get("training", "n_its")
        self.n_ckpts = config.get("training", "n_ckpts", default=10)
        self.log_invl = config.get("interval", "log", default=100)
        self.name = config.name
        self.device = config.device
        self.workspace_root = config.workspace_root
        self.roots = {}

        # set path
        self.model_root = os.path.join(self.workspace_root, "models")
        if not os.path.exists(self.model_root):
            os.makedirs(self.model_root)
        self.ckpt_path = os.path.join(self.model_root, "{}.ckpt.pth".format(self.name))
        self.model_path = os.path.join(self.model_root, "{}.pth".format(self.name))

        # set dataset
        self.data_factory = None
        self.dataset = None
        self.tr = None
        self.val = None
        self.te = None
        self.labelled_tr = None
        self.labelled_te = None
        self.init_dataset()

        # set model
        self.models = {}
        self.init_model()
        self.model = self.models.get("model")
        self.q = self.models.get("q")

        # set criterion
        self.criterion = None
        self.init_criterion()

        # states of training
        self.it = None
        self.best_val_loss = None

        # set evaluator
        self.evaluator_root = os.path.join(self.workspace_root, "evaluation")
        self.evaluator_family = config.get("evaluator", "family", default="BaseEvaluator")
        self.evaluator = eval(self.evaluator_family)(self, self.evaluator_root)

    def init_dataset(self):
        self.dataset = config.get("data", "dataset")
        self.data_factory = eval(self.dataset)()

        if config.get("data", "use_val", default=False):
            self.tr = self.data_factory.get_train_data()
            self.val = self.data_factory.get_val_data()
        else:
            self.tr = self.data_factory.get_train_val_data()
            self.val = None
        self.te = self.data_factory.get_test_data()

        if self.data_factory.allow_labelled():
            self.labelled_tr = self.data_factory.get_train_val_data(labelled=True)
            self.labelled_te = self.data_factory.get_test_data(labelled=True)

    def init_model(self):
        self.models["model"] = eval(config.get("model", "family"))()
        if "q" in config:
            self.models["q"] = eval(config.get("q", "family"))()

    def init_criterion(self):
        self.criterion = eval(config.get("criterion", "family"))(self.models)

    def remove_model(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.ckpt_path):
            os.remove(self.ckpt_path)

    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        model = torch.load(path)
        if 'model' in model.keys():
            model = model['model']
        self.load_state_dict(model)
        self.to(self.device)
        self.eval()

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)

    def save_ckpt(self, it):
        opts_states, schs_states = self.criterion.get_optim_ckpt()
        ckpt = {
            'model': self.state_dict(),
            'opts': opts_states,
            'schs': schs_states,
            'best_val_loss': self.best_val_loss,
            'it': it,
        }
        torch.save(ckpt, self.ckpt_path)
        torch.save(ckpt, self.ckpt_path + ".{}".format(it))  # backup

    def load_ckpt(self, path=None):
        if path is None:
            path = self.ckpt_path
        ckpt = torch.load(path)
        self.load_state_dict(ckpt['model'])
        opts_states = ckpt['opts']
        schs_states = ckpt['schs']
        self.criterion.load_optim_ckpt(opts_states, schs_states)
        self.best_val_loss = ckpt['best_val_loss']
        self.it = ckpt['it']
        self.to(self.device)

    def evaluate_during_training(self):
        mode = self.training
        self.eval()
        for fn, flag in config.get("evaluate_options").items():
            times = 0
            if flag is True:
                times = 20
            elif isinstance(flag, int):
                times = flag
            if times > 0:
                invl = self.n_its // times
                if self.it % invl == invl - 1:
                    eval("self.evaluator.{}".format(fn))(name="it_{}".format(self.it), it=self.it)
        self.train(mode)

    def report_train(self, loss, **kwargs):
        logging.info("[train] [it: {}] [{}: {}]".format(self.it, self.criterion.train_loss_name(), loss))
        writer.add_scalar("train_{}".format(self.criterion.train_loss_name()), loss, global_step=self.it)
        for k, v in kwargs.items():
            writer.add_scalar("{}".format(k), v, global_step=self.it)

    def report_test(self, loss):
        logging.info("[test] [it: {}] [{}: {}]".format(self.it, self.criterion.evaluate_loss_name(), loss))
        writer.add_scalar("test_{}".format(self.criterion.evaluate_loss_name()), loss, global_step=self.it)

    def report_val(self, loss):
        logging.info("[val] [it: {}] [{}: {}] [previous best {}: {}]"
                     .format(self.it, self.criterion.evaluate_loss_name(), loss,
                             self.criterion.evaluate_loss_name(), self.best_val_loss))
        writer.add_scalar("val_{}".format(self.criterion.evaluate_loss_name()), loss, global_step=self.it)

    def fit(self):
        self.to(self.device)

        # load model if it exists
        self.best_val_loss = float('inf')  # the smaller the better
        self.it = 0
        if os.path.exists(self.ckpt_path):
            self.load_ckpt()

        tr_loader = infinite_loader(self.tr, batch_size=self.batch_size)

        self.train()
        while self.it < self.n_its:
            v = next(tr_loader).to(self.device)
            self.criterion.update(v, self)
            if self.it % self.log_invl == 0:
                self.report_train(self.criterion.loss_val, **self.criterion.mid_vals)

            self.evaluate_during_training()

            invl = self.n_its // self.n_ckpts
            if self.it % invl == invl - 1 or self.it == self.n_its - 1:
                self.eval()
                if self.te is not None:
                    loss = self.criterion.evaluate(self.te, self)
                    self.report_test(loss)
                if self.val is not None:
                    loss = self.criterion.evaluate(self.val, self)
                    self.report_val(loss)
                    if loss < self.best_val_loss or self.it == invl - 1:  # update the best model
                        self.save_model()
                    if loss < self.best_val_loss:
                        self.best_val_loss = loss
                else:  # when we don't have a validation set, directly save the model
                    self.save_model()
                self.save_ckpt(self.it + 1)  # save the current training checkpoint
                self.train()

            self.it += 1
