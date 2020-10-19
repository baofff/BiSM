import os
import yaml
import subprocess
import logging
import numpy as np
import copy
from .misc import assign_gpu


def set_logger(fname):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(fname, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)


def format_device(device):
    if isinstance(device, int):
        return "{}".format(device)
    elif isinstance(device, tuple) or isinstance(device, list):
        res = "{}".format(device)[1: -1]
        res = res.replace(" ", "")
        return res


def load_yaml(fname):
    with open(fname, "r") as f:
        config = yaml.full_load(f)
    if "include" in config.keys():
        dirname, _ = os.path.split(fname)
        base_fname = os.path.join(dirname, config["include"])
        with open(base_fname, "r") as f:
            base_config = yaml.full_load(f)
        for key in config.keys():
            base_config[key] = config[key]
        config = base_config
        del config["include"]
    return config


class Tuner(object):
    def __init__(self, template):
        self.state_spaces = {}
        self.idx = 0

        self.config_template = load_yaml(os.path.join("configs", template))

        template_dir, template_name = os.path.split(template)
        self.template_name, _ = os.path.splitext(template_name)

        self.tuning_config_root = os.path.join("configs", "tuning", template_dir, self.template_name)
        if not os.path.exists(self.tuning_config_root):
            os.makedirs(self.tuning_config_root)

        # set logger
        log_root = os.path.join("workspace", "tuning_logs", self.config_template['model']['family'])
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        set_logger(os.path.join(log_root, "%s.log" % self.template_name))

    def set_state_space(self, hyperparameters, state_space):
        self.state_spaces[hyperparameters] = state_space

    def tune_grouped_hyperparameters(self, init_state, group, state_space):
        state = copy.deepcopy(init_state)
        best_loss = float('inf')
        best_val = state_space[0]
        for val in state_space:
            name = "{}_{}".format(self.template_name, self.idx)
            for hyperparameter in group:
                state[hyperparameter] = val

            # set config
            config = copy.deepcopy(self.config_template)
            config['name'] = name
            for k, v in state.items():
                k0, k1 = k.split("/")
                config[k0][k1] = v

            # save config
            config_path = os.path.join(self.tuning_config_root, "{}.yml".format(name))
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # conduct experiment
            workspace_root = os.path.join("workspace", "runner", config['model']['family'],
                                          self.template_name, config['name'])
            subprocess.call("python main.py --config {} --workspace {} --mode train"
                            .format(config_path, workspace_root), shell=True)
            subprocess.call("python main.py --config {} --workspace {} --mode test"
                            .format(config_path, workspace_root), shell=True)

            # get experiment result
            test_mean_loss = np.load(os.path.join(workspace_root, "test", "test_mean_loss.npy")).item()
            logging.info("[name: {}] [hyperparameters: {}] [test_mean_loss: {}]".format(name, state, test_mean_loss))
            if test_mean_loss < best_loss:
                best_loss = test_mean_loss
                best_val = val
            self.idx += 1
        for hyperparameter in group:
            state[hyperparameter] = best_val
        logging.info("Choose {} as {} with test_mean_loss {}".format(best_val, group, best_loss))
        return state, best_loss

    def tune(self, state_spaces, tune_order, n_rounds=1):
        state = {}
        for k, v in state_spaces.items():
            self.set_state_space(k, v)  # set state space
            for p in k:
                state[p] = v[0]  # set the initial state
        for r in range(n_rounds):
            for group in tune_order:
                state, loss = self.tune_grouped_hyperparameters(state, group, self.state_spaces[group])
        logging.info("{} is the final hyperparameter with loss {}".format(state, loss))

    def _task_assign(self, state, device, idx, mode):
        name = "{}_{}".format(self.template_name, idx)
        # set config
        config = copy.deepcopy(self.config_template)
        config['name'] = name
        for k, v in state.items():
            k0, k1 = k.split("/")
            config[k0][k1] = v

        # save config
        config_path = os.path.join(self.tuning_config_root, "{}.yml".format(name))
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # conduct experiment
        workspace_root = os.path.join("workspace", "runner", config['model']['family'],
                                      self.template_name, config['name'])
        subprocess.Popen("CUDA_VISIBLE_DEVICES={} python main.py --config {} --workspace {} --mode {}"
                         .format(format_device(device), config_path, workspace_root, mode), shell=True)

    def task_assign(self, states, devices=None, mode="train", **kwargs):
        if devices is None:
            forbidden_devices = kwargs.get("forbidden_devices", ())
            devices = assign_gpu(n_tasks=len(states), black_list=forbidden_devices)
        for idx, (state, device) in enumerate(zip(states, devices)):
            self._task_assign(state, device, idx, mode)
