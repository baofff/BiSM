import logging
import socket
import traceback
import os
import argparse
import torch
import numpy as np
import sys
import random
from utils import set_logger, load_yaml
from utils.writer import writer
from utils.config import config
from runners import *
from evaluator.utils import valid_sample, sample_from_ckpt


def init():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--workspace', type=str, required=True, help='Path to the workspace')
    parser.add_argument('--mode', type=str, default='train', help='Train, valid or test the model (or others)')

    args = parser.parse_args()

    # set config
    config_ = load_yaml(args.config)
    config_["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config_["workspace_root"] = args.workspace
    config.set_config(config_)

    # set writer
    summary_root = os.path.join(config.workspace_root, "summary")
    if not os.path.exists(summary_root):
        os.makedirs(summary_root)
    writer.set_path(summary_root)

    # set seed
    seed = config.get("others", "seed", default=1234)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # set logger
    log_root = os.path.join(config.workspace_root, "logs")
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    log_path = os.path.join(log_root, "{}.log".format(args.mode))
    set_logger(log_path)

    logging.info("running @ {}".format(socket.gethostname()))
    logging.info(config)

    return args


def main():
    args = init()
    runner = BaseRunner()
    if args.mode == "train":
        try:
            runner.fit()
        except:
            logging.error("training is failed")
            logging.error(traceback.format_exc())
    elif args.mode == "valid":
        try:
            valid_sample(runner)
        except:
            logging.error("validation is failed")
            logging.error(traceback.format_exc())
    elif args.mode == "test":
        test_root = os.path.join(config.workspace_root, "test")
        if not os.path.exists(test_root):
            os.makedirs(test_root)
        try:
            runner.load_model()
            runner.eval()
            mean_loss = runner.criterion.evaluate(runner.te, runner).item()
            np.save(os.path.join(test_root, "test_mean_loss.npy"), mean_loss)
            for fn, flag in config.get("evaluate_options").items():
                if flag:
                    eval("runner.evaluator.{}".format(fn))(name="{}.png".format(fn), it=0)
        except:
            np.save(os.path.join(test_root, "test_mean_loss.npy"), float("nan"))
            logging.error("testing is failed")
            logging.error(traceback.format_exc())
    elif args.mode == "test_ckpt":  # test a specific ckpt
        ckpt = config.get("others", "tested_ckpt")
        test_root = os.path.join(config.workspace_root, "test", ckpt)
        sample_from_ckpt(runner, ckpt, 50000, os.path.join(test_root, "samples"))
    else:
        raise NotImplementedError
    return 0


if __name__ == "__main__":
    sys.exit(main())
