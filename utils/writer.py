from torch.utils.tensorboard import SummaryWriter
import torch
import os


class Writer(object):
    def __init__(self):
        self.writer = None
        self.path = None

    def set_path(self, path):
        self.writer = SummaryWriter(path)
        self.path = path

    def add_scalar(self, tag, scalar_value, global_step):
        self.writer.add_scalar(tag, scalar_value, global_step)


writer = Writer()
