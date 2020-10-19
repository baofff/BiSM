import numpy as np


def get_val(dict_, *fields, **kwargs):
    try:
        cur = dict_
        for field in fields:
            cur = cur[field]
        return cur
    except Exception as e:
        if "default" in kwargs.keys():
            return kwargs["default"]
        else:
            raise e


class Config(object):
    def __init__(self):
        self.dict_ = None

    def set_config(self, dict_):
        assert self.dict_ is None
        self.dict_ = dict_

    def get(self, *fields, **kwargs):
        return get_val(self.dict_, *fields, **kwargs)

    def __contains__(self, item):
        return item in self.dict_

    def __str__(self):
        return str(self.dict_)

    @ property
    def workspace_root(self):
        return self.get("workspace_root")

    @ property
    def device(self):
        return self.get("device")

    @ property
    def name(self):
        return self.get("name")

    @ property
    def h_dim(self):
        return config.get("model", "h_dim", default=0)

    @property
    def v_dim(self):
        v_dim = config.get("model", "v_dim", default=None)
        if v_dim is None:
            v_dim = int(np.prod(self.v_shape))
        return v_dim

    @property
    def v_shape(self):
        return config.get("model", "v_shape", default=None)


config = Config()
