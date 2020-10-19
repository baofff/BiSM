from utils.config import config
from utils.plot import *
from torchvision.utils import make_grid, save_image
import numpy as np
import torch
from .utils import *
import os
from utils import my_mkdir
from torch.utils.data import DataLoader
from utils.writer import writer
from .utils import classify_features, dimension_reduction, rect_interpolate, linear_interpolate
import logging
from utils import amortize


class BaseEvaluator(object):
    def __init__(self, runner, evaluator_root):
        self.runner = runner
        self.models = runner.models
        self.model = self.models.get("model")
        self.q = self.models.get("q")
        self.evaluator_root = evaluator_root
        self.data_factory = runner.data_factory
        self.tr = runner.tr
        self.te = runner.te
        self.labelled_tr = runner.labelled_tr
        self.labelled_te = runner.labelled_te

    def _prepare(self, func_name, name, ext):
        dir_path = os.path.join(self.evaluator_root, func_name)
        my_mkdir(dir_path)
        fname = os.path.join(dir_path, name + "." + ext)
        return fname

    def plot_sample_density(self, name, **kwargs):
        fname = self._prepare("plot_sample_density", name, "png")
        if config.v_dim == 2:
            if config.h_dim > 10:
                logging.warning("plot_density is skipped, since h_dim is too large to get an accurate estimation")
                return
            xs = torch.linspace(0, 1, steps=100)
            xs, ys = torch.meshgrid([xs, xs])
            xs, ys = xs.flatten().unsqueeze(dim=-1), ys.flatten().unsqueeze(dim=-1)
            v = torch.cat([xs, ys], dim=-1).to(config.device)
            density = self.model.log_likelihood(v).exp()
            xs, ys, = xs.view(100, 100).detach().cpu().numpy(), ys.view(100, 100).detach().cpu().numpy()
            density = density.view(100, 100).detach().cpu().numpy()
            plot_density(xs, ys, density, fname)

    def plot_sample_kde(self, name, **kwargs):
        fname = self._prepare("plot_sample_kde", name, "png")
        if config.v_dim == 2:
            samples = self.model.sample(1000).detach().cpu().numpy()
            plot_kde(samples, fname)

    def plot_sample_scatter(self, name, **kwargs):
        fname = self._prepare("plot_sample_scatter", name, "png")
        if config.v_dim == 2:
            samples = self.model.sample(1000, random=True).detach().cpu().numpy()
            plot_scatter(samples, fname)

    def plot_prior(self, name, **kwargs):
        fname = self._prepare("plot_prior", name, "png")
        if config.h_dim == 2:
            samples = self.model.sample_h(1000).detach().cpu().numpy()
            plot_kde(samples, fname)

    def plot_posterior(self, name, mode="approx", **kwargs):
        fname = self._prepare("plot_posterior", name, "png")
        clustered = {}
        for i in range(10):
            clustered[i] = []
        if mode == "approx":
            te_features, te_ys = extract_labelled_feature(self.q, self.labelled_te)
            if config.h_dim > 2:
                te_features = dimension_reduction(te_features)
            for feature, y in zip(te_features, te_ys):
                clustered[y].append(feature)
        elif mode == "true":
            for i in range(len(self.labelled_te)):
                x, y = self.labelled_te[i]
                clustered[int(y)].append(x.unsqueeze(dim=0))
            vs = []
            for i in range(10):
                clustered[i] = torch.cat(clustered[i][:100], dim=0)
                vs.append(clustered[i])
            vs = torch.cat(vs, dim=0)
            hs = self.model.csample_h(vs).detach().cpu().numpy()
            if config.h_dim > 2:
                hs = dimension_reduction(hs)
            for i in range(10):
                clustered[i] = hs[i * 100: (i + 1) * 100]
        for i in range(10):
            dat = np.array(clustered[i])
            plt.scatter(dat[:, 0], dat[:, 1], label="{}".format(i))
        plt.legend()
        plt.savefig(fname)
        plt.close()
        torch.save(clustered, fname + ".pt")

    def sample(self, name, **kwargs):
        fname = self._prepare("sample", name, "png")
        sample_method = config.get("others", "sample_method", default="normal")
        if sample_method == "normal":
            samples = self.model.sample(100, random=False)
            samples = self.data_factory.unpreprocess(samples)
            grid = make_grid(samples, 10)
            save_image(grid, fname)
        elif sample_method == "via_q":
            idxes = np.random.randint(0, len(self.tr), size=100)
            v = torch.stack(list(map(lambda idx: self.tr[idx], idxes)), dim=0).to(config.device)
            feature, _ = self.q.moments(v)
            samples = self.model.csample_v(feature, random=False)
            samples = self.data_factory.unpreprocess(samples)
            grid = make_grid(samples, 10)
            save_image(grid, fname)

    def cond_sample(self, name, **kwargs):
        fname = self._prepare("cond_sample", name, "png")
        labels = [0, 1, 4, 9]
        samples = list(map(lambda label: get_sample(label, self.labelled_te).unsqueeze(dim=0), labels))
        v = torch.cat(samples, dim=0).to(config.device)
        feature, _ = self.q.moments(v)
        feature = feature.unsqueeze(dim=1).expand(4, 25, config.h_dim).contiguous().view(100, config.h_dim)
        samples = self.model.csample_v(feature, random=False)
        samples = self.data_factory.unpreprocess(samples)
        grids = []
        for i in range(4):
            grids.append(make_grid(samples[i * 25: (i + 1) * 25], 5))
        grid = make_grid(grids, 2, padding=4)
        save_image(grid, fname)

    def v_energy_hist(self, name, **kwargs):  # fix h in a hist
        fname = self._prepare("v_energy_hist", name, "png")
        name, ext = os.path.splitext(fname)

        test_samples = []
        for i in range(10):
            test_samples.append(get_samples(i, 100, self.labelled_te))
        test_samples = torch.cat(test_samples, dim=0).to(config.device)

        for i in range(10):
            sample = get_sample(i, self.labelled_tr).unsqueeze(dim=0).to(config.device)
            feature, _ = self.q.moments(sample)
            dup_feature = feature.expand(1000, *feature.shape[1:])
            test_energy = self.model.energy_net(test_samples, dup_feature)

            figure = plt.figure()
            for j in range(10):
                plt.hist(test_energy[j * 100: (j + 1) * 100].data.cpu().numpy(),
                         alpha=0.5, label="E(v_%d, h_%d)" % (j, i))
            plt.legend(loc="upper right")
            figure.savefig(name + "_h_%d.png" % i)

    def h_energy_hist(self, name, **kwargs):  # fix v in a hist
        fname = self._prepare("h_energy_hist", name, "png")
        name, ext = os.path.splitext(fname)

        test_samples = []
        for i in range(10):
            test_samples.append(get_samples(i, 100, self.labelled_te))
        test_samples = torch.cat(test_samples, dim=0).to(config.device)
        test_features, _ = self.q.moments(test_samples)

        for i in range(10):
            sample = get_sample(i, self.labelled_tr).unsqueeze(dim=0).to(config.device)
            dup_sample = sample.expand(1000, *sample.shape[1:])
            test_energy = self.model.energy_net(dup_sample, test_features)

            figure = plt.figure()
            for j in range(10):
                plt.hist(test_energy[j * 100: (j + 1) * 100].data.cpu().numpy(),
                         alpha=0.5, label="E(v_%d, h_%d)" % (i, j))
            plt.legend(loc="upper right")
            figure.savefig(name + "_v_%d.png" % i)

    def reconstruct(self, name, **kwargs):
        fname = self._prepare("reconstruct", name, "png")
        tr_loader = DataLoader(self.tr, batch_size=50, shuffle=True)
        v = next(iter(tr_loader)).to(config.device)
        v_reconstruct = self.model.reconstruct(v).detach()
        v = self.data_factory.unpreprocess(v)
        v_reconstruct = self.data_factory.unpreprocess(v_reconstruct)
        grid_v = make_grid(v, 5)
        grid_v_reconstruct = make_grid(v_reconstruct, 5)
        grid = make_grid([grid_v, grid_v_reconstruct])
        save_image(grid, fname)

    def _rect_interpolate(self, fname, h_from="approx"):
        labels = [0, 1, 4]
        samples = list(map(lambda label: get_sample(label, self.labelled_te).unsqueeze(dim=0), labels))
        v = torch.cat(samples, dim=0).to(config.device)
        if h_from == "approx":
            feature, _ = self.q.moments(v)
        elif h_from == "true":
            feature = self.model.csample_h(v)
        else:
            raise NotImplementedError
        h = rect_interpolate(*feature, steps=10).to(config.device)
        init_v = rect_interpolate(*v, steps=10).to(config.device)
        samples = self.model.csample_v(h, random=False, init_v=init_v)
        samples = self.data_factory.unpreprocess(samples)
        grid = make_grid(samples, 10)
        save_image(grid, fname)

    def _linear_interpolate(self, fname, h_from="approx"):
        labels = [1, 7]
        samples = list(map(lambda label: get_sample(label, self.labelled_te).unsqueeze(dim=0), labels))
        v = torch.cat(samples, dim=0).to(config.device)
        if h_from == "approx":
            feature, _ = self.q.moments(v)
        elif h_from == "true":
            feature = self.model.csample_h(v)
        else:
            raise NotImplementedError
        h = linear_interpolate(*feature, steps=10).to(config.device)
        init_v = linear_interpolate(*v, steps=10).to(config.device)
        samples = self.model.csample_v(h, random=False, init_v=init_v)
        samples = self.data_factory.unpreprocess(samples)
        grid = make_grid(samples, 10)
        save_image(grid, fname)

    def interpolate(self, name, mode="rect", h_from="approx", **kwargs):
        fname = self._prepare("interpolate", name, "png")
        if mode == "linear":
            self._linear_interpolate(fname, h_from=h_from)
        elif mode == "rect":
            self._rect_interpolate(fname, h_from=h_from)
        else:
            raise NotImplementedError

    def classify(self, it, **kwargs):
        tr_features, tr_ys = extract_labelled_feature(self.q, self.labelled_tr)
        te_features, te_ys = extract_labelled_feature(self.q, self.labelled_te)

        classifiers = kwargs.get("classifiers", "lsvm")
        if not isinstance(classifiers, list):
            assert isinstance(classifiers, str)
            if classifiers == "all":
                classifiers = ["kn", "svm", "lsvm", "logistic"]
            else:
                classifiers = [classifiers]
        for classifier in classifiers:
            acc = classify_features(tr_features, tr_ys, te_features, te_ys, classifier)[-1]
            logging.info("[{} classify] [it: {}] [acc: {}]".format(classifier, it, acc))
            writer.add_scalar("{}_classify".format(classifier), acc, global_step=it)

    def sample2dir(self, path, n_samples):
        sample_method = config.get("others", "sample_method", default="normal")
        idx = 0
        for batch_size in amortize(n_samples, self.runner.batch_size):
            if sample_method == "normal":
                samples = self.model.sample(batch_size, random=False)
                samples = self.data_factory.unpreprocess(samples)
            elif sample_method == "via_q":
                idxes = np.random.randint(0, len(self.tr), size=batch_size)
                v = torch.stack(list(map(lambda idx: self.tr[idx], idxes)), dim=0).to(config.device)
                feature, _ = self.q.moments(v)
                samples = self.model.csample_v(feature, random=False)
                samples = self.data_factory.unpreprocess(samples)
            else:
                raise NotImplementedError
            for sample in samples:
                save_image(sample, os.path.join(path, "%d.png" % idx))
                idx += 1
