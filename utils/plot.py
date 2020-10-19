import matplotlib.pyplot as plt
import seaborn as sns


class PlotContext(object):
    def __init__(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.subplots_adjust(left=0.1, right=0.99, bottom=0.06, top=1.)
        ax.axis('equal')
        ax.margins(0)
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xticks([0., 0.5, 1.])
        ax.set_yticks([0., 0.5, 1.])
        ax.tick_params(axis="both", labelsize=40)
        ax.locator_params(axis="both", nbins=3)
        self.cmap = plt.get_cmap('GnBu')
        self.fig = fig
        self.ax = ax

    def __enter__(self):
        return self.fig, self.ax, self.cmap

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.close(self.fig)


def plot_density(xs, ys, density, fname):
    with PlotContext() as (fig, ax, cmap):
        ax.pcolormesh(xs, ys, density, cmap=cmap)
        fig.savefig(fname)


def plot_scatter(samples, fname):
    with PlotContext() as (fig, ax, cmap):
        ax.scatter(samples[:, 0], samples[:, 1], cmap=cmap)
        fig.savefig(fname)


def plot_kde(samples, fname):
    with PlotContext() as (fig, ax, cmap):
        sns.kdeplot(samples[:, 0], samples[:, 1], shade=True, cmap=cmap, ax=ax)
        fig.savefig(fname)
