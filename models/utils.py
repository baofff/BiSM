import torch.nn as nn
import torch
import numpy as np
import torch.autograd as autograd
from tqdm import tqdm


def get_nonlinear(nonlinear):
    if nonlinear == "relu":
        return nn.ReLU()
    elif nonlinear == "tanh":
        return nn.Tanh()
    elif nonlinear == "softplus":
        return nn.Softplus()
    else:
        raise NotImplementedError


def sample_gumbel(shape, eps=1e-20):
    u = torch.rand(shape)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax(logits, temperature):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return (y / temperature).softmax(dim=-1)


def log_p_normal(v, mean, log_std):
    # v: batch_size * v_dim
    v_dim = v.size(1)
    e = ((v - mean) ** 2 * (-2 * log_std).exp()).sum(dim=-1) * -0.5
    reg = -log_std.sum(dim=-1)
    c = np.log(2 * np.pi) * v_dim * -0.5
    return e + reg + c


def _ss_denoise(fn, noised, sigma0):
    sigma02 = sigma0 ** 2
    for x in noised:
        x.requires_grad_()
    e = fn(noised)
    grads = autograd.grad(e.sum(), noised)
    denoised = []
    for x, grad in zip(noised, grads):
        denoised.append(x.detach() - sigma02 * grad)
    return denoised, e


def _annealed_langevin_dynamic(fn, init, sigma, Ts, sample_every, desc, denoise, **kwargs):
    sigma2 = sigma ** 2
    samples, es = [], []
    inputs = init
    for i, T in tqdm(enumerate(Ts), desc=desc):
        for x in inputs:
            x.requires_grad_()
        e = fn(inputs)
        es.append(e.detach())
        grads = autograd.grad(e.sum(), inputs)
        old_inputs = inputs
        inputs = []
        for x, grad in zip(old_inputs, grads):
            inputs.append(x.detach() - 0.5 * sigma2 * grad + T ** 0.5 * sigma * torch.randn_like(x))
        if (i + 1) % sample_every == 0:
            samples.append(inputs)
    if denoise:
        sigma0 = kwargs["sigma0"]
        denoised, e = _ss_denoise(fn, inputs, sigma0)
        es.append(e)
        samples.append(denoised)
    return samples, es


def annealed_langevin_dynamic(model, init, sigma, Ts, sample_every, mode, denoise, **kwargs):
    if mode == "marginal":
        energy_fn = lambda inputs: model.free_energy_net(*inputs)
    elif mode == "joint":
        energy_fn = lambda inputs: model.energy_net(*inputs)
    elif mode == "csample_v":
        h = kwargs["h"].detach()
        energy_fn = lambda inputs: model.energy_net(*inputs, h)
    elif mode == "csample_h":
        v = kwargs["v"].detach()
        energy_fn = lambda inputs: model.energy_net(v, *inputs)
    else:
        raise NotImplementedError
    return _annealed_langevin_dynamic(energy_fn, init, sigma, Ts, sample_every, mode, denoise, **kwargs)
