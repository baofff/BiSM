import torch
import torch.nn.functional as F


def sos(a, start_dim=1):  # sum of square
    return a.pow(2).flatten(start_dim=start_dim).sum(dim=-1)


def inner_product(a, b, start_dim=1):
    return (a * b).flatten(start_dim=start_dim).sum(dim=-1)


def duplicate(tensor, n_particles):
    return tensor.unsqueeze(dim=0).expand(n_particles, *tensor.shape)


def my_instance_norm(inputs, weight, bias, eps=1e-5):
    res = F.instance_norm(inputs, eps=eps)
    weight_ = weight.view(*weight.shape, 1, 1)
    bias_ = bias.view(*bias.shape, 1, 1)
    return res * weight_ + bias_


def my_log(a, eps=1e-20):
    a = torch.clamp_min(a, eps)
    return a.log()


def binary_ce(a, b, eps=1e-20):
    return -a * my_log(b, eps) - (1 - a) * my_log(1 - b, eps)


def binary_kl(a, b, eps=1e-20):
    return binary_ce(a, b, eps) - binary_ce(a, a, eps)


def mylogsumexp(tensor, dim, keepdim=False):
    # the logsumexp of pytorch is not stable!
    tensor_max, _ = tensor.max(dim=dim, keepdim=True)
    ret = (tensor - tensor_max).exp().sum(dim=dim, keepdim=True).log() + tensor_max
    if not keepdim:
        ret.squeeze_(dim=dim)
    return ret


def fisher_divergence_between_normal(mu_1, cov_1, mu_2, cov_2):
    pre_1 = cov_1.inverse()
    pre_2 = cov_2.inverse()
    mu_ = (mu_1 - mu_2).unsqueeze(dim=1)
    m = mu_ @ mu_.t()

    a = (pre_2 @ pre_2 @ (cov_1 + m)).trace() * 0.5
    b = -pre_2.trace()
    c = pre_1.trace() * 0.5
    return a + b + c


def kl_divergence_between_normal(mu_1, cov_1, mu_2, cov_2):
    pre_2 = cov_2.inverse()
    mu_ = mu_2 - mu_1

    a = (pre_2 @ cov_1).trace()
    b = ((mu_ @ pre_2) * mu_).sum()
    c = -float(len(mu_1)) + cov_2.det().log() - cov_1.det().log()

    return 0.5 * (a + b + c)
