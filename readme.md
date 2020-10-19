# Bi-level Score Matching for Learning Energy-based Latent Variable Models

Code for the paper [Bi-level Score Matching for Learning Energy-based Latent Variable Models](https://arxiv.org/abs/2010.07856).


## Requirements

See environment.yml. You can create the environment by running
```
conda env create -f environment.yml
```

## Run BiSM
To compare BiSM with baselines (including [CD](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf), [PCD](http://icml2008.cs.helsinki.fi/papers/638.pdf), [DSM](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf), [SSM](https://arxiv.org/abs/1905.07088), [NCE](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf), [VNCE](https://arxiv.org/abs/1810.08010)) on the toy (checkerboard) dataset, run
```
python checkerboard_exps.py
```

To see how N and K influence the results, run
```
python freyface_exps.py
```

To train an EBLVM on the mnist dataset (1 * 28 * 28) with BiSM, run
```
python tune_latent_ebm_resnet_mnist_bimdsm.py
```

To train an EBLVM on the cifar10 dataset (3 * 32 * 32) with BiSM, run
```
python tune_latent_ebm_resnet_cifar10_bimdsm.py
```

To train an EBLVM on the celeba dataset (3 * 32 * 32) with BiSM, run
```
python tune_latent_ebm_resnet_celeba_bimdsm.py
```


## Run Baselines
To compare BiSM with baselines (including [CD](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf), [PCD](http://icml2008.cs.helsinki.fi/papers/638.pdf), [DSM](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf), [SSM](https://arxiv.org/abs/1905.07088), [NCE](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf), [VNCE](https://arxiv.org/abs/1810.08010)) on the toy (checkerboard) dataset, run
```
python checkerboard_exps.py
```

To train an EBM on the mnist dataset (1 * 28 * 28) with [MDSM](https://arxiv.org/abs/1910.07762), run
```
python tune_ebm_resnet_mnist_mdsm.py
```

To train an EBM on the cifar10 dataset (3 * 32 * 32) with [MDSM](https://arxiv.org/abs/1910.07762), run
```
python tune_ebm_resnet_cifar10_mdsm.py
```


## Remark
* The code will detect free GPUs by command **gpustat** and run on these GPUs.
You can manually assign GPUs by modify the **devices** argument of function **task_assign** in the above .py files.

* The downloaded dataset and running result will be saved to **workspace** directory by default.
