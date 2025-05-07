# PSKD: Strengthen Out-of-Distribution Detection Capability with Progressive Self-Knowledge Distillation

## Dataset Preparation

The experiment is based on the following benchmarks provided by [OpenOOD v1.5](https://github.com/Jingkang50/OpenOOD):

> - ID: CIFAR-10
>      > Near-OOD: `CIFAR-100`, `TinyImageNet`;<br>
>      > Far-OOD: `MNIST`, `SVHN`, `Texture`, `Places365`;<br>
> - ID: CIFAR-100
>      > Near-OOD: `CIFAR-10`, `TinyImageNet`;<br>
>      > Far-OOD: `MNIST`, `SVHN`, `Texture`, `Places365`;<br>
> - ID: ImageNet-200
>      > Near-OOD: `SSB-hard`, `NINCO`;<br>
>      > Far-OOD: `iNaturalist`, `Texture`, `OpenImage-O`;<br>

The entire dataset preparation process can be automated by executing the following command:
```
sh ./scripts/download/download.sh
```

## Preliminaries
It is run under Ubuntu Linux 18.04 and Python 3.8.19 environment, and requires some packages to be installed.
* [PyTorch](https://pytorch.org/)
* [numpy](http://www.numpy.org/)

## Run

### 1. CIFAR-10 Benchmark

```
# Train
sh scripts/ood/pskd/cifar10_train_pskd.sh <GPU_ID>
# Test
sh cifar10_test.sh
```

### 2. CIFAR-100 Benchmark

```
# Train
sh scripts/ood/pskd/cifar100_train_pskd.sh <GPU_ID>
# Test
sh cifar100_test.sh
```

### 3. ImageNet-200 Benchmark

```
# Train
sh scripts/ood/pskd/imagenet200_train_pskd.sh <GPU_ID>
# Test
sh imagenet200_test.sh
```







