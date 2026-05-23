## ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection
This is the code for the paper "ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection (ICLR 2024)".

Abstract: *Post-hoc out-of-distribution (OOD) detection has garnered intensive attention in reliable machine learning. Many efforts have been dedicated to deriving score functions based on logits, distances, or rigorous data distribution assumptions to identify low-scoring OOD samples. Nevertheless, these estimate scores may fail to accurately reflect the true data density or impose impractical constraints. To provide a unified perspective on density-based score design, we propose a novel theoretical framework grounded in Bregman divergence, which extends distribution considerations to encompass an exponential family of distributions. Leveraging the conjugation constraint revealed in our theorem, we introduce a \textsc{ConjNorm} method, reframing density function design as a search for the optimal norm coefficient $p$ against the given dataset. In light of the computational challenges of normalization, we devise an unbiased and analytically tractable estimator of the partition function using the Monte Carlo-based importance sampling technique. Extensive experiments across OOD detection benchmarks empirically demonstrate that our proposed \textsc{ConjNorm} has established a new state-of-the-art in a variety of OOD detection setups, outperforming the current best method by up to 13.25$\%$ and 28.19$\%$ (FPR95) on CIFAR-100 and ImageNet-1K, respectively.*



## Usage

### 1. Dataset Preparation for ImageNet Experiment 

#### In-distribution dataset

##### ResNet50:
> python imagenet_resnet/feature_extraction.py

> python imagenet_resnet/feature_sort_resnet_50.py

Place the sorted features in the folder of cache/imagenet_feature_resnet_50_sorted

##### MobileNetv2:
> python imagenet_mobilenet/feature_extraction.py

> python imagenet_mobilenet/feature_sort_mobilenet_v2.py

Place the sorted features in the folder of cache/imagenet_feature_mobilenet_v2_sorted

#### Out-of-distribution dataset

We have curated 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./datasets/ood_data`.

### 2. Dataset Preparation for CIFAR Experiment 

#### In-distribution dataset

The downloading process will start immediately upon running. 

#### Out-of-distribution dataset

We provide links and instructions to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_datasets/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_datasets/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_datasets/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. 
* [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN`.
* [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `datasets/ood_datasets/LSUN_resize`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_datasets/iSUN`.

### 3.  Pre-trained model
Pre-trained models are placed in the `./checkpoints` folder.

## Demo
### 1. Demo code for Cifar-10 Experiment 
> python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN LSUN_resize iSUN dtd places365 --name densenet  --model-arch densenet --epochs 100

> python run_cifar_densenet.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN LSUN_resize iSUN dtd places365 --name densenet  --model-arch densenet

### 2. Demo code for Cifar-100 Experiment 
> python feat_extract.py --in-dataset CIFAR-100  --out-datasets SVHN LSUN LSUN_resize iSUN dtd places365 --name densenet  --model-arch densenet --epochs 100

> python run_cifar_densenet_c100.py --in-dataset CIFAR-100  --out-datasets SVHN LSUN LSUN_resize iSUN dtd places365 --name densenet  --model-arch densenet

### 3. Demo code for ImageNet Experiment on ResNet50
> python feat_extract_largescale.py --in-dataset imagenet  --out-datasets inat sun50 places50 dtd  --name resnet50  --model-arch resnet50

> python run_imagenet.py --in-dataset imagenet  --out-datasets inat sun50 places50 dtd  --name resnet50  --model-arch resnet50

### 4. Demo code for ImageNet Experiment on MobileNetv2
> python feat_extract_largescale.py --in-dataset imagenet  --out-datasets inat sun50 places50 dtd  --name mobilenetv2  --model-arch mobilenetv2

> python run_imagenet_mobilenet.py --in-dataset imagenet  --out-datasets inat sun50 places50 dtd  --name mobilenetv2  --model-arch mobilenetv2


