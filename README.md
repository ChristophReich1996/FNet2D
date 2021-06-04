# FNet 2D: Scaling Fourier Transform Token Mixing To Vision

This repository implements a 2D version of the FNet [1] by Lee-Thorp et al. for image classification (CIFAR-10). The original FNet replaces 
costly self-attention layers with simple but effective Fourier transforms. The resulting FNet model achieves 
competitive results on the GLUE benchmark while being highly more computationally efficient than BERT.

This repository scales the FNet encoder to images. The resulting FNet 2D utilizes an FFT over the feature dimension as 
well as over the spatial dimensions. The performance of the FNet 2D is evaluated on the CIFAR-10 dataset for image 
classification. FNet 2D falls somewhat short in terms of classification accuracy but shows a very high computational 
efficiency. This approach may help further research, especially applying FNet 2D to other 2D data than natural images, 
such as spectrograms for (ECG) signal classification could be interesting.

## Installation

This implementation uses PyTorch 1.8.1 and is tested on an Ubuntu system. To install all required packages run:

``shell script
pip install -r requirements.txt
```

## Results

## Usage



```bibtex
[1] @article{Lee2021,
        title={{FNet: Mixing Tokens with Fourier Transforms}},
        author={Lee-Thorp, James and Ainslie, Joshua and Eckstein, Ilya and Ontanon, Santiago},
        journal={arXiv preprint arXiv:2105.03824},
        year={2021}
}
```