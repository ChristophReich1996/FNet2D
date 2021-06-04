from typing import Tuple

import torch
import torch.nn as nn
from torch.fft import fftn


class FNet2D(nn.Module):
    """
    This class implements the FNet 2D
    """

    def __init__(self,
                 channels: Tuple[Tuple[int, int], ...] = ((32, 64, True),
                                                          (64, 64, False),
                                                          (64, 256, True),
                                                          (256, 256, False),
                                                          (256, 256, False),
                                                          (256, 512, True),
                                                          (512, 512, False),
                                                          (512, 512, False),
                                                          (512, 1024, True)),
                 embedding_sizes: Tuple[Tuple[int, int]] = ((32, 32),
                                                            (16, 16),
                                                            (16, 16),
                                                            (8, 8),
                                                            (8, 8),
                                                            (8, 8),
                                                            (4, 4),
                                                            (4, 4),
                                                            (4, 4)),
                 out_features: int = 10,
                 no_fft: bool = False) -> None:
        """
        Constructor method
        :param channels: (Tuple[Tuple[int, int], ...]) Number of in and output channels used in each block
        :param embedding_sizes: (Tuple[int, int]) Embedding sizes to be used for each block
        :param out_features: (int) Output features (classes) to be utilized
        :param no_fft: (bool) If true no FFT is utilized (used for ablation)
        """
        # Call super constructor
        super(FNet2D, self).__init__()
        # Make initial convolutional stem
        self.convolution_stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels[0][0], kernel_size=(1, 1), bias=False)
        )
        # Init blocks
        self.blocks = nn.Sequential(
            *[FNet2D_module(in_channels=channel[0], out_channels=channel[1], hidden_channels=2 * channel[1],
                            downscale=channel[2], embedding_size=embedding_size)
              for channel, embedding_size in zip(channels, embedding_sizes)]
        )
        # Init final mapping
        self.final_mapping = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=channels[-1][-2], out_features=out_features, bias=False),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input image of the shape [batch size, channels, height, width]
        :return: (torch.Tensor) Output classification of the shape [batch size, classes]
        """
        # Initial mapping
        output = self.convolution_stem(input)
        # Perform forward pass of blocks
        output = self.blocks(output)
        # Perform final mapping
        output = self.final_mapping(output)
        return output


class FNet2D_module(nn.Module):
    """
    This class the core module of the FNet 2D inspired by:
    https://arxiv.org/pdf/2105.03824.pdf
    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, embedding_size: Tuple[int, int],
                 dropout: float = 0.2, downscale: bool = True, no_fft: bool = False) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param hidden_channels: (int) Number of channels in hidden linear (2D 1X1 convolution) layer
        :param embedding_size: (Tuple[int, int]) Size of the used embedding
        :param dropout: (float) Dropout rate
        :param downscale: (bool) If true pooling is utilized
        :param no_fft: (bool) If true no FFT is utilized (used for ablation)
        """
        # Call super constructor
        super(FNet2D_module, self).__init__()
        # Save parameter
        self.no_fft = no_fft
        # Init embedding
        self.embedding_vertical = nn.Parameter(torch.randn(1, 1, embedding_size[0]), requires_grad=True)
        self.embedding_horizontal = nn.Parameter(torch.randn(1, 1, embedding_size[1]), requires_grad=True)
        # Init normalization layers
        self.normalization_1 = nn.BatchNorm2d(num_features=in_channels, affine=True, track_running_stats=True)
        self.normalization_2 = nn.BatchNorm2d(num_features=out_channels, affine=True, track_running_stats=True)
        # Init feed forward network
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(1, 1), bias=False),
            nn.PReLU(num_parameters=hidden_channels),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False),
            nn.PReLU(num_parameters=out_channels),
            nn.Dropout2d(p=dropout)
        )
        # Init skip connection
        self.skip_connection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                         bias=False)
        # Init pooling layer
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) if downscale else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in_channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out_channels, height // 2, width // 2]
        """
        # Make embedding
        embedding = torch.einsum("ijk, ijl -> ijkl", self.embedding_vertical, self.embedding_horizontal)
        # Perform 3D fft
        if self.no_fft:
            output_fft = input + embedding
        else:
            output_fft = fftn(input + embedding, dim=(1, 2, 3), norm="ortho").real
        # Perform first normalization
        output_norm_1 = self.normalization_1(output_fft) + input + embedding
        # Perform feed forward network
        output_ff = self.feed_forward(output_norm_1)
        # Perform second normalization
        output_norm_2 = self.normalization_2(output_ff) + self.skip_connection(output_norm_1)
        # Perform pooling
        output_pooling = self.pooling(output_norm_2)
        return output_pooling
