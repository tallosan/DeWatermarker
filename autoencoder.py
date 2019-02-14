# MIT License
# 
# Copyright (c) 2019 Andrew Tallos
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ================================================================

import torch
import torch.nn as nn


class BaseAutoencoder(nn.Module):
    """
    Provides some common functionality across the different autoencoder
    model architectures.
    """

    def forward(self, x):
        """
        Perform the forward pass on the given input.
        Args:
            x (Tensor): The input to perform the forward pass on.
        """
        x = self.resize(x)
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        return decoded_x

    def load(self):
        """
        Load in any existing weights belonging to this model.
        """
        try:
            self.load_state_dict(torch.load(self.FPATH))
            self.eval()
        except FileNotFoundError:
            msg = "No existing model to initialize from. Creating new one ..."
            print(msg)

    def save(self):
        """
        Save the current state of this model.
        """
        torch.save(self.state_dict(), self.FPATH)

    def resize(self, sample):
        """
        Resize a sample so that it can be inputted to a PyTorch
        Conv2D layer. We need to do this, as PyTorch our input
        is expected in the following shape:
            (batch_size, n_channels, height, width)
        """
        return sample.permute(0, 3, 1, 2).type("torch.FloatTensor")


class ARCH0Autoencoder(BaseAutoencoder):
    """
    First autoencoder architecture.
    We'll go with a shallow 1-layer convolutional autoencoder.
    """
    KERNEL_SIZE = 3
    STRIDE = 1
    FPATH = "arch_0.pt"

    def __init__(self, inpt_shape):
        super().__init__()
        _, _, inpt_channels = inpt_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=inpt_channels,
                out_channels=6,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE
            ),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=6,
                out_channels=3,
                kernel_size=self.KERNEL_SIZE
            ),
            nn.ReLU(inplace=True)
        )


class ARCH1Autoencoder(BaseAutoencoder):
    """
    Second autoencoder architecture. This will be a 2-layer convolutional
    autoencoder model.
    """
    KERNEL_SIZE = 3
    STRIDE = 1
    FPATH = "arch_1.pt"

    def __init__(self, inpt_shape):
        super().__init__()
        _, _, inpt_channels = inpt_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=inpt_channels,
                out_channels=6,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=6,
                out_channels=12,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=12,
                out_channels=24,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE
            ),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=24,
                out_channels=12,
                kernel_size=self.KERNEL_SIZE
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=12,
                out_channels=6,
                kernel_size=self.KERNEL_SIZE
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=6,
                out_channels=inpt_channels,
                kernel_size=self.KERNEL_SIZE
            ),
            nn.ReLU(inplace=True)
        )


class ARCH2Autoencoder(BaseAutoencoder):
    """
    Second autoencoder architecture. This will be a 2-layer convolutional
    autoencoder model.
    """
    KERNEL_SIZE = 3
    STRIDE = 1
    FPATH = "arch_2.pt"

    def __init__(self, inpt_shape):
        super().__init__()
        _, _, inpt_channels = inpt_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=inpt_channels,
                out_channels=64,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE
            ),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=self.KERNEL_SIZE
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=self.KERNEL_SIZE
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=inpt_channels,
                kernel_size=self.KERNEL_SIZE
            ),
            nn.ReLU(inplace=True)
        )
