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

import _pickle as cPickle

from torch.utils.data import Dataset


class DeWatermarkerDataset(Dataset):
    """
    DeWatermarker project dataset.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir ()
            transform (callable): Optional ransform function to apply
                to samples.
        """
        self.root_dir = root_dir
        self.transform = transform
        with open(root_dir, "rb") as fp:
            self.dewatermarker_frame = cPickle.load(fp)

    def __len__(self):
        return len(self.dewatermarker_frame)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        Args:
            index (int): The index of the dataset element we're accessing.
        """
        # Get the sample, and apply any necessary transform (if any).
        sample = self.dewatermarker_frame[index]
        if self.transform:
            sample = self.transform(sample)

        return sample
