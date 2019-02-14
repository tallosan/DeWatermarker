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
from torch.utils.data import DataLoader

from data import DeWatermarkerDataset
from autoencoder import ARCH0Autoencoder, ARCH1Autoencoder, ARCH2Autoencoder
from utils import display 


# Hyperparameters.
BATCH_SIZE = 4
SHUFFLE = True
NUM_WORKERS = 4
N_EPOCHS = 2000
N_BATCHES = 10
ETA = 1e-3


# Data setup.
dataset = DeWatermarkerDataset(root_dir="data/training/set.pkl")
INPT_SHAPE = dataset[0]["watermarked"].shape
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS
)


# Model setup. Note, we have the option to load in an existing model.
model = ARCH1Autoencoder(inpt_shape=INPT_SHAPE)
model.load()

# Loss function & optimizer setup.
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=ETA,
    weight_decay=1e-5
)


# Train.
# TODO: Make this a method on the Autoencoder class.
min_loss = float("inf")
for epoch in range(N_EPOCHS):
    for i_batch, sample_batched in enumerate(dataloader):
        watermarked = sample_batched["watermarked"]
        original = sample_batched["original"]
        original = model.resize(sample=original)
        # NOTE: This resizing operation is __inverting__ the colours.
        # It's fine given that our task is still to remove watermarks, but
        # we'll want to fix this going forwards.

        output = model(x=watermarked)
        loss = criterion(output, original)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # TODO: Move this into `debug()` method.
    if epoch == 0 or epoch % 100 == 0:
        print(">> epoch # {}: {}".format(epoch, loss.data))
        display(output)
        if loss < min_loss:
            print(">> updating weights.")
            model.save()
            min_loss = loss
