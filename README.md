# DeWatermarker

The goal of this project is to write software that can remove watermarks from images, using the power of autoencoders.

The actual implementation can be divided into two stages -- data-set generation, & encoder training.

## Data-Set Generation:

First, I'll need to implement some method of dataset generation.

This essentially amounts to choosing a watermark to test with (I've gone with the standard Getty watermark), and
then writing some logic to add this watermark to some images (currently just Edvard Munch's 'The Scream') that
I'd like to build my dataset from. Each data point in my dataset will consist of a 2-tuple, where we've got the
raw image (sans watermark) that we'd like to recreate, and an image with a watermark that we'll be inputting into
our autoencoder.

## Autoencoder Generation:
Second, I'll start training an autoencoder on the generated dataset to do the actual watermark removal.

This should be interesting, as there's a lot of literature to be inspired by, and a lot of room for creativity
w.r.t. architectural decisions. I'm planning to use the PyTorch library for the actual implementation, as I think
it's beautifully designed.

## Project Lifecycle:

I'm going to first ensure that this __can__ work. The most obvious way to me to do this is to build out the data-set
using only one data point, and then to overfit the autoencoder on it.

The next stage in the projet life-cycle will be scaling this up so that it can work on a range of different images, and
different watermarks. I'll also add some client-facing UI, as it'd be cool to make this accessible to more people.
