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

import copy
import numpy
import _pickle as cPickle

from PIL import Image


class DSGenerator:
    """
    Generates training & validation datasets for a specific watermark on
    a given set of images.
    """
    RESAMPLING_FILTER = Image.BICUBIC
    TRAINING_FNAME = "data/training/set.pkl"
    WATERMARK_PREFIX = "wm"

    @classmethod
    def generate_dataset(cls, watermark, images):
        """
        Generate a dataset for the given watermark on the given images.
        Each entry in our dataset will consist of two elements -- the
        raw image, and a version of the image with the given watermark
        on it.
        Args:
            watermark (Image): The watermark to generate the dataset for.
            images (list of Image): The images that will make up our
                new dataset.
        """
        # Create a dataset, and save it.
        training_set = []
        for image in images:
            raw_image = copy.deepcopy(image)
            watermarked_image = cls._add_watermark(
                watermark=watermark,
                image=raw_image
            )

            datapoint = cls._create_datapoint(
                watermarked=watermarked_image,
                original=image
            )
            training_set.append(datapoint)

        cls._save_dataset(dataset=training_set, fname=cls.TRAINING_FNAME)

    @classmethod
    def _add_watermark(cls, watermark, image):
        """
        Add a watermark to a given image.
        Args:
            watermark (Image): The watermark to add to the given image.
            image (Image): The image to add the watermark to.
        """
        # We'll need to resize our logo to ensure that it actually fits
        # on the given image.
        cls._resize_watermark(
            watermark=watermark,
            dim_boundary=image.size,
            resize_ratio=2
        )

        # TODO: Perform this step with variable positions, rather than
        # one hard set.
        POSITION = (0, 0)
        image.paste(im=watermark, box=POSITION, mask=watermark)
        return image

    @classmethod
    def _resize_watermark(cls, watermark, dim_boundary, resize_ratio=2):
        """
        Resize the given watermark so that it fits within a set target
        dimension. Note, we'll want to ensure that we keep the aspect
        ratio of the watermark.
        Args:
            watermark (Image): The watermark to resize.
            dim_boundary (tuple of int): These are just the dimensions of
                the image we're adding the watermark to. No matter how we
                resize our watermark, the dimensions must be within this
                boundary, or the watermark will overflow from the target image.
            resize_ratio (int): The ratio to resize. Default is 2 (50%).
        """
        # Note, I'm using Bicubic interpolation here rather than the
        # defacto standard, Lanczos resampling, as the trade-off between
        # quality and speed simply makes sense. We don't need super HD
        # watermarks, and we __do__ want fast dataset generation.
        max_width, max_height = dim_boundary
        watermark.thumbnail(
            size=(max_width // resize_ratio, max_height // resize_ratio),
            resample=cls.RESAMPLING_FILTER
        )

        return watermark

    @classmethod
    def _create_datapoint(cls, watermarked, original):
        return {
            "watermarked": numpy.asarray(watermarked),
            "original": numpy.asarray(original)
        }

    @classmethod
    def _save_dataset(cls, dataset, fname):
        """
        Save a datapoint. Each datapoint is made of a 2-tuple, where
        one element is the watermarked image, and the other the original.
        Args:
            dataset (list of dict of Image): A list of image objects in a
                dataset, where each element is a datapoint containing a
                watermarked image, and its original.
            fname (str): The filename to use for the dataset.
        """
        with open(fname, "wb") as fp:
            cPickle.dump(dataset, fp)
