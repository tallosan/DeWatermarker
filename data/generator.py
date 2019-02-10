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

from PIL import Image


class DSGenerator:
    """
    Generates training & validation datasets for a specific watermark on
    a given set of images.
    """
    resampling_filter = Image.BICUBIC
    TRAINING_DIR = "data/training/"
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
        # Watermark each image, and create the corresponding datapoint.
        for image in images:
            raw_image = copy.deepcopy(image)
            watermarked_image = cls._add_watermark(
                watermark=watermark,
                image=raw_image
            )
            cls._save_datapoint(
                watermarked_image=watermarked_image,
                original_image=image
            )

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
                the image we're resizing to. No matter how we resize our
                watermark, the dimensions must be within this boundary, or
                the watermark will overflow from the base image.
            resize_ratio (int): The ratio to resize. Default is 2 (50%).
        """
        # Note, I'm using Bicubic interpolation here rather than the
        # defacto standard, Lanczos resampling, as the trade-off between
        # quality and speed simply makes sense. We don't need super HD
        # watermarks, and we __do__ want fast dataset generation.
        max_width, max_height = dim_boundary
        watermark.thumbnail(
            size=(max_width // resize_ratio, max_height // resize_ratio),
            resample=cls.resampling_filter
        )

        return watermark

    @classmethod
    def _save_datapoint(cls, watermarked_image, original_image):
        """
        Save a datapoint. Each datapoint is made of a 2-tuple, where
        one element is the watermarked image, an the other the original.
        Args:
            watermarked_image (Image): The watermarked image.
            original_image (Image): The original image that the watermarked
                image was generated from.
        """
        # Save the data-point pair, where the watermarked version will
        # have a 'WM' prefix in its filename.
        watermarked_fname, original_image_fname = cls._get_datapoint_fname(
            fname=original_image.filename
        )

        watermarked_image.save(fp=watermarked_fname)
        original_image.save(fp=original_image_fname)

    @classmethod
    def _get_datapoint_fname(cls, fname):
        """
        Generates filenames for a watermarked image and its original.
        Args:
            fname (str): The original 
        """
        fname = fname.split("/")[-1]
        watermarked_fname = "{fdir}{wm_prefix}_{fname}".format(
            fdir=cls.TRAINING_DIR,
            wm_prefix=cls.WATERMARK_PREFIX,
            fname=fname,
        )
        original_image_fname = "{fdir}{fname}".format(
            fdir=cls.TRAINING_DIR,
            fname=fname
        )
    
        return watermarked_fname, original_image_fname
