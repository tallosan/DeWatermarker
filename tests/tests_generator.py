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

from unittest import mock, TestCase
from PIL import Image

from data import DSGenerator


class TestDSGenerator(TestCase):
    """
    Test suite for the data-set generator.
    """
    def setUp(self):
        TEST_WATERMARK_PATH = "tests/images/test-watermark.png"
        self.watermark = Image.open(TEST_WATERMARK_PATH)

        TEST_IMG_PATH = "tests/images/test-image.jpg"
        self.primary_image = Image.open(TEST_IMG_PATH)
        self.N_IMAGES = 3
        self.images = [self.primary_image for _ in range(self.N_IMAGES)]

    @mock.patch("data.generator.DSGenerator._add_watermark")
    @mock.patch("data.generator.DSGenerator._save_datapoint")
    def test_generate_dataset(self, mock_save_dp, mock_add_watermark):
        """
        Ensure that the dataset generation method calls the correct
        methods, and functions as expected.
        """
        # We'll mock this so as to not clutter the `mock_calls` attribute
        # of our `mock_add_watermark` object in the second set of tests below.
        MOCK_ADD_WM = "<MOCK_ADD_WATERMARK>"
        mock_add_watermark.return_value = MOCK_ADD_WM
        DSGenerator.generate_dataset(
            watermark=self.watermark,
            images=self.images
        )
        for mock_save_call in mock_save_dp.mock_calls:
            _, _, kwargs = mock_save_call
            self.assertEqual(kwargs["watermarked_image"], MOCK_ADD_WM)
            self.assertEqual(kwargs["original_image"], self.primary_image)

        # We'll want to check that we've created a dataset point for each
        # image in our image set, and that each dataset point is created
        # correctly, as per the corresponding method call/s.
        self.assertEqual(len(mock_add_watermark.mock_calls), self.N_IMAGES)
        for mock_add_watermark_call in mock_add_watermark.mock_calls:
            _, _, kwargs = mock_add_watermark_call
            self.assertEqual(kwargs["watermark"], self.watermark)
            self.assertEqual(kwargs["image"], self.primary_image)

    def test__save_datapoint(self):
        """
        Ensure that we can save a datapoint.
        """
        mocked_watermark = copy.deepcopy(self.primary_image)
        with mock.patch.object(
            mocked_watermark,
            "save",
            wraps=mocked_watermark.save
        ) as mock_watermark_save:
            with mock.patch.object(
                self.primary_image,
                "save",
                wraps=self.primary_image.save
            ) as mock_original_save:
                DSGenerator._save_datapoint(
                    watermarked_image=mocked_watermark,
                    original_image=self.primary_image
                )
                wm_fname, org_fname = DSGenerator._get_datapoint_fname(
                    fname=self.primary_image.filename
                )

                mock_watermark_save.assert_called_with(fp=wm_fname)
                mock_original_save.assert_called_with(fp=org_fname)

    @mock.patch("data.generator.DSGenerator._resize_watermark")
    def test__add_watermark(self, mock_wm):
        """
        Ensure that we can add watermarks to our images.
        """
        # We'll need to test that the watermark resized correclty, that
        # it is 'pasted' onto our image, & that the image is saved.
        with mock.patch.object(
            self.primary_image,
            "paste",
            wraps=self.primary_image.paste
        ) as mock_paste:
            DSGenerator._add_watermark(self.watermark, self.primary_image)
            mock_wm.assert_called_with(
                watermark=self.watermark,
                dim_boundary=self.primary_image.size,
                resize_ratio=2
            )
            mock_paste.assert_called_with(
                im=self.watermark,
                box=(0, 0),
                mask=self.watermark
            )

    def test__resize_watermark(self):
        """
        Ensure that we can resize our watermarks.
        """
        # Note, we really just need to test that the `Image.thumbnail()`
        # method is called correctly.
        with mock.patch.object(
            self.watermark,
            "thumbnail",
            wraps=self.watermark.thumbnail
        ) as mock_thumbnail:
            test_resize_ratio = 2
            DSGenerator._resize_watermark(
                watermark=self.watermark,
                dim_boundary=self.primary_image.size,
                resize_ratio=test_resize_ratio
            )

            test_width, test_height = self.primary_image.size
            mock_thumbnail.assert_called_with(
                size=(
                    test_width // test_resize_ratio,
                    test_height // test_resize_ratio
                ),
                resample=DSGenerator.resampling_filter
            )
