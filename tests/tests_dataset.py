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

from unittest import TestCase, mock

from data import DeWatermarkerDataset


class TestDeWatermarkerDataset(TestCase):
    """
    Test cases for the DeWatermarker dataset.
    """

    def setUp(self):
        self.root_dir = "/test_dir"
        self.transform = None

    @mock.patch("builtins.open")
    @mock.patch("data.generator.cPickle.load")
    def test_dataset__init__(self, mock_cPickle, mock_open):
        MOCK_FILE = "<MOCK_FILE>"
        mock_open.return_value.__enter__.return_value = MOCK_FILE
        MOCK_DATAFRAME = "<MOCK_DATAFRAME>"
        mock_cPickle.return_value = MOCK_DATAFRAME

        dataset = DeWatermarkerDataset(root_dir=self.root_dir)
        mock_open.assert_called_with(self.root_dir, "rb")
        mock_cPickle.assert_called_with(MOCK_FILE)
        self.assertEqual(dataset.dewatermarker_frame, MOCK_DATAFRAME)

    @mock.patch.object(DeWatermarkerDataset, "__init__", return_value=None)
    def test_dataset__len__(self, mock__init__):
        """
        Ensure that we can get the number of samples in the dataset.
        """
        dataset = DeWatermarkerDataset(root_dir=self.root_dir)
        MOCK_DATAFRAME = [
            {"test_a": "entry_a"},
            {"test_b": "entry_b"},
            {"test_c": "entry_c"}
        ]
        dataset.dewatermarker_frame = MOCK_DATAFRAME

        self.assertEqual(len(dataset), len(MOCK_DATAFRAME))

    @mock.patch.object(DeWatermarkerDataset, "__init__", return_value=None)
    def test_data__getitem(self, mock__init__):
        """
        Ensure that we can get a sample from our dataset.
        """
        dataset = DeWatermarkerDataset(root_dir=self.root_dir)
        MOCK_DATAFRAME = [
            {"test_a": "entry_a"},
            {"test_b": "entry_b"},
            {"test_c": "entry_c"}
        ]
        MOCK_SAMPLE_INDEX = 0
        dataset.dewatermarker_frame = MOCK_DATAFRAME
        self.assertEqual(
            dataset.dewatermarker_frame[MOCK_SAMPLE_INDEX],
            MOCK_DATAFRAME[MOCK_SAMPLE_INDEX]
        )

        # Ensure that if we set a `transform` function, then it will be
        # called on the sample correctly.
        dataset.transform = mock.MagicMock()
        with mock.patch.object(
            dataset,
            "transform",
            warps=dataset.transform
        ) as mock_transform_func:
            dataset[MOCK_SAMPLE_INDEX]
            mock_transform_func.assert_called_with(
                MOCK_DATAFRAME[MOCK_SAMPLE_INDEX]
            )
