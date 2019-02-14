#
# Utility functions.
#
# @author :: tallosan
# ================================================================

from torchvision import transforms


def display(image):
    """
    Display an image.
    """
    display = image.resize(3, 1000, 800)
    transforms.ToPILImage(mode="RGB")(display).show()

