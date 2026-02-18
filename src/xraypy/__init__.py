# read version from installed package
from importlib.metadata import version
from .plot_image import plot_image

__version__ = version("xraypy")
