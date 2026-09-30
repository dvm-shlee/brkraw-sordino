"""BrkRaw SORDINO-ZTE converter hook package."""

from .hook import get_dataobj, get_dataobj_info
from .memguard import SordinoResourceError

__all__ = ["__version__", "get_dataobj", "get_dataobj_info", "SordinoResourceError"]

__version__ = "0.2.0"
