import sys

if sys.version_info[:2] >= (3, 8):
    from importlib.metadata import PackageNotFoundError, version
else:
    from importlib_metadata import PackageNotFoundError, version

<<<<<<< HEAD
dist_name = "easyocr"
try:
    __version__ = version(dist_name)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .easyocr import Reader
=======
__version__ = '1.6.2'
>>>>>>> 06753992c0aa7b9c74f46ce558bba4ba5a28493b
