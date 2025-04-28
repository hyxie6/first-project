from .__version__ import __version__
from .__git_hash__ import __git_hash_str__


def info():
    print("First project with version = {}, git hash = {}".format(
        __version__, __git_hash_str__)
    )
