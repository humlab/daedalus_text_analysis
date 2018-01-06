
import sys
import os

def _add_lib_path(some_relativepath):
    sys.path.insert(0, os.path.join(os.path.split(__file__)[0], some_relativepath))

add_lib_path = _add_lib_path
