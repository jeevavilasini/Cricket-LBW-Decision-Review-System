"""Root conftest so `pytest` can import the `drs` package without installation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
