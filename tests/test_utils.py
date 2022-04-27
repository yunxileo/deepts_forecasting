import pytest
import sys
import os
from deepts_forecasting.utils.utils import get_embedding_size

work_dir = os.getcwd()
sys.path.append(work_dir)


def test_get_embedding_size():
    dim = get_embedding_size(10)
    assert dim == 6, "The embedding size should be 6."
