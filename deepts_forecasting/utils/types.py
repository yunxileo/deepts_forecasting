"""

"""

from typing import Mapping, Union

import torch

_NUMBER = Union[int, float]
_METRIC = Union[torch.Tensor, _NUMBER]
_METRIC_COLLECTION = Union[_METRIC, Mapping[str, _METRIC]]
