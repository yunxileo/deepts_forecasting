"""
The temporal fusion transformer is a powerful predictive model for forecasting timeseries
"""
from copy import copy
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from deepts_forecasting.models.modules import MultiEmbedding
from torch import nn


class TemporalFusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_embeddings = MultiEmbedding()
