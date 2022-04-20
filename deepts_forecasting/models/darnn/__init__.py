import torch
from deepts_forecasting.models.base_model import BaseModel
from torch import nn


class DARNN(nn.Module):
    def __init__(self):
        super().__init__()
