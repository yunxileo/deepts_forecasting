# """
# Implementation of complete structure of N-BEATS network.
# """
# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from typing import Dict, Union
# from deepts_forecasting.models.nbeats.sub_modules import NBEATSTrendBlock, NBEATSGenericBlock, NBEATSSeasonalBlock
# from deepts_forecasting.models.base_model import BaseModel
#
#
# class NBEATS(BaseModel):
#     def __init__(
#             self,
#             stack_types=['generic', 'generic'],
#             num_blocks=[3, 3],
#             num_block_layers=[3, 3],
#             widths=[512, 512],
#             sharing=[False, False],
#             expansion_coefficient_lengths=[10, 10],
#             prediction_length=14,
#             context_length=30,
#             dropout=0.1,
#         ):
#         """
#
#         Args:
#             stack_types (List[str]): type of each stack. the length of stack_types is the number of stacks. each element
#                 of list is the interpretation type of each stack
#             num_blocks (List[int]): number of blocks of each stack. the length of num_blocks is the number of stacks.
#                 each element of list is the number of blocks of each stack
#             num_block_layers (List[int]): depth of FC layers. the length of num_block_layers is the number of stacks.
#                 each element of list is the depth of FC layers of a block in each stack
#             widths (List[int]): width of FC layers. the length of widths is the number of stacks. each element of list
#                 is the width of FC layers of a block in each stack
#             sharing (bool): if sharing expansion coefficients. theta_b is the same as theta_f is True
#             expansion_coefficient_lengths (List[int]): length of theta_f and theta_b of each stack
#             prediction_length (int): length of forecast output
#             context_length (int): length of backcast output
#             dropout (float): dropout
#             device (Union[torch.device, str]): device
#         """
#         super(NBEATS, self).__init__()
#
#         self.stack_types = stack_types
#         self.num_blocks = num_blocks
#         self.num_block_layers = num_block_layers
#         self.widths = widths
#         self.sharing = sharing
#         self.expansion_coefficient_lengths = expansion_coefficient_lengths
#         self.prediction_length = prediction_length
#         self.context_length = context_length
#         self.dropout = dropout
#
#         self.net_blocks = nn.ModuleList()
#         for stack_id, stack_type in enumerate(stack_types):
#             for _ in range(num_blocks[stack_id]):
#                 if stack_type == 'generic':
#                     net_block = NBEATSGenericBlock(
#                         units=self.widths[stack_id],
#                         thetas_dim=self.expansion_coefficient_lengths[stack_id],
#                         num_block_layers=self.num_block_layers[stack_id],
#                         backcast_length=self.context_length,
#                         forecast_length=self.prediction_length,
#                         dropout=self.dropout
#                     )
#                 elif stack_type == 'seasonality':
#                     net_block = NBEATSSeasonalBlock(
#                         units=self.widths[stack_id],
#                         num_block_layers=self.num_block_layers[stack_id],
#                         backcast_length=self.context_length,
#                         forecast_length=self.prediction_length,
#                         min_period=self.expansion_coefficient_lengths[stack_id],
#                         dropout=self.dropout
#                     )
#                 elif stack_type == 'trend':
#                     net_block = NBEATSTrendBlock(
#                         units=self.widths[stack_id],
#                         thetas_dim=self.expansion_coefficient_lengths[stack_id],
#                         num_block_layers=self.num_block_layers[stack_id],
#                         backcast_length=self.context_length,
#                         forecast_length=self.prediction_length,
#                         dropout=self.dropout
#                     )
#                 else:
#                     raise ValueError(f"Unknown stack type {stack_type}")
#
#                 self.net_blocks.append(net_block)
#
#     def forward(self, x: Dict[str, torch.Tensor]):
#         device = x['encoder_target'].device
#
#         x = x['encoder_target'].squeeze(-1)
#
#         timesteps = self.context_length + self.prediction_length
#
#         generic_forecast = [torch.zeros((x.size(0), timesteps),
#                                         dtype=torch.float32,
#                                         device=device)]
#         trend_forecast = [torch.zeros((x.size(0), timesteps),
#                                       dtype=torch.float32,
#                                       device=device)]
#         seasonal_forecast = [torch.zeros((x.size(0), timesteps),
#                                          dtype=torch.float32,
#                                          device=device)]
#         forecast = torch.zeros((x.size(0), self.prediction_length),
#                                dtype=torch.float32,
#                                device=device)
#
#         backcast = x
#
#         for i, block in enumerate(self.net_blocks):
#             backcast_block, forecast_block = block(backcast)
#
#             full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)
#
#             if isinstance(block, NBEATSTrendBlock):
#                 trend_forecast.append(full)
#             elif isinstance(block, NBEATSSeasonalBlock):
#                 seasonal_forecast.append(full)
#             else:
#                 generic_forecast.append(full)
#
#             backcast = (backcast - backcast_block)
#             forecast = forecast + forecast_block
#
#         return forecast
