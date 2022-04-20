import os
import pathlib
import pickle
from typing import Any, Dict, List, Set, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import yaml


def to_list(value: Any) -> List[Any]:
    """
    Convert value or list to list of values.
    If already list, return object directly

    注：删除了rnn条件
    Args:
        value (Any): value to convert

    Returns:
        List[Any]: list of values
    """
    if isinstance(value, (tuple, list)):
        return value
    else:
        return [value]


def to_singular(x: List[Any]):
    if len(x) == 1:
        return x[0]
    else:
        raise ValueError("Input should be a list containing only one entry")


def get_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).

    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.

    Returns:
        int: embedding size
    """
    if n > 2:
        return min(round(1.6 * n**0.56), max_size)
    else:
        return 1


def save_dict(path, name, _dict) -> None:
    """

    Args:
        path:
        name:
        _dict:
    """
    with open(path / f"{name}.pickle", "wb") as handle:
        pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def check_for_nonfinite(
    tensor: torch.Tensor, names: Union[str, List[str]]
) -> torch.Tensor:
    """
    Check if 2D tensor contains NAs or inifinite values.

    Args:
        names (Union[str, List[str]]): name(s) of column(s) (used for error messages)
        tensor (torch.Tensor): tensor to check

    Returns:
        torch.Tensor: returns tensor if checks yield no issues
    """
    if isinstance(names, str):
        names = [names]
        assert tensor.ndim == 1
        nans = (~torch.isfinite(tensor).unsqueeze(-1)).sum(0)
    else:
        assert tensor.ndim == 2
        nans = (~torch.isfinite(tensor)).sum(0)
    for name, na in zip(names, nans):
        if na > 0:
            raise ValueError(
                f"{na} ({na/tensor.size(0):.2%}) of {name} "
                "values were found to be NA or infinite (even after encoding). NA values are not allowed "
                "`allow_missing_timesteps` refers to missing rows, not to missing values. Possible strategies to "
                f"fix the issue are (a) dropping the variable {name}, "
                "(b) using `NaNLabelEncoder(add_nan=True)` for categorical variables, "
                "(c) filling missing values and/or (d) optionally adding a variable indicating filled values"
            )
    return tensor


class OutputMixIn:
    """
    MixIn to give namedtuple some access capabilities of a dictionary
    """

    def __getitem__(self, k):
        if isinstance(k, str):
            return getattr(self, k)
        else:
            return super().__getitem__(k)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def items(self):
        return zip(self._fields, self)

    def keys(self):
        return self._fields


def move_to_device(
    x: Union[
        Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
        torch.Tensor,
        List[torch.Tensor],
        Tuple[torch.Tensor],
    ],
    device: Union[str, torch.DeviceObjType],
) -> Union[
    Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
    torch.Tensor,
    List[torch.Tensor],
    Tuple[torch.Tensor],
]:
    """
    Move object to device.

    Args:
        x (dictionary of list of tensors): object (e.g. dictionary) of tensors to move to device
        device (Union[str, torch.DeviceObjType]): device, e.g. "cpu"

    Returns:
        x on targeted device
    """
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(x, dict):
        for name in x.keys():
            x[name] = move_to_device(x[name], device=device)
    elif isinstance(x, OutputMixIn):
        for xi in x:
            move_to_device(xi, device=device)
        return x
    elif isinstance(x, torch.Tensor) and x.device != device:
        x = x.to(device)
    elif isinstance(x, (list, tuple)) and x[0].device != device:
        x = [move_to_device(xi, device=device) for xi in x]
    return x


def create_mask(
    size: int, lengths: torch.LongTensor, inverse: bool = False
) -> torch.BoolTensor:
    """
    Create boolean masks of shape len(lenghts) x size.

    An entry at (i, j) is True if lengths[i] > j.

    Args:
        size (int): size of second dimension
        lengths (torch.LongTensor): tensor of lengths
        inverse (bool, optional): If true, boolean mask is inverted. Defaults to False.

    Returns:
        torch.BoolTensor: mask
    """

    if inverse:  # return where values are
        return torch.arange(size, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
    else:  # return where no values are
        return torch.arange(size, device=lengths.device).unsqueeze(
            0
        ) >= lengths.unsqueeze(-1)


def fill_missing_values(
    data: pd.DataFrame,
    reals: Union[List[str], Set[str]],
    categoricals: Union[List[str], Set[str]],
    method: str = "zero",
) -> pd.DataFrame:
    """
    Args:
        data (pd.DataFrame):
        reals (Union[str, List[str]]):
        categoricals (Union[str, List[str]]):
        method (str):

    Returns:
        pd.DataFrame: data without missing values.
    """
    if method == "mean":
        for i in reals:
            assert data[i].dtypes.kind in (
                "i",
                "f",
            ), f"Feature '{i}' is not numeric type."
            data[i].fillna(data[i].mean(skipna=True))
    elif method == "zero":
        for i in reals:
            assert data[i].dtypes.kind in (
                "i",
                "f",
            ), f"Feature '{i}' is not numeric type."
            data[i].fillna(0)
    else:
        raise ValueError("Missing value imputation method should be 'zero' or 'mean'!")

    for j in categoricals:
        assert isinstance(j, str), f"Feature '{j}' is not string type."
        data[j].fillna("other")

    return data


def load_config(config_path: str, config_name: str):
    """
    Load config file from path.

    Args:
        config_path (str): path saving config file
        config_name (str): name of config file

    """
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config


class TimeDistributedEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, *args, batch_first: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return super().forward(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        y = super().forward(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class MultiEmbedding(nn.Module):

    concat_output: bool

    def __init__(
        self,
        embedding_sizes: Union[
            Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]]
        ],
        x_categoricals: List[str] = None,
        categorical_groups: Dict[str, List[str]] = {},
        embedding_paddings: List[str] = [],
        max_embedding_size: int = None,
    ):
        """Embedding layer for categorical variables including groups of categorical variables.

        Enabled for static and dynamic categories (i.e. 3 dimensions for batch x time x categories).

        Args:
            embedding_sizes (Union[Dict[str, Tuple[int, int]], Dict[str, int], List[int], List[Tuple[int, int]]]):
                either

                * dictionary of embedding sizes, e.g. ``{'cat1': (10, 3)}``
                  indicates that the first categorical variable has 10 unique values which are mapped to 3 embedding
                  dimensions. Use :py:func:`~pytorch_forecasting.utils.get_embedding_size` to automatically obtain
                  reasonable embedding sizes depending on the number of categories.
                * dictionary of categorical sizes, e.g. ``{'cat1': 10}`` where embedding sizes are inferred by
                  :py:func:`~pytorch_forecasting.utils.get_embedding_size`.
                * list of embedding and categorical sizes, e.g. ``[(10, 3), (20, 2)]`` (requires ``x_categoricals`` to
                  be empty)
                * list of categorical sizes where embedding sizes are inferred by
                  :py:func:`~pytorch_forecasting.utils.get_embedding_size` (requires ``x_categoricals`` to be empty).

                If input is provided as list, output will be a single tensor of shape batch x (optional) time x
                sum(embedding_sizes). Otherwise, output is a dictionary of embedding tensors.
            x_categoricals (List[str]): list of categorical variables that are used as input.
            categorical_groups (Dict[str, List[str]]): dictionary of categories that should be summed up in an
                embedding bag, e.g. ``{'cat1': ['cat2', 'cat3']}`` indicates that a new categorical variable ``'cat1'``
                is mapped to an embedding bag containing the second and third categorical variables.
                Defaults to empty dictionary.
            embedding_paddings (List[str]): list of categorical variables for which the value 0 is mapped to a zero
                embedding vector. Defaults to empty list.
            max_embedding_size (int, optional): if embedding size defined by ``embedding_sizes`` is larger than
                ``max_embedding_size``, it will be constrained. Defaults to None.
        """
        super().__init__()
        if isinstance(embedding_sizes, dict):
            self.concat_output = False  # return dictionary of embeddings
            # conduct input data checks
            assert x_categoricals is not None, "x_categoricals must be provided."
            categorical_group_variables = [
                name for names in categorical_groups.values() for name in names
            ]
            if len(categorical_groups) > 0:
                assert all(
                    name in embedding_sizes for name in categorical_groups
                ), "categorical_groups must be in embedding_sizes."
                assert not any(
                    name in embedding_sizes for name in categorical_group_variables
                ), "group variables in categorical_groups must not be in embedding_sizes."
                assert all(
                    name in x_categoricals for name in categorical_group_variables
                ), "group variables in categorical_groups must be in x_categoricals."
            assert all(
                name in embedding_sizes
                for name in embedding_sizes
                if name not in categorical_group_variables
            ), (
                "all variables in embedding_sizes must be in x_categoricals - but only if"
                "not already in categorical_groups."
            )
        else:
            assert (
                x_categoricals is None and len(categorical_groups) == 0
            ), "If embedding_sizes is not a dictionary, categorical_groups and x_categoricals must be empty."
            # number embeddings based on order
            embedding_sizes = {
                str(name): size for name, size in enumerate(embedding_sizes)
            }
            x_categoricals = list(embedding_sizes.keys())
            self.concat_output = True

        # infer embedding sizes if not determined
        self.embedding_sizes = {
            name: (size, get_embedding_size(size)) if isinstance(size, int) else size
            for name, size in embedding_sizes.items()
        }
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.max_embedding_size = max_embedding_size
        self.x_categoricals = x_categoricals

        self.init_embeddings()

    def init_embeddings(self):
        self.embeddings = nn.ModuleDict()
        for name in self.embedding_sizes.keys():
            embedding_size = self.embedding_sizes[name][1]
            if self.max_embedding_size is not None:
                embedding_size = min(embedding_size, self.max_embedding_size)
            # convert to list to become mutable
            self.embedding_sizes[name] = list(self.embedding_sizes[name])
            self.embedding_sizes[name][1] = embedding_size
            if name in self.categorical_groups:  # embedding bag if related embeddings
                self.embeddings[name] = TimeDistributedEmbeddingBag(
                    self.embedding_sizes[name][0],
                    embedding_size,
                    mode="sum",
                    batch_first=True,
                )
            else:
                if name in self.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.embeddings[name] = nn.Embedding(
                    self.embedding_sizes[name][0],
                    embedding_size,
                    padding_idx=padding_idx,
                )

    def names(self):
        return list(self.keys())

    def items(self):
        return self.embeddings.items()

    def keys(self):
        return self.embeddings.keys()

    def values(self):
        return self.embeddings.values()

    def __getitem__(self, name: str):
        return self.embeddings[name]

    @property
    def input_size(self) -> int:
        return len(self.x_categoricals)

    @property
    def output_size(self) -> Union[Dict[str, int], int]:
        if self.concat_output:
            return sum([s[1] for s in self.embedding_sizes.values()])
        else:
            return {name: s[1] for name, s in self.embedding_sizes.items()}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): input tensor of shape batch x (optional) time x categoricals in the order of
                ``x_categoricals``.

        Returns:
            Union[Dict[str, torch.Tensor], torch.Tensor]: dictionary of category names to embeddings
                of shape batch x (optional) time x embedding_size if ``embedding_size`` is given as dictionary.
                Otherwise, returns the embedding of shape batch x (optional) time x sum(embedding_sizes).
                Query attribute ``output_size`` to get the size of the output(s).
        """
        input_vectors = {}
        for name, emb in self.embeddings.items():
            if name in self.categorical_groups:
                input_vectors[name] = emb(
                    x[
                        ...,
                        [
                            self.x_categoricals.index(cat_name)
                            for cat_name in self.categorical_groups[name]
                        ],
                    ]
                )
            else:
                input_vectors[name] = emb(x[..., self.x_categoricals.index(name)])

        if self.concat_output:  # concatenate output
            return torch.cat(list(input_vectors.values()), dim=-1)
        else:
            return input_vectors
