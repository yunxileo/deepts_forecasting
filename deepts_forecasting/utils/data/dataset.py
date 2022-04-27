"""
Dataset for time series forecasting.
"""
import inspect
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from deepts_forecasting.utils import (
    check_for_nonfinite,
    fill_missing_values,
    to_list,
    to_singular,
)
from deepts_forecasting.utils.data.encoders import (
    EncoderNormalizer,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

# from torch.utils.data.sampler import Sampler
from torch.nn.utils import rnn
from torch.utils.data import DataLoader, Dataset


def _find_end_indices(
    diffs: np.ndarray, max_lengths: np.ndarray, min_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify end indices in series even if some values are missing.

    Args:
        diffs (np.ndarray): array of differences to next time step. nans should be filled up with ones
        max_lengths (np.ndarray): maximum length of sequence by position.
        min_length (int): minimum length of sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray]: tuple of arrays where first is end indices and second is list of start
            and end indices that are currently missing.
    """
    missing_start_ends = []
    end_indices = []
    length = 1
    start_idx = 0
    max_idx = len(diffs) - 1
    max_length = max_lengths[start_idx]

    for idx, diff in enumerate(diffs):
        if length >= max_length:
            while length >= max_length:
                if length == max_length:
                    end_indices.append(idx)
                else:
                    end_indices.append(idx - 1)
                length -= diffs[start_idx]
                if start_idx < max_idx:
                    start_idx += 1
                max_length = max_lengths[start_idx]
        elif length >= min_length:
            missing_start_ends.append([start_idx, idx])
        length += diff
    if len(missing_start_ends) > 0:  # required for numba compliance
        return np.asarray(end_indices), np.asarray(missing_start_ends)
    else:
        return np.asarray(end_indices), np.empty((0, 2), dtype=np.int64)


try:
    import numba

    _find_end_indices = numba.jit(nopython=True)(_find_end_indices)
except ImportError:
    pass

NORMALIZER = Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer]


class TimeSeriesDataSet(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        time_idx: Union[str, List[str]],
        target: Union[str, List[str]],
        group_ids: Union[str, List[str]],
        max_encoder_length: int = 30,
        min_encoder_length: int = None,
        min_prediction_idx: int = None,
        max_prediction_length: int = 1,
        min_prediction_length: int = None,
        static_categoricals: Union[str, List[str]] = [],
        static_reals: Union[str, List[str]] = [],
        time_varying_known_categoricals: Union[str, List[str]] = [],
        time_varying_known_reals: Union[str, List[str]] = [],
        time_varying_unknown_categoricals: Union[str, List[str]] = [],
        time_varying_unknown_reals: Union[str, List[str]] = [],
        variable_groups: Dict[str, List[int]] = {},
        lags: Dict[str, List[int]] = {},
        categorical_encoders: Dict[str, BaseEstimator] = {},
        scalers: Dict[str, BaseEstimator] = {},
        target_normalizer: Union[
            NORMALIZER, str, List[NORMALIZER], Tuple[NORMALIZER]
        ] = "auto",
        randomize_length: Union[None, Tuple[float, float], bool] = False,
        predict_mode: bool = False,
    ) -> None:
        """

        Args:
            data (pd.DataFrame): dataframe with sequence data - each row can be identified with
                ``time_idx`` and the ``group_ids``
            time_idx (str): integer column denoting the time index. This columns is used to determine
                the sequence of samples.
            target (Union[str, List[str]]): column denoting the target.
            group_ids (Union[str, List[str]]): list of column names identifying a time series. This means that the ``group_ids``
                identify a sample together with the ``time_idx``. If you have only one timeseries, set this to the
                name of column that is constant.
            max_encoder_length (int): maximum length to encode.
                This is the maximum history length used by the time series dataset.
            min_encoder_length (int): minimum allowed length to encode. Defaults to max_encoder_length.
            min_prediction_idx (int): minimum ``time_idx`` from where to start predictions. This parameter
                can be useful to create a validation or test set.
            max_prediction_length (int): maximum prediction/decoder length.
            min_prediction_length (int): minimum prediction/decoder length. Defaults to max_prediction_length.
            static_categoricals (Union[str, List[str]]): list of categorical variables that do not change over time.
            static_reals (Union[str, List[str]]): list of continuous variables that do not change over time.
            time_varying_known_categoricals (Union[str, List[str]]): list of categorical variables that change over
                time and are known in the future, entries can be also lists which are then encoded together
                (e.g. useful for special days or promotion categories).
            time_varying_known_reals (Union[str, List[str]]): list of continuous variables that change over
                time and are known in the future (e.g. price of a product, but not demand of a product).
            time_varying_unknown_categoricals (Union[str, List[str]]): list of categorical variables that change over
                time and are not known in the future, entries can be also lists which are then encoded together
                (e.g. useful for weather categories). You might want to include your target here.
            time_varying_unknown_reals (Union[str, List[str]]): list of continuous variables that change over
                time and are not known in the future.  You might want to include your target here.
            categorical_encoders (Dict[str, NaNLabelEncoder]): dictionary of scikit learn label transformers.
            scalers (Dict[str, Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer]]): dictionary of
                scikit-learn scalers.
        """
        super().__init__()
        self.time_idx = time_idx
        self.target = to_list(target)
        self.group_ids = to_list(group_ids)
        self.max_encoder_length = max_encoder_length
        if min_encoder_length is None:
            min_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        if min_prediction_idx is None:
            min_prediction_idx = data[self.time_idx].min()
        self.min_prediction_idx = min_prediction_idx
        self.max_prediction_length = max_prediction_length
        if min_prediction_length is None:
            min_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        assert (
            self.min_prediction_length <= self.max_prediction_length
        ), "max prediction length has to be larger equals min prediction length"
        self.static_categoricals = to_list(static_categoricals)
        self.static_reals = to_list(static_reals)
        self.time_varying_known_categoricals = to_list(time_varying_known_categoricals)
        for tar in self.target:
            assert (
                tar not in self.time_varying_known_categoricals
            ), f"target {target} should be an unknown categorical variable in the future"
        self.time_varying_known_reals = to_list(time_varying_known_reals)
        for tar in self.target:
            assert (
                tar not in self.time_varying_known_reals
            ), f"target {target} should be an unknown continuous variable in the future"
        self.time_varying_unknown_categoricals = to_list(
            time_varying_unknown_categoricals
        )
        self.time_varying_unknown_reals = to_list(time_varying_unknown_reals)
        self.variable_groups = {} if len(variable_groups) == 0 else variable_groups
        self.lags = {} if len(lags) == 0 else lags

        # set automatic defaults
        # if isinstance(randomize_length, bool):
        #     if not randomize_length:
        #         randomize_length = None
        #     else:
        #         randomize_length = (0.2, 0.05)
        self.randomize_length = randomize_length
        self.predict_mode = predict_mode
        # initalize encoders for categoricals and scalers for reals
        self.categorical_encoders = (
            {} if len(categorical_encoders) == 0 else categorical_encoders
        )
        self.scalers = {} if len(scalers) == 0 else scalers
        self.target_normalizer = target_normalizer
        # throw out warnings if data does not meet requirements
        self._warning(data)

        # filter data
        if min_prediction_idx is not None:
            data = data[
                lambda x: x[self.time_idx]
                >= self.min_prediction_idx - self.max_encoder_length
            ]
        data = self._sort_data(data, self.group_ids + [self.time_idx])

        # target normalizer
        self._set_target_normalizer(data)
        # preprocessing
        data, scales = self._preprocess_data(data)

        # create index
        self.index = self._construct_index(data, predict_mode=predict_mode)
        #
        # # convert to torch tensor for high performance data loading later
        self.data = self._data_to_tensors(data)

    @property
    def _group_ids_mapping(self):
        """
        Mapping of group id names to group ids used to identify series in dataset -
        group ids can also be used for target normalizer.
        The former can change from training to validation and test dataset while the later must not.
        """
        return {name: f"__group_id__{name}" for name in self.group_ids}

    @property
    def _group_ids(self):
        """
        Group ids used to identify series in dataset.
        """
        return list(self._group_ids_mapping.values())

    def save(self, fname: str) -> None:
        """
        Save dataset to disk

        Args:
            fname (str): filename to save to
        """
        torch.save(self, fname)

    @classmethod
    def load(cls, fname: str):
        """
        Load dataset from disk

        Args:
            fname (str): filename to load from

        Returns:
            TimeSeriesDataSet
        """
        obj = torch.load(fname)
        assert isinstance(obj, cls), f"Loaded file is not of class {cls}"
        return obj

    @property
    def categoricals(self) -> List[str]:
        """
        Categorical variables as used for modelling. Excluding categorical target if classification dataset.

        Returns:
            List[str]: list of variables
        """
        return (
            self.static_categoricals
            + self.time_varying_known_categoricals
            + self.time_varying_unknown_categoricals
        )

    @property
    def flat_categoricals(self) -> List[str]:
        """
        Categorical variables as defined in input data.

        Returns:
            List[str]: list of variables
        """
        categories = []
        for name in self.categoricals:
            if name in self.variable_groups:
                categories.extend(self.variable_groups[name])
            else:
                categories.append(name)
        return categories

    @property
    def variable_to_group_mapping(self) -> Dict[str, str]:
        """
        Mapping from categorical variables to variables in input data.

        Returns:
            Dict[str, str]: dictionary mapping from :py:meth:`~categorical` to :py:meth:`~flat_categoricals`.
        """
        groups = {}
        for group_name, sublist in self.variable_groups.items():
            groups.update({name: group_name for name in sublist})
        return groups

    @property
    def reals(self) -> List[str]:
        """
        Continous variables as used for modelling. Excluding continuous target if regression dataset.

        Returns:
            List[str]: list of variables
        """
        return (
            self.static_reals
            + self.time_varying_known_reals
            + self.time_varying_unknown_reals
        )

    @property
    def time_index(self) -> List[str]:
        return self.time_idx

    @property
    def target_name(self) -> List[str]:
        return self.target

    def _set_target_normalizer(self, data: pd.DataFrame):
        """
        Determine target normalizer.

        Args:
            data (pd.DataFrame): input data
        """
        if isinstance(self.target_normalizer, str) and self.target_normalizer == "auto":
            normalizers = []
            for target in self.target:
                if data[target].dtype.kind != "f":  # category
                    normalizers.append(NaNLabelEncoder())
                    # if self.add_target_scales:
                    #     warnings.warn("Target scales will be only added for continous targets", UserWarning)
                else:
                    data_positive = (data[target] > 0).all()
                    if data_positive:
                        if data[target].skew() > 2.5:
                            transformer = "log"
                        else:
                            transformer = "relu"
                    else:
                        transformer = None
                    if self.max_encoder_length > 20 and self.min_encoder_length > 1:
                        normalizers.append(
                            EncoderNormalizer(transformation=transformer)
                        )
                    else:
                        normalizers.append(GroupNormalizer(transformation=transformer))
            if self.multi_target:
                self.target_normalizer = MultiNormalizer(normalizers)
            else:
                self.target_normalizer = normalizers[0]
        elif isinstance(self.target_normalizer, (tuple, list)):
            self.target_normalizer = MultiNormalizer(self.target_normalizer)
        elif self.target_normalizer is None:
            self.target_normalizer = TorchNormalizer(method="identity")
        assert self.min_encoder_length > 1 or not isinstance(
            self.target_normalizer, EncoderNormalizer
        ), "EncoderNormalizer is only allowed if min_encoder_length > 1"
        assert isinstance(
            self.target_normalizer, (TorchNormalizer, NaNLabelEncoder)
        ), f"target_normalizer has to be either None or of class TorchNormalizer but found {self.target_normalizer}"
        assert not self.multi_target or isinstance(
            self.target_normalizer, MultiNormalizer
        ), (
            "multiple targets / list of targets requires MultiNormalizer as target_normalizer "
            f"but found {self.target_normalizer}"
        )

    @property
    def multi_target(self) -> bool:
        """
        If dataset encodes one or multiple targets.

        Returns:
            bool: true if multiple targets
        """
        return len(self.target) > 1

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.

        Returns:
            Dict[str, Any]: dictionary of parameters
        """
        kwargs = {
            name: getattr(self, name)
            for name in inspect.signature(self.__class__.__init__).parameters.keys()
            if name not in ["data", "self"]
        }
        kwargs["categorical_encoders"] = self.categorical_encoders
        kwargs["scalers"] = self.scalers
        return kwargs

    @classmethod
    def from_dataset(
        cls,
        dataset,
        data: pd.DataFrame,
        stop_randomization: bool = False,
        predict: bool = False,
        **update_kwargs,
    ):
        """
        Generate dataset with different underlying data but same variable encoders and scalers, etc.

        Calls :py:meth:`~from_parameters` under the hood.

        Args:
            dataset (TimeSeriesDataSet): dataset from which to copy parameters
            data (pd.DataFrame): data from which new dataset will be generated
            stop_randomization (bool, optional): If to stop randomizing encoder and decoder lengths,
                e.g. useful for validation set. Defaults to False.
            predict (bool, optional): If to predict the decoder length on the last entries in the
                time index (i.e. one prediction per group only). Defaults to False.
            **kwargs: keyword arguments overriding parameters in the original dataset

        Returns:
            TimeSeriesDataSet: new dataset
        """
        return cls.from_parameters(
            dataset.get_parameters(),
            data,
            stop_randomization=stop_randomization,
            predict=predict,
            **update_kwargs,
        )

    @classmethod
    def from_parameters(
        cls,
        parameters: Dict[str, Any],
        data: pd.DataFrame,
        stop_randomization: bool = None,
        predict: bool = False,
        **update_kwargs,
    ):
        """
        Generate dataset with different underlying data but same variable encoders and scalers, etc.

        Args:
            parameters (Dict[str, Any]): dataset parameters which to use for the new dataset
            data (pd.DataFrame): data from which new dataset will be generated
            stop_randomization (bool, optional): If to stop randomizing encoder and decoder lengths,
                e.g. useful for validation set. Defaults to False.
            predict (bool, optional): If to predict the decoder length on the last entries in the
                time index (i.e. one prediction per group only). Defaults to False.
            **kwargs: keyword arguments overriding parameters

        Returns:
            TimeSeriesDataSet: new dataset
        """
        parameters = deepcopy(parameters)
        if predict:
            if stop_randomization is None:
                stop_randomization = True
            elif not stop_randomization:
                warnings.warn(
                    "If predicting, no randomization should be possible - setting stop_randomization=True",
                    UserWarning,
                )
                stop_randomization = True
            parameters["min_prediction_length"] = parameters["max_prediction_length"]
            parameters["predict_mode"] = True
        elif stop_randomization is None:
            stop_randomization = False

        if stop_randomization:
            parameters["randomize_length"] = None
        parameters.update(update_kwargs)

        new = cls(data, **parameters)
        return new

    def _warning(self, data: pd.DataFrame) -> None:
        """
        Check if dataset meets the requirements of TimeSeriesDataSet class. If not, throw out
        warnings.

        e.g. Multiple rows with the same group id and time index exist.

        """
        # check data type

        assert (
            data[self.time_idx].dtypes.kind == "i"
        ), "Timeseries index should be of type integer"

        # if multiple rows with the same group ids and time index exist
        if sum(data.groupby(self.group_ids + [self.time_idx]).size() > 1) > 0:
            raise ValueError(
                "Data Error: Multiple rows with the same group id and time index exist"
            )

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale continuous variables, encode categories and set aside target and weight.

        Args:
            data (pd.DataFrame): original data

        Returns:
            pd.DataFrame: pre-processed dataframe

        """
        # filter groups whose size is smaller than requirement
        data = self._filter(data)

        # ensure data order

        data = self._sort_data(data=data, columns=self.group_ids + [self.time_idx])

        # fill missing values with zero. TODO：more methods to fill missing values
        data = fill_missing_values(
            data=data, reals=self.reals, categoricals=self.categoricals, method="zero"
        )

        # step 1
        # encode group ids - this encoding
        for name, group_name in self._group_ids_mapping.items():
            # use existing encoder - but a copy of it not too loose current encodings
            encoder = deepcopy(
                self.categorical_encoders.get(group_name, NaNLabelEncoder())
            )
            self.categorical_encoders[group_name] = encoder.fit(
                data[name].to_numpy().reshape(-1), overwrite=False
            )
            data[group_name] = self.transform_values(
                name, data[name], inverse=False, group_id=True
            )

        # step 2
        # encode categoricals first to ensure that group normalizer for relies on encoded categories
        if isinstance(
            self.target_normalizer, (GroupNormalizer, MultiNormalizer)
        ):  # if we use a group normalizer, group_ids must be encoded as well
            group_ids_to_encode = self.group_ids
        else:
            group_ids_to_encode = []

        for name in dict.fromkeys(group_ids_to_encode + self.categoricals):
            if name in self.variable_groups:  # fit groups
                columns = self.variable_groups[name]
                if name not in self.categorical_encoders:
                    self.categorical_encoders[name] = NaNLabelEncoder().fit(
                        data[columns].to_numpy().reshape(-1)
                    )
                elif self.categorical_encoders[name] is not None:
                    try:
                        check_is_fitted(self.categorical_encoders[name])
                    except NotFittedError:
                        self.categorical_encoders[name] = self.categorical_encoders[
                            name
                        ].fit(data[columns].to_numpy().reshape(-1))
            else:
                if name not in self.categorical_encoders:
                    self.categorical_encoders[name] = LabelEncoder().fit(data[name])
                elif (
                    self.categorical_encoders[name] is not None
                    and name not in self.target
                ):
                    try:
                        check_is_fitted(self.categorical_encoders[name])
                    except NotFittedError:
                        self.categorical_encoders[name] = self.categorical_encoders[
                            name
                        ].fit(data[name])

        # transform them
        for name in dict.fromkeys(group_ids_to_encode + self.flat_categoricals):
            # targets and its lagged versions are handled separetely
            if name not in self.target:
                data[name] = self.transform_values(
                    name,
                    data[name],
                    inverse=False,
                )

        # save special variables
        assert (
            "__time_idx__" not in data.columns
        ), "__time_idx__ is a protected column and must not be present in data"
        data["__time_idx__"] = data[self.time_idx]  # save unscaled
        for target in self.target:
            assert (
                f"__target__{target}" not in data.columns
            ), f"__target__{target} is a protected column and must not be present in data"
            data[f"__target__{target}"] = data[target]

        # step 3
        # encode and  normalizer target
        if self.target_normalizer is not None:

            # fit target normalizer
            try:
                check_is_fitted(self.target_normalizer)
            except NotFittedError:
                if isinstance(self.target_normalizer, EncoderNormalizer):
                    self.target_normalizer.fit(data[self.target])
                elif isinstance(
                    self.target_normalizer, (GroupNormalizer, MultiNormalizer)
                ):
                    self.target_normalizer.fit(data[self.target], data)
                else:
                    self.target_normalizer.fit(data[self.target])

            # transform target
            if isinstance(self.target_normalizer, EncoderNormalizer):
                # we approximate the scales and target transformation by assuming one
                # transformation over the entire time range but by each group
                common_init_args = [
                    name
                    for name in inspect.signature(
                        GroupNormalizer.__init__
                    ).parameters.keys()
                    if name
                    in inspect.signature(EncoderNormalizer.__init__).parameters.keys()
                    and name not in ["data", "self"]
                ]
                copy_kwargs = {
                    name: getattr(self.target_normalizer, name)
                    for name in common_init_args
                }
                normalizer = GroupNormalizer(groups=self.group_ids, **copy_kwargs)
                data[self.target], scales = normalizer.fit_transform(
                    data[self.target], data, return_norm=True
                )

            elif isinstance(self.target_normalizer, GroupNormalizer):
                data[self.target], scales = self.target_normalizer.transform(
                    data[self.target], data, return_norm=True
                )

            elif isinstance(self.target_normalizer, MultiNormalizer):
                transformed, scales = self.target_normalizer.transform(
                    data[self.target], data, return_norm=True
                )

                for idx, target in enumerate(self.target):
                    data[target] = transformed[idx]

                    if isinstance(self.target_normalizer[idx], NaNLabelEncoder):
                        # overwrite target because it requires encoding (continuous targets should not be normalized)
                        data[f"__target__{target}"] = data[target]

            elif isinstance(self.target_normalizer, NaNLabelEncoder):
                data[self.target] = self.target_normalizer.transform(data[self.target])
                # overwrite target because it requires encoding (continuous targets should not be normalized)
                data[f"__target__{self.target}"] = data[self.target]
                scales = None

            else:
                data[self.target], scales = self.target_normalizer.transform(
                    data[self.target], return_norm=True
                )

        # step 4
        # rescale continuous variables apart from target
        for name in self.reals:
            if name in self.target:
                # lagged variables are only transformed - not fitted
                continue
            elif name not in self.scalers:
                self.scalers[name] = StandardScaler().fit(data[[name]])
            elif self.scalers[name] is not None:
                try:
                    check_is_fitted(self.scalers[name])
                except NotFittedError:
                    if isinstance(self.scalers[name], GroupNormalizer):
                        self.scalers[name] = self.scalers[name].fit(data[[name]], data)
                    else:
                        self.scalers[name] = self.scalers[name].fit(data[[name]])

        # transformer them after fitting
        for name in self.reals:
            # targets are handled separately
            transformer = self.get_transformer(name)
            if (
                name not in self.target
                and transformer is not None
                and not isinstance(transformer, EncoderNormalizer)
            ):
                data[name] = self.transform_values(
                    name, data[name], data=data, inverse=False
                )

        return data, scales

    def get_transformer(self, name: str, group_id: bool = False):
        """
        Get transformer for variable.

        Args:
            name (str): variable name
            group_id (bool, optional): If the passed name refers to a group id (different encoders are used for these).
                Defaults to False.

        Returns:
            transformer
        """
        if group_id:
            name = self._group_ids_mapping[name]

        if name in set(self.flat_categoricals + self.group_ids + self._group_ids):
            name = self.variable_to_group_mapping.get(name, name)  # map name to encoder
            transformer = self.categorical_encoders.get(name, None)
            return transformer

        elif name in self.reals:
            # take target normalizer if required
            transformer = self.scalers.get(name, None)
            return transformer
        else:
            return None

    def transform_values(
        self,
        name: str,
        values: Union[pd.Series, torch.Tensor, np.ndarray],
        data: pd.DataFrame = None,
        inverse=False,
        group_id: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Scale and encode values.

        Args:
            name (str): name of variable
            values (Union[pd.Series, torch.Tensor, np.ndarray]): values to encode/scale
            data (pd.DataFrame, optional): extra data used for scaling (e.g. dataframe with groups columns).
                Defaults to None.
            inverse (bool, optional): if to conduct inverse transformation. Defaults to False.
            group_id (bool, optional): If the passed name refers to a group id (different encoders are used for these).
                Defaults to False.
            **kwargs: additional arguments for transform/inverse_transform method

        Returns:
            np.ndarray: (de/en)coded/(de)scaled values
        """
        transformer = self.get_transformer(name, group_id=group_id)
        if transformer is None:
            return values
        if inverse:
            transform = transformer.inverse_transform
        else:
            transform = transformer.transform

        if group_id:
            name = self._group_ids_mapping[name]
        # remaining categories
        if name in self.flat_categoricals + self.group_ids + self._group_ids:
            return transform(values, **kwargs)

        # reals
        elif name in self.reals:
            if isinstance(transformer, GroupNormalizer):
                return transform(values, data, **kwargs)
            elif isinstance(transformer, EncoderNormalizer):
                return transform(values, **kwargs)
            else:
                if isinstance(values, pd.Series):
                    values = values.to_frame()
                    return np.asarray(transform(values, **kwargs)).reshape(-1)
                else:
                    values = values.reshape(-1, 1)
                    return transform(values, **kwargs).reshape(-1)
        else:
            return values

    def __len__(self) -> int:
        """
        Length of dataset.

        Returns:
            int: length
        """
        return len(self.index)

    def summary(self):
        """
        Summarize basic statistics of given dataset.

        Missing values, number of categorical features, number of numeric features,
        size of dataset, time interval, number of groups, and etc.


        """
        pass

    def extract_features(self):
        """
        Based on given timeseries dataset, generate new features to extract time
        series information.

        """
        pass

    def _sort_data(self, data: pd.DataFrame, columns: Union[str, List[str]]):
        """
        sort data and reset index.
        """
        return data.sort_values(by=columns).reset_index(drop=True)

    def _filter(self, data: pd.DataFrame):
        """
        filter groups whose size does not meet `min_encoder_length`+`min_prediction_length`.
        """
        group_size = data.groupby(self.group_ids).size()
        drop_list = group_size[
            group_size < (self.min_encoder_length + self.min_prediction_length)
        ].index
        data[~data[self.group_ids].isin(drop_list)].reset_index()
        return data

    def _construct_index(self, data: pd.DataFrame, predict_mode: bool) -> pd.DataFrame:
        """
        Create index of samples.

        Args:
            data (pd.DataFrame): preprocessed data
            index_name (str):
        Returns:
            pd.DataFrame: index dataframe
        """
        g = data.groupby(self.group_ids, observed=True)

        df_index_first = g["__time_idx__"].transform("nth", 0).to_frame("time_first")
        df_index_last = g["__time_idx__"].transform("nth", -1).to_frame("time_last")
        df_index_diff_to_next = (
            -g["__time_idx__"]
            .diff(-1)
            .fillna(-1)
            .astype(int)
            .to_frame("time_diff_to_next")
        )
        df_index = pd.concat(
            [df_index_first, df_index_last, df_index_diff_to_next], axis=1
        )
        df_index["index_start"] = np.arange(len(df_index))
        df_index["time"] = data["__time_idx__"]
        df_index["count"] = (df_index["time_last"] - df_index["time_first"]).astype(
            int
        ) + 1
        group_ids = g.ngroup()
        df_index["group_id"] = group_ids

        min_sequence_length = self.min_prediction_length + self.min_encoder_length
        max_sequence_length = self.max_prediction_length + self.max_encoder_length

        # calculate maximum index to include from current index_start
        max_time = (df_index["time"] + max_sequence_length - 1).clip(
            upper=df_index["count"] + df_index.time_first - 1
        )

        df_index["index_end"], missing_sequences = _find_end_indices(
            diffs=df_index.time_diff_to_next.to_numpy(),
            max_lengths=(max_time - df_index.time).to_numpy() + 1,
            min_length=min_sequence_length,
        )
        # filter out where encode and decode length are not satisfied
        df_index["sequence_length"] = (
            df_index["index_start"].iloc[df_index["index_end"]].to_numpy()
            - df_index["index_start"]
            + 1
        )

        # filter too short sequences
        df_index = df_index[
            # sequence must be at least of minimal prediction length
            lambda x: (x.sequence_length >= min_sequence_length)
        ]

        if (
            predict_mode
        ):  # keep longest element per series (i.e. the first element that spans to the end of the series)
            # filter all elements that are longer than the allowed maximum sequence length
            df_index = df_index[
                lambda x: (x["time_last"] - x["time"] + 1 <= max_sequence_length)
                & (x["sequence_length"] >= min_sequence_length)
            ]
            # choose longest sequence
            df_index = df_index.loc[
                df_index.groupby("group_id").sequence_length.idxmax()
            ]

        assert (
            len(df_index) > 0
        ), "filters should not remove entries all entries - check encoder/decoder lengths "

        df_index.reset_index(inplace=True)
        return df_index

    def _data_to_tensors(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Convert data to tensors for faster access with :py:meth:`~__getitem__`.

        Args:
            data (pd.DataFrame): preprocessed data

        Returns:
            Dict[str, torch.Tensor]: dictionary of tensors for continous, categorical data, groups, target and
                time index
        """
        index = check_for_nonfinite(
            torch.tensor(data[self.group_ids].to_numpy(np.int64), dtype=torch.int64),
            self.group_ids,
        )
        time = check_for_nonfinite(
            torch.tensor(data[self.time_idx].to_numpy(np.int64), dtype=torch.int64),
            self.time_idx,
        )
        categorical = check_for_nonfinite(
            torch.tensor(data[self.categoricals].to_numpy(np.int64), dtype=torch.int64),
            self.categoricals,
        )
        target = check_for_nonfinite(
            torch.tensor(
                data[self.target].to_numpy(dtype=np.float64), dtype=torch.float
            ),
            self.target,
        )
        continuous = check_for_nonfinite(
            torch.tensor(
                data[self.reals].to_numpy(dtype=np.float64), dtype=torch.float
            ),
            self.reals,
        )
        # target_scale = torch.tensor(target_scale, dtype=torch.float)
        tensors = dict(
            reals=continuous,
            categoricals=categorical,
            groups=index,
            target=target,
            time=time,
            # target_scale=target_scale
        )

        return tensors

    def _collate_fn(
        self, batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Collate function to combine items into mini-batch for dataloader.

        Args:
            batches (List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]): List of samples generated with
                :py:meth:`~__getitem__`.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]: minibatch
        """
        # collate function for dataloader
        # lengths
        encoder_lengths = torch.tensor(
            [batch[0]["encoder_length"] for batch in batches], dtype=torch.long
        )
        decoder_lengths = torch.tensor(
            [batch[0]["decoder_length"] for batch in batches], dtype=torch.long
        )

        # ids
        decoder_time_idx_start = (
            torch.tensor(
                [batch[0]["encoder_time_idx_start"] for batch in batches],
                dtype=torch.long,
            )
            + encoder_lengths
        )
        decoder_time_idx = decoder_time_idx_start.unsqueeze(1) + torch.arange(
            decoder_lengths.max()
        ).unsqueeze(0)
        groups = torch.stack([batch[0]["groups"] for batch in batches])

        # features
        encoder_cont = rnn.pad_sequence(
            [
                batch[0]["x_cont"][:length]
                for length, batch in zip(encoder_lengths, batches)
            ],
            batch_first=True,
        )
        encoder_cat = rnn.pad_sequence(
            [
                batch[0]["x_cat"][:length]
                for length, batch in zip(encoder_lengths, batches)
            ],
            batch_first=True,
        )

        decoder_cont = rnn.pad_sequence(
            [
                batch[0]["x_cont"][length:]
                for length, batch in zip(encoder_lengths, batches)
            ],
            batch_first=True,
        )
        decoder_cat = rnn.pad_sequence(
            [
                batch[0]["x_cat"][length:]
                for length, batch in zip(encoder_lengths, batches)
            ],
            batch_first=True,
        )

        # target
        target = rnn.pad_sequence([batch[1][0] for batch in batches], batch_first=True)
        encoder_target = rnn.pad_sequence(
            [batch[0]["encoder_target"] for batch in batches], batch_first=True
        )

        return (
            dict(
                encoder_cat=encoder_cat,
                encoder_cont=encoder_cont,
                encoder_target=encoder_target,
                encoder_lengths=encoder_lengths,
                decoder_cat=decoder_cat,
                decoder_cont=decoder_cont,
                decoder_target=target,
                decoder_lengths=decoder_lengths,
                decoder_time_idx=decoder_time_idx,
                groups=groups,
            ),
            target,
        )

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get sample for model

        Args:
            idx (int): index of prediction (between ``0`` and ``len(dataset) - 1``)

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: x and y for model
        """
        index = self.index.iloc[idx]
        # get index data
        data_cont = self.data["reals"][index.index_start : index.index_end + 1].clone()
        data_cat = self.data["categoricals"][
            index.index_start : index.index_end + 1
        ].clone()
        time = self.data["time"][index.index_start : index.index_end + 1].clone()
        target = self.data["target"][index.index_start : index.index_end + 1].clone()
        groups = self.data["groups"][index.index_start].clone()
        target_scale = self.target_normalizer.get_parameters(groups, self.group_ids)
        # determine encoder length and decoder length
        # why we should determine these values: if min_encoder_length != max_encoder_length
        # and min_prediction_length != max_prediction_length and sequence length <
        # max_encoder_length + max_prediction_length, we must determine what encoder length is
        # and decoder length is.

        decoder_length = min(
            self.max_prediction_length,
            index.sequence_length - self.min_encoder_length,
        )

        encoder_length = index.sequence_length - decoder_length

        assert (
            decoder_length >= self.min_prediction_length
        ), "Decoder length should be at least minimum prediction length"

        assert (
            encoder_length >= self.min_encoder_length
        ), "Encoder length should be at least minimum encoder length"

        assert decoder_length > 0, "Decoder length should be greater than 0"

        assert encoder_length > 0, "Encoder length should be greater than 0"

        encoder_cat = data_cat[:encoder_length]
        encoder_cont = data_cont[:encoder_length]
        encoder_target = target[:encoder_length]
        decoder_cat = data_cat[encoder_length:]
        decoder_cont = data_cont[encoder_length:]
        decoder_time_idx = time[encoder_length:]
        target = target[encoder_length:]

        return (
            dict(
                encoder_cat=encoder_cat,
                encoder_cont=encoder_cont,
                encoder_target=encoder_target,
                encoder_length=encoder_length,
                decoder_cat=decoder_cat,
                decoder_cont=decoder_cont,
                decoder_length=decoder_length,
                encoder_time_idx_start=time[0],
                decoder_time_idx=decoder_time_idx,
                groups=groups,
                target_scale=target_scale,
            ),
            target,
        )

    def x_to_index(self, x: Dict[str, torch.Tensor]) -> pd.DataFrame:
        """
        Decode dataframe index from x.

        Returns:
            dataframe with time index column for first prediction and group ids
        """
        index_data = {self.time_idx: x["decoder_time_idx"][:, 0].cpu()}
        for id in self.group_ids:
            index_data[id] = x["groups"][:, self.group_ids.index(id)].cpu()
            # decode if possible
            index_data[id] = self.transform_values(
                id, index_data[id], inverse=True, group_id=True
            )
        index = pd.DataFrame(index_data)
        return index


if __name__ == "__main__":

    def main():
        data_with_covariates = pd.DataFrame(
            dict(
                # as before
                value=np.random.rand(60),
                group=np.repeat(np.arange(3), 20),
                time_idx=np.tile(np.arange(20), 3),
                # now adding covariates
                categorical_covariate=np.random.choice(["a", "b"], size=60),
                real_covariate=np.random.rand(60),
            )
        ).astype(
            dict(group=str)
        )  # categorical covariates have to be of string type

        # print(test_data_with_covariates.head())

        training_cutoff = data_with_covariates["time_idx"].max() - 6
        # create the dataset from the pandas dataframe
        rawdata = data_with_covariates[lambda x: x.time_idx <= training_cutoff]

        dataset = TimeSeriesDataSet(
            data=rawdata,
            group_ids=["group"],
            target="value",
            time_idx="time_idx",
            max_encoder_length=4,
            min_encoder_length=4,
            max_prediction_length=2,
            min_prediction_length=2,
            static_reals=[],
            static_categoricals=["group"],
            time_varying_known_reals=["real_covariate"],
            time_varying_unknown_reals=["value"],
            time_varying_known_categoricals=["categorical_covariate"],
            time_varying_unknown_categoricals=[],
            target_normalizer=TorchNormalizer(method="standard")
            # target_normalizer=GroupNormalizer(groups=["group"], method="standard",
            #                                   transformation=None)
        )

        # print(dataset.data)
        # print(dataset.scalers)
        print(dataset.get_parameters())
        train_dataloader = DataLoader(
            dataset, batch_size=16, shuffle=False, drop_last=True
        )
        # validation = TimeSeriesDataSet.from_dataset(
        #     dataset,
        #     data_with_covariates[lambda x: x.time_idx > training_cutoff],
        #     predict=True,
        #     stop_randomization=False,
        # )
        # val_dataloader = DataLoader(validation, batch_size=2, shuffle=False, drop_last=False)
        x, y = next(iter(train_dataloader))
        print(x)
        y_rescale = dataset.target_normalizer(
            dict(prediction=y, target_scale=x["target_scale"])
        )
        y_index = dataset.x_to_index(x)
        print(y_rescale)
        print(y_index)
