"""
Datasets
--------

A few popular time series datasets
"""
from abc import ABC

from deepts_forecasting.datasets.get_data import DatasetLoader, DatasetLoaderMetadata

"""
    Overall usage of this package:
    from darts.datasets import AirPassengersDataset
    ts: TimeSeries = AirPassengersDataset.load()
"""

# _DEFAULT_PATH = "https://raw.githubusercontent.com/unit8co/darts/master/datasets"

_DEFAULT_PATH = "https://raw.github.com/yunxileo/deepts_forecasting/main/deepts_forecasting/datasets/data"


class AirPassengersDataset(DatasetLoader, ABC):
    """
    Monthly Air Passengers Dataset, from 1949 to 1960.
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "air_passengers.csv",
                uri=_DEFAULT_PATH + "/AirPassengers.csv",
                hash="925157518f55c59964c33ab46e38f6a5",
                header_time="Month",
            )
        )


class TemperatureDataset(DatasetLoader, ABC):
    """
    Daily temperature in Melbourne between 1981 and 1990
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "temperatures.csv",
                uri=_DEFAULT_PATH + "/temps.csv",
                hash="ce5b5e4929793ec8b6a54711110acebf",
                header_time="Date",
                format_time="%m/%d/%Y",
                freq="D",
            )
        )


class SunspotsDataset(DatasetLoader):
    """
    Monthly Sunspot Numbers, 1749 - 1983

    Monthly mean relative sunspot numbers from 1749 to 1983.
    Collected at Swiss Federal Observatory, Zurich until 1960, then Tokyo Astronomical Observatory.

    Source: [1]_

    References
    ----------
    .. [1] https://www.rdocumentation.org/packages/datasets/versions/3.6.1/topics/sunspots
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "sunspots.csv",
                uri=_DEFAULT_PATH + "/monthly-sunspots.csv",
                hash="d7312c8cc0b80ec511e008327270efe1",
                header_time="Month",
                format_time="%Y-%m",
            )
        )


class EnergyDataset(DatasetLoader, ABC):
    """
    Hourly energy dataset coming from [1]_.

    Contains a time series with 28 hourly components between 2014-12-31 23:00:00 and 2018-12-31 22:00:00

    References
    ----------
    .. [1] https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "energy.csv",
                uri=_DEFAULT_PATH + "/energy_dataset.csv",
                hash="f564ef18e01574734a0fa20806d1c7ee",
                header_time="time",
                format_time="%Y-%m-%d %H:%M:%S",
            )
        )


class StoreDataset(DatasetLoader, ABC):
    """
    Hourly energy dataset coming from [1]_.

    Contains a time series with 28 hourly components between 2014-12-31 23:00:00 and 2018-12-31 22:00:00

    References
    ----------
    .. [1] https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather
    """

    def __init__(self):
        super().__init__(
            metadata=DatasetLoaderMetadata(
                "store_demand.csv",
                uri=_DEFAULT_PATH + "/Store_Demand/train.csv",
                hash="9566c3963451137c2d7733f39fedb397",
                header_time="date",
                format_time="%Y-%m-%d",
            )
        )


if __name__ == "__main__":
    data = AirPassengersDataset().load()
    print(len(data))
