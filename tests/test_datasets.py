from deepts_forecasting.datasets import AirPassengersDataset


def test_air_passenger_dataset():
    data = AirPassengersDataset().load()
    assert len(data) == 144, "The length should be 144."
