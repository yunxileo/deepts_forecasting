# DeepTS_Forecasting


<p align="center">
<a href="https://pypi.python.org/pypi/deepts_forecasting">
    <img src="https://img.shields.io/pypi/v/deepts_forecasting.svg"
        alt = "Release Status">
</a>

<a href="https://github.com/yunxileo/deepts_forecasting/actions">
    <img src="https://github.com/yunxileo/deepts_forecasting/actions/workflows/main.yml/badge.svg?branch=release" alt="CI Status">
</a>

<a href="https://deepts-forecasting.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/deepts-forecasting/badge/?version=latest" alt="Documentation Status">
</a>

</p>


**Deepts_forecasting** is a Easy-to-use package for time series forecasting with deep Learning models.
It contains a variety of models, from classics such as seq2seq to more complex deep neural networks.
The models can all be used in the same way, using `fit()` and `predict()` functions,


* Free software: MIT

##  Documentation

* <https://yunxileo.github.io/deepts_forecasting/>


## Features

* TODO


## Models list

| Model        |        Paper                            |
|--------------|-----------------------------------------|
| Seq2Seq      |   [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)                                      |
| DeepAR       |[DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)                                         |
| Lstnet       |[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://arxiv.org/pdf/1703.07015.pdf)                                         |
| MQ-RNN       |  [A Multi-Horizon Quantile Recurrent Forecaster](https://arxiv.org/pdf/1711.11053.pdf)                                       |
| N-Beats      | [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)                                |
| TCN          |  [An empirical evaluation of generic convolutional and recurrent networks for sequence modeling](https://arxiv.org.1803.01271)                                     |
| Transformer  |    [Attention Is All You Need](https://arxiv.org/abs/1706.03762)                                     |
| Informer     |[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)                                         |
| Autoformer   | [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)                                        |
| TFT          | [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)                                        |
| MAE          |  [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)                                       |


## LICENSE

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [zillionare/cookiecutter-pypackage](https://github.com/zillionare/cookiecutter-pypackage) project template.
