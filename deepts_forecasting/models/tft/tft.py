"""
The temporal fusion transformer is a powerful predictive model for forecasting timeseries
"""
from copy import copy
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import LSTM
from deepts_forecasting.models.modules import MultiEmbedding
from deepts_forecasting.models.base_model import BaseModelWithCovariates
from deepts_forecasting.models.tft.sub_module import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
)


class TemporalFusionTransformer(BaseModelWithCovariates):
    def __init__(
        self,
        hidden_size,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        max_prediction_length: int = 7,
        max_encoder_length: int = 14,
        attention_head_size: int = 4,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_categoricals: List[str] = [],
        x_reals: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Dict[str, int] = {},
        share_single_variable_networks: bool = False,
        output_size: Union[int, List[int]] = 1,
        loss=nn.L1Loss(),
        **kwargs,
    ):
        self.save_hyperparameters()
        # store loss function separately as it is a module
        super().__init__(loss=loss, **kwargs)

        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(
                    1,
                    self.hparams.hidden_continuous_sizes.get(
                        name, self.hparams.hidden_continuous_size
                    ),
                )
                for name in self.reals
            }
        )

        # variable selection
        # variable selection for static variables
        static_input_sizes = {
            name: self.input_embeddings[name].embedding_dim
            for name in self.hparams.static_categoricals
        }
        static_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.static_reals
            }
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={
                name: True for name in self.hparams.static_categoricals
            },
            dropout=self.hparams.dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.input_embeddings[name].embedding_dim
            for name in self.hparams.time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {
            name: self.input_embeddings[name].embedding_dim
            for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.time_varying_reals_decoder
            }
        )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={
                name: True for name in self.hparams.time_varying_categoricals_encoder
            },
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={
                name: True for name in self.hparams.time_varying_categoricals_decoder
            },
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.hparams.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            self.hparams.dropout,
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(
            self.hparams.hidden_size, dropout=self.hparams.dropout
        )
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = GatedLinearUnit(self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_add_norm_encoder = AddNorm(
            self.hparams.hidden_size, trainable_add=False
        )
        # self.post_lstm_add_norm_decoder = AddNorm(self.hparams.hidden_size, trainable_add=True)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size,
            n_head=self.hparams.attention_head_size,
            dropout=self.hparams.dropout,
        )
        self.post_attn_gate_norm = GateAddNorm(
            self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(
            self.hparams.hidden_size, dropout=None, trainable_add=False
        )

        self.output_layer = nn.Linear(
            self.hparams.hidden_size, self.hparams.output_size
        )

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def forward(self, x: Dict[str, torch.Tensor]):
        x_cat = torch.cat(
            [x["encoder_cat"], x["decoder_cat"]], dim=1
        )  # concatenate in time dimension
        x_cont = torch.cat(
            [x["encoder_cont"], x["decoder_cont"]], dim=1
        )  # concatenate in time dimension
        timesteps = x_cont.size(1)  # encode + decode length
        max_encoder_length = self.hparams.max_encoder_length
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {
                name: input_vectors[name][:, 0] for name in self.static_variables
            }
            (
                static_embedding,
                static_variable_selection,
            ) = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            # static_variable_selection = torch.zeros(
            #     (x_cont.size(0), 0), dtype=self.dtype, device=self.device
            # )

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length]
            for name in self.encoder_variables
        }
        (
            embeddings_varying_encoder,
            encoder_sparse_weights,
        ) = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:]
            for name in self.decoder_variables  # select decoder
        }
        (
            embeddings_varying_decoder,
            decoder_sparse_weights,
        ) = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell)
        )
        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder, (hidden, cell)
        )
        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(
            lstm_output_encoder, embeddings_varying_encoder
        )

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(
            lstm_output_decoder, embeddings_varying_decoder
        )

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output,
            self.expand_static_context(static_context_enrichment, timesteps),
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=None,
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(
            attn_output, attn_input[:, max_encoder_length:]
        )

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])

        output = self.output_layer(output)

        return output


if __name__ == "__main__":
    import pandas as pd
    import torch
    from deepts_forecasting.utils.data import TimeSeriesDataSet
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    from deepts_forecasting.utils.data.encoders import GroupNormalizer, TorchNormalizer

    data_with_covariates = pd.DataFrame(
        dict(
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
    max_encoder_length = 4
    max_prediction_length = 3
    training_cutoff = (
        data_with_covariates["time_idx"].max()
        - max_encoder_length
        - max_prediction_length
    )

    # create the dataset from the pandas dataframe
    rawdata = data_with_covariates[lambda x: x.time_idx <= training_cutoff]

    training = TimeSeriesDataSet(
        data=rawdata,
        group_ids=["group"],
        target="value",
        time_idx="time_idx",
        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_prediction_length=max_prediction_length,
        static_reals=[],
        static_categoricals=["group"],
        time_varying_known_reals=["real_covariate"],
        time_varying_unknown_reals=["value"],
        time_varying_known_categoricals=["categorical_covariate"],
        time_varying_unknown_categoricals=[],
        target_normalizer=TorchNormalizer(method="standard"),
    )

    # print(dataset.data)
    # print(dataset.scalers)
    print(training.get_parameters())
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data_with_covariates[lambda x: x.time_idx > training_cutoff],
        predict=True,
        stop_randomization=False,
    )

    batch_size = 2
    train_dataloader = DataLoader(
        training, batch_size=batch_size, shuffle=False, drop_last=False
    )
    val_dataloader = DataLoader(
        validation, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # create PyTorch Lighning Trainer with early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=50, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=30,
        gpus=0,  # run on CPU, if on multiple GPUs, use accelerator="ddp"
        gradient_clip_val=0.1,
        limit_train_batches=30,  # 30 batches per epoch
        callbacks=[lr_logger, early_stop_callback],
        logger=TensorBoardLogger("lightning_logs"),
    )

    model = TemporalFusionTransformer.from_dataset(
        training, hidden_size=16, lstm_layers=2, hidden_continuous_size=8
    )
    trainer.fit(
        model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # calcualte mean absolute error on validation set
    actuals = torch.cat(
        [
            model.transform_output(prediction=y, target_scale=x["target_scale"])
            for x, y in iter(val_dataloader)
        ]
    )
    predictions, x_index = best_model.predict(val_dataloader)
    mae = (actuals - predictions).abs().mean()
    # print('predictions shape is:', predictions.shape)
    # print('actuals shape is:', actuals.shape)
    print(torch.cat([actuals, predictions]))
    print("MAE is:", mae)
