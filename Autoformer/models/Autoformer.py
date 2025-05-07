import torch
import torch.nn as nn
from Autoformer.layers.Embed import DataEmbedding_wo_pos
from Autoformer.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from Autoformer.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        if configs.multihead:
            self.head = nn.Sequential(
                nn.Linear(configs.d_model, configs.hidden_size),
                nn.BatchNorm1d(configs.hidden_size),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(configs.hidden_size, configs.output_dim),
                )
        else:
            self.head = None



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_latent=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        start_idx = self.seq_len // 2 - self.pred_len // 2 - self.label_len // 2
        end_idx = self.seq_len // 2 + self.pred_len // 2 + self.label_len // 2 
        trend_init = torch.cat([trend_init[:, start_idx:start_idx + self.label_len // 2, :], mean, 
                                trend_init[:, end_idx - self.label_len // 2:end_idx, :]], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, start_idx:start_idx + self.label_len // 2, :], zeros,
                                   seasonal_init[:, end_idx - self.label_len // 2:end_idx, :]], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        if x_latent is not None:
            x_enc = x_enc + x_latent.unsqueeze(1)
            # enc_out = torch.cat([enc_out, x_latent.unsqueeze(1)], dim=-1)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        head_out = self.head(enc_out.sum(1)) if self.head is not None else torch.zeros(x_enc.shape[0]) 

        if self.output_attention:                
            return dec_out[:, self.label_len//2:-self.label_len//2, :], head_out, attns
        else:
            return dec_out[:, self.label_len//2:-self.label_len//2, :], head_out  # [B, L, D]
