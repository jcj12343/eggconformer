import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 正弦位置编码 --------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        T = x.size(1)
        x = x + self.pe[:, :T]
        return self.dropout(x)


# -------------------------- 可学习位置编码 --------------------------
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:, :T]
        return self.dropout(x)


# -------------------------- Pre‑LN Transformer Encoder层 --------------------------
class TransformerEncoderLayerPreLN(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        attn_out, _ = self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src))
        src = src + self.dropout1(attn_out)

        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(ff_out)
        return src


class EEGCNNTransformer(nn.Module):
    def __init__(
        self,
        n_channels,
        n_timepoints,
        n_classes,
        F1=8,
        D=2,
        temporal_kernel=32,
        pool_size=4,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=None,
        ffn_expansion=4,
        conf_kernel=31,
        dropout=0.1,
        use_pre_ln=True,
        pos_enc_type="learnable",
        pe_max_len=1000
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # FFN维度兼容逻辑
        if dim_feedforward is not None:
            self._ffn_dim = dim_feedforward
        else:
            self._ffn_dim = d_model * ffn_expansion

        # -------- CNN Stem --------
        self.temp_conv = nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial_conv = nn.Conv2d(F1, F1*D, kernel_size=(n_channels, 1), groups=F1, bias=False)
        self.bn_cnn = nn.BatchNorm2d(F1*D)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_size))
        self.drop_cnn = nn.Dropout(dropout)

        # 兼容旧train.py占位
        self.channel_att = nn.Identity()

        # -------- Projection --------
        cnn_out_ch = F1 * D
        self.proj = nn.Linear(cnn_out_ch, d_model)

        # -------- 位置编码 --------
        if pos_enc_type == "sin":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=pe_max_len, dropout=dropout)
        elif pos_enc_type == "learnable":
            self.pos_enc = LearnablePositionalEncoding(d_model, max_len=pe_max_len, dropout=dropout)
        else:
            raise ValueError("pos_enc_type must be 'sin' or 'learnable'")

        # -------- Transformer Encoder Stack --------
        if use_pre_ln:
            encoder_list = [TransformerEncoderLayerPreLN(d_model, nhead, self._ffn_dim, dropout) for _ in range(num_layers)]
            self.conformer_encoder = nn.Sequential(*encoder_list)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=self._ffn_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu"
            )
            self.conformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # -------- 分类头（GAP全局平均池化，不再使用cls_token） --------
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        B = x.shape[0]
        # CNN主干
        x = self.temp_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn_cnn(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.drop_cnn(x)

        # 维度变换 [B, C, 1, L] -> [B, L, C]
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.proj(x)

        # 不再拼接cls_token！直接加位置编码
        x = self.pos_enc(x)
        x = self.conformer_encoder(x)

        # ====== 全局平均池化 Global Avg Pooling (seq_len维度dim=1求平均) ======
        x = x.mean(dim=1)

        out = self.head_norm(x)
        logits = self.head(out)
        return logits