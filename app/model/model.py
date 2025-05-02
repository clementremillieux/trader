"""DatasetCreator class to create datasets for machine learning."""

import math

import torch

import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        # (T, d)
        position = torch.arange(0, max_len).unsqueeze(1)  # (T,1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1,T,d)

    def forward(self, x):
        """
        x : (batch, seq_len, d_model)
        """
        return x + self.pe[:, : x.size(1)]


class DilatedTCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, ks=3, n_layers=4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2**i
            pad = (ks - 1) // 2 * dilation
            layers += [
                nn.Conv1d(
                    in_ch if i == 0 else hidden_ch,
                    hidden_ch,
                    ks,
                    dilation=dilation,
                    padding=pad,
                ),
                nn.ReLU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x:(B,T,F)
        x = x.transpose(1, 2)  # (B,F,T)
        return self.net(x).transpose(1, 2)  # (B,T,H)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """

    def __init__(self, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for query, key, and value for all heads
        self.query_proj = nn.Linear(d_model, d_model)

        self.key_proj = nn.Linear(d_model, d_model)

        self.value_proj = nn.Linear(d_model, d_model)

        # Final projection to combine attention heads
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim**0.5  # Scale factor for stability

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Project the queries, keys, and values
        Q = self.query_proj(x)  # (batch, seq_len, d_model)
        K = self.key_proj(x)  # (batch, seq_len, d_model)
        V = self.value_proj(x)  # (batch, seq_len, d_model)

        # Split the dimensions for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention weights
        attn_weights = (
            torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        )  # (batch, n_heads, seq_len, seq_len)

        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # Apply attention to the values
        attn_output = torch.matmul(
            attn_weights, V
        )  # (batch, n_heads, seq_len, head_dim)

        # Reshape back to (batch, seq_len, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Final linear projection
        output = self.out_proj(attn_output)  # (batch, seq_len, d_model)

        return output, attn_weights


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block that includes multi-head attention with residual connections,
    layer normalization, and a feed-forward network.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, ff_hidden_multiplier=4):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_hidden_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_hidden_multiplier, d_model),
        )

    def forward(self, x):
        # Multi-head attention sub-layer with residual connection and layer normalization
        attn_output, attn_weights = self.mha(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward sub-layer with residual connection and layer normalization
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoderBlockPreNorm(nn.Module):
    def __init__(self, d_model, n_heads, p_attn=0.1, p_ff=0.3, ff_mult=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout=p_attn)
        self.drop_attn = nn.Dropout(p_attn)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),  # ← ReLU → GELU
            nn.Dropout(p_ff),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.drop_ff = nn.Dropout(p_ff)

    def forward(self, x):
        # ─── Pre-Norm Multi-Head Attention ──────────────────────
        y = self.norm1(x)
        attn_out, _ = self.mha(y)
        x = x + self.drop_attn(attn_out)

        # ─── Pre-Norm Feed-Forward ─────────────────────────────
        y = self.norm2(x)
        ff_out = self.ff(y)
        x = x + self.drop_ff(ff_out)
        return x


class AttentionPooling(nn.Module):
    """
    Learned attention pooling mechanism. Instead of using a simple mean pooling,
    this module learns to weight different time steps.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        # Initialize a learnable attention vector
        self.attn_vector = nn.Parameter(torch.randn(hidden_size))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_size)
        Returns a pooled representation: (batch, hidden_size)
        """
        # Compute attention scores for each time step
        scores = torch.matmul(x, self.attn_vector)  # (batch, seq_len)
        weights = self.softmax(scores).unsqueeze(-1)  # (batch, seq_len, 1)
        pooled = torch.sum(x * weights, dim=1)  # (batch, hidden_size)
        return pooled


class PatchEmbed(nn.Module):
    """
    Découpe la séquence (T, F) en patches contigus de taille `patch_size`
    puis projette chaque patch en un vecteur `d_model`.
    - Si T n’est pas multiple de patch_size, on zero-pad la fin.
    """

    def __init__(self, in_ch: int, patch_size: int = 16, d_model: int = 128):
        super().__init__()
        self.ps = patch_size
        self.proj = nn.Linear(in_ch * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, F)
        B, T, F = x.shape
        pad = (-T) % self.ps  # pour avoir un nombre entier de patches
        if pad:
            x = torch.cat([x, x.new_zeros(B, pad, F)], dim=1)

        n_patches = x.size(1) // self.ps
        x = x.view(B, n_patches, self.ps * F)  # (B, n_p, ps·F)
        return self.proj(x)  # (B, n_p, d_model)


class SimpleModel(nn.Module):
    def __init__(
        self,
        num_historical_features: int,
        encoder_length: int,
        hidden_size: int = 1024,
        dropout: float = 0.1,
        lstm_layers: int = 1,
        n_heads: int = 4,
        num_attention_layers: int = 3,
        pooling_type: str = "attn",
        patch_size: int = 128,
    ):
        """
        pooling_type:
          - "mean": simple average over time steps
          - "last": use the LSTM's final hidden state
          - "attn": use a learned attention pooling mechanism
        """
        super(SimpleModel, self).__init__()
        self.encoder_length = encoder_length
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.num_attention_layers = num_attention_layers
        self.pooling_type = pooling_type

        self.input_proj = nn.Linear(num_historical_features, hidden_size)

        self.patch_embed = PatchEmbed(
            num_historical_features, patch_size=patch_size, d_model=hidden_size
        )

        n_patches = math.ceil(encoder_length / patch_size)

        self.pos_enc = PositionalEncoding(hidden_size, max_len=n_patches)

        self.tcn = DilatedTCN(
            in_ch=hidden_size, hidden_ch=hidden_size, ks=3, n_layers=4
        )

        self.encoder_lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Stack multiple Transformer encoder blocks (attention layers)
        self.attention_layers = nn.ModuleList(
            [
                TransformerEncoderBlockPreNorm(
                    hidden_size, n_heads, p_attn=0.1, p_ff=0.3
                )
                for _ in range(num_attention_layers)
            ]
        )

        # self.attention_layers = nn.ModuleList(
        #     [
        #         TransformerEncoderBlock(
        #             d_model=hidden_size, n_heads=n_heads, dropout=dropout
        #         )
        #         for _ in range(num_attention_layers)
        #     ]
        # )

        self.pool = AttentionPooling(hidden_size)

        # Final classification layer (adjust output size as needed)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3),  # 3 output logits
        )

        self.ret_head = nn.Linear(hidden_size, 1)  # régression

        self.vol_head = nn.Linear(hidden_size, 1)

    def forward(self, historical_input):
        """
        historical_input: shape (batch, encoder_length, num_historical_features)
        """
        # 1) Project input to hidden_size
        x = self.patch_embed(historical_input)

        # x = self.input_proj(historical_input)

        x = self.pos_enc(x)

        x = self.tcn(x)

        # 2) LSTM encoding
        enc_output, (h, c) = self.encoder_lstm(
            x
        )  # enc_output: (batch, seq_len, hidden_size)

        # Apply stacked attention layers
        for layer in self.attention_layers:
            enc_output = layer(enc_output)

            # Learned attention pooling
        seq_rep = self.pool(enc_output)  # (batch, hidden_size)

        # 4) Final classification (logits)
        logits = self.fc_out(seq_rep)  # (batch, 3)

        return logits


class Runner:
    """
    Runner class to run the model.
    """

    def __init__(
        self,
        num_historical_features: int,
        encoder_length: int,
        model_path: str,
        hidden_size: int,
        dropout: float,
        lstm_layers: int,
        n_heads: int,
        num_attention_layers: int,
        pooling_type: str,
        patch_size: int,
    ):
        self.model = SimpleModel(
            num_historical_features=num_historical_features,
            encoder_length=encoder_length,
            hidden_size=hidden_size,
            dropout=dropout,
            lstm_layers=lstm_layers,
            n_heads=n_heads,
            num_attention_layers=num_attention_layers,
            pooling_type=pooling_type,
            patch_size=patch_size,
        )

        checkpoint = torch.load(
            model_path, weights_only=False, map_location=torch.device("cpu")
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])

    def run(self, data: torch.Tensor):
        """
        Run the model on the data.
        """

        print(f"data.shape: {data.shape}")

        return self.model(data)
