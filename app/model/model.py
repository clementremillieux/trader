import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


class DilatedTCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3, n_layers=4, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2**i
            pad = (kernel_size - 1) // 2 * dilation
            layers += [
                nn.Conv1d(
                    in_ch if i == 0 else hidden_ch,
                    hidden_ch,
                    kernel_size,
                    dilation=dilation,
                    padding=pad,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, D) → (B, D, T) → ... → (B, T, H)
        y = x.transpose(1, 2)
        y = self.net(y)
        return y.transpose(1, 2)


class TransformerBlockPreNorm(nn.Module):
    def __init__(self, d_model, n_heads, p_attn=0.1, p_ff=0.1, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=p_attn, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(p_ff),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.drop = nn.Dropout(p_ff)

    def forward(self, x):
        # x: (B, T, D)
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y)
        x = x + self.drop(attn_out)
        y = self.norm2(x)
        x = x + self.drop(self.ff(y))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.v = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        # x: (B, T, H)
        scores = torch.matmul(x, self.v)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class ImprovedModel(nn.Module):
    def __init__(
        self,
        num_features: int = 32,
        seq_len: int = 512,
        hidden_size: int = 512,
        tcn_layers: int = 4,
        transformer_layers: int = 3,
        n_heads: int = 4,
        patch_size: int = 64,
        pooling: str = "attn",  # "mean", "last", "attn"
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pooling = pooling

        # 1) Patch + Positional
        self.patch_size = patch_size
        self.n_patches = math.ceil(seq_len / patch_size)
        self.proj_patch = nn.Linear(num_features * patch_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, self.n_patches)

        # 2) TCN + skip
        self.tcn = DilatedTCN(
            hidden_size,
            hidden_size,
            kernel_size=3,
            n_layers=tcn_layers,
            dropout=dropout,
        )
        self.skip_proj = nn.Linear(hidden_size, hidden_size)

        # 3) Blocs Transformer
        self.transformer = nn.ModuleList(
            [
                TransformerBlockPreNorm(
                    hidden_size, n_heads, p_attn=dropout, p_ff=dropout, ff_mult=4
                )
                for _ in range(transformer_layers)
            ]
        )

        # 4) Pooling
        if pooling == "attn":
            self.pool = AttentionPooling(hidden_size)
        elif pooling == "last":
            self.pool = lambda x: x[:, -1, :]
        else:  # mean
            self.pool = lambda x: x.mean(dim=1)

        # 5) Têtes de sortie
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3),
        )

        self.regressor = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

        self.regressor_vol = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

    def forward(self, x):
        # x: (B, T=512, F=32)
        B, T, F = x.shape
        # → patches
        pad = (-T) % self.patch_size
        if pad:
            x = torch.cat([x, x.new_zeros(B, pad, F)], dim=1)

        n_patches = x.size(1) // self.patch_size

        x = x.view(B, n_patches, self.patch_size * F)
        x = self.proj_patch(x)  # (B, n_patches, H)
        x = self.pos_enc(x)

        # TCN + skip
        tcn_out = self.tcn(x)  # (B, n_patches, H)
        x = x + self.skip_proj(tcn_out)

        # Transformer
        for blk in self.transformer:
            x = blk(x)

        # Pooling
        rep = self.pool(x)  # (B, H)

        # Sorties
        logits = self.classifier(rep)  # (B, 3)

        return logits


class Runner:
    """
    Runner class to run the model.
    """

    def __init__(
        self,
        model_path: str,
    ):
        self.model = ImprovedModel(
            num_features=52,
            seq_len=1024,
            hidden_size=1024,
            dropout=0.5,
            n_heads=8,
            transformer_layers=8,
        )

        checkpoint = torch.load(model_path, map_location="cpu")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

        self.device = torch.device("cpu")

        self.model.to(self.device)

    @torch.no_grad()
    def run(self, data: torch.Tensor):
        """
        Run the model on the data.
        """

        return self.model(data)
