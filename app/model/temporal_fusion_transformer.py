"""Temporal Fusion Transformer-inspired architecture for multi-task crypto forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "selu":
        return nn.SELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "elu":
        return nn.ELU(alpha=1.0)
    raise ValueError(f"Unsupported activation: {name}")


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, dim)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """Implementation inspired by the TFT GRN block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.1,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = _get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.skip = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        gating = self.sigmoid(self.gate(x))
        x = x * gating
        x = x + residual
        return self.norm(x)


class TemporalConvBlock(nn.Module):
    """Multi-scale temporal convolutional block with residual connection."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_sizes: Tuple[int, ...] = (3, 5),
        dilations: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        branches = []
        for k in kernel_sizes:
            for d in dilations:
                padding = ((k - 1) // 2) * d
                branches.append(
                    nn.Sequential(
                        nn.Conv1d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=k,
                            dilation=d,
                            padding=padding,
                        ),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, hidden)
        y = 0.0
        for branch in self.branches:
            conv_in = x.transpose(1, 2)
            conv_out = branch(conv_in).transpose(1, 2)
            y = y + conv_out
        y = y / len(self.branches)
        y = self.proj(y)
        y = self.dropout(y)
        return self.norm(x + y)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_multiplier, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        attn_out, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + self.dropout(attn_out)
        y = self.ln2(x)
        x = x + self.dropout(self.ff(y))
        return x


class AttentionPooling(nn.Module):
    """Learned attention pooling over the temporal dimension."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_scores = self.v(torch.tanh(self.query(x)))  # (batch, time, 1)
        attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=-1).unsqueeze(-1)
        return (x * attn_weights).sum(dim=1)


@dataclass
class TFTConfig:
    input_dim: int
    seq_len: int
    tau_vocab_size: int
    hidden_dim: int = 256
    num_heads: int = 8
    num_transformer_blocks: int = 4
    dropout: float = 0.1
    conv_kernel_sizes: Tuple[int, ...] = (3, 5)
    conv_dilations: Tuple[int, ...] = (1, 2, 4)
    static_dim: int = 128


class TemporalFusionTransformer(nn.Module):
    """High capacity temporal model for multi-task forecasting."""

    def __init__(self, cfg: TFTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_projection = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.positional_encoding = PositionalEncoding(cfg.hidden_dim, cfg.seq_len)
        self.temporal_conv = TemporalConvBlock(
            cfg.hidden_dim,
            kernel_sizes=cfg.conv_kernel_sizes,
            dilations=cfg.conv_dilations,
            dropout=cfg.dropout,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_transformer_blocks)
            ]
        )
        self.temporal_layer_norm = nn.LayerNorm(cfg.hidden_dim)

        self.tau_embedding = nn.Embedding(cfg.tau_vocab_size, cfg.static_dim)
        self.static_projection = nn.Linear(cfg.input_dim, cfg.static_dim)
        self.context_fusion = GatedResidualNetwork(
            input_dim=cfg.hidden_dim + cfg.static_dim * 2,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )
        self.attention_pool = AttentionPooling(cfg.hidden_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 3),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        self.vol_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, tau_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # x: (batch, time, features)
        # tau_ids: (batch,)
        h = self.input_projection(x)
        h = self.positional_encoding(h)
        h = self.temporal_conv(h)
        for block in self.transformer_blocks:
            h = block(h)
        h = self.temporal_layer_norm(h)

        pooled = self.attention_pool(h)

        static_mean = x.mean(dim=1)
        static_context = self.static_projection(static_mean)
        tau_embed = self.tau_embedding(tau_ids.clamp(max=self.cfg.tau_vocab_size - 1))

        fusion = torch.cat([pooled, static_context, tau_embed], dim=-1)
        fused = self.context_fusion(fusion)

        logits = self.classifier(fused)
        ret = self.reg_head(fused).squeeze(-1)
        vol = self.vol_head(fused).squeeze(-1)

        return {"logits": logits, "ret": ret, "vol": vol}
