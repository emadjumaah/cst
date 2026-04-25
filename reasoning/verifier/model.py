from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class VerifierConfig:
    vocab_size: int
    max_len: int = 256
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    d_ff: int = 768
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VerifierModel(nn.Module):
    def __init__(self, cfg: VerifierConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seqlen = ids.shape
        pos = torch.arange(seqlen, device=ids.device).unsqueeze(0).expand(batch, seqlen)

        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        pad_mask = ~mask.bool()
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        mask_f = mask.float().unsqueeze(-1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        logits = self.head(pooled).squeeze(-1)
        return logits


def save_checkpoint(
    ckpt_path: Path,
    *,
    model: VerifierModel,
    cfg: VerifierConfig,
    pad_id: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "pad_id": int(pad_id),
    }
    if extra:
        payload.update(extra)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[VerifierModel, VerifierConfig, int, dict[str, Any]]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = VerifierConfig(**blob["config"])
    model = VerifierModel(cfg)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(device).eval()
    pad_id = int(blob.get("pad_id", 0))
    return model, cfg, pad_id, blob
