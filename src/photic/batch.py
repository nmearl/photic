from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(slots=True)
class NPBatch:
    context_x: torch.Tensor
    context_y: torch.Tensor
    context_yerr: torch.Tensor
    context_band: torch.Tensor
    context_mask: torch.Tensor

    target_x: torch.Tensor
    target_y: torch.Tensor
    target_yerr: torch.Tensor
    target_band: torch.Tensor
    target_mask: torch.Tensor

    labels: torch.Tensor | None = None
    redshift: torch.Tensor | None = None
    morph_targets: torch.Tensor | None = None
    metadata: dict[str, torch.Tensor] | None = None

    def to(self, device: torch.device | str) -> "NPBatch":
        md = None if self.metadata is None else {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in self.metadata.items()
        }
        return NPBatch(
            context_x=self.context_x.to(device),
            context_y=self.context_y.to(device),
            context_yerr=self.context_yerr.to(device),
            context_band=self.context_band.to(device),
            context_mask=self.context_mask.to(device),
            target_x=self.target_x.to(device),
            target_y=self.target_y.to(device),
            target_yerr=self.target_yerr.to(device),
            target_band=self.target_band.to(device),
            target_mask=self.target_mask.to(device),
            labels=None if self.labels is None else self.labels.to(device),
            redshift=None if self.redshift is None else self.redshift.to(device),
            morph_targets=None if self.morph_targets is None else self.morph_targets.to(device),
            metadata=md,
        )


def pad_sequence_1d(values: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(v.numel() for v in values)
    out = torch.full((len(values), max_len), pad_value, dtype=values[0].dtype)
    for i, v in enumerate(values):
        out[i, : v.numel()] = v
    return out


def collate_irregular_samples(samples: Iterable[dict[str, Any]]) -> NPBatch:
    samples = list(samples)
    ctx_x = pad_sequence_1d([torch.as_tensor(s["context_x"], dtype=torch.float32) for s in samples])
    ctx_y = pad_sequence_1d([torch.as_tensor(s["context_y"], dtype=torch.float32) for s in samples])
    ctx_yerr = pad_sequence_1d([torch.as_tensor(s["context_yerr"], dtype=torch.float32) for s in samples], pad_value=1.0)
    ctx_band = pad_sequence_1d([torch.as_tensor(s["context_band"], dtype=torch.long) for s in samples]).long()
    ctx_mask = pad_sequence_1d([torch.ones(len(s["context_x"]), dtype=torch.float32) for s in samples])

    tgt_x = pad_sequence_1d([torch.as_tensor(s["target_x"], dtype=torch.float32) for s in samples])
    tgt_y = pad_sequence_1d([torch.as_tensor(s["target_y"], dtype=torch.float32) for s in samples])
    tgt_yerr = pad_sequence_1d([torch.as_tensor(s["target_yerr"], dtype=torch.float32) for s in samples], pad_value=1.0)
    tgt_band = pad_sequence_1d([torch.as_tensor(s["target_band"], dtype=torch.long) for s in samples]).long()
    tgt_mask = pad_sequence_1d([torch.ones(len(s["target_x"]), dtype=torch.float32) for s in samples])

    labels = None
    if "label" in samples[0] or "labels" in samples[0]:
        key = "label" if "label" in samples[0] else "labels"
        labels = torch.as_tensor([float(s[key]) for s in samples], dtype=torch.float32)

    redshift = None
    if "redshift" in samples[0]:
        redshift = torch.as_tensor([float(s["redshift"]) for s in samples], dtype=torch.float32)

    morph_targets = None
    if "morph_targets" in samples[0]:
        morph_targets = torch.stack([torch.as_tensor(s["morph_targets"], dtype=torch.float32) for s in samples], dim=0)

    return NPBatch(
        context_x=ctx_x,
        context_y=ctx_y,
        context_yerr=ctx_yerr,
        context_band=ctx_band,
        context_mask=ctx_mask,
        target_x=tgt_x,
        target_y=tgt_y,
        target_yerr=tgt_yerr,
        target_band=tgt_band,
        target_mask=tgt_mask,
        labels=labels,
        redshift=redshift,
        morph_targets=morph_targets,
    )
