from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualClassificationHead(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.classifier_ce = ClassificationHead(d_in, hidden_dim, num_classes, dropout)
        self.classifier_focal = ClassificationHead(d_in, hidden_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classifier_ce(x), self.classifier_focal(x)

    @staticmethod
    def merge_logits(logits_ce: torch.Tensor, logits_focal: torch.Tensor) -> torch.Tensor:
        return (logits_ce + logits_focal) * 0.5


class MulticlassFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.register_buffer("alpha", alpha.float() if alpha is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        targets = targets.long()
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        loss = -((1.0 - target_probs).clamp_min(0.0) ** self.gamma) * target_log_probs
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).gather(0, targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def dual_classification_loss(
    logits_ce: torch.Tensor,
    logits_focal: torch.Tensor,
    targets: torch.Tensor,
    ce_weight: torch.Tensor | None = None,
    focal_alpha: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    ce_loss_weight: float = 1.0,
    focal_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_ce = F.cross_entropy(logits_ce, targets.long(), weight=ce_weight)
    focal = MulticlassFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    loss_focal = focal(logits_focal, targets.long())
    loss = ce_loss_weight * loss_ce + focal_loss_weight * loss_focal
    return loss, loss_ce, loss_focal


class PHQRegressionHead(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PHQCoralHead(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int, n_thresholds: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        if n_thresholds < 1:
            raise ValueError("n_thresholds must be >= 1")
        self.score = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_thresholds = nn.Parameter(torch.full((n_thresholds,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.score(x).squeeze(-1)
        thresholds = torch.cumsum(F.softplus(self.raw_thresholds), dim=0)
        return score.unsqueeze(-1) - thresholds.view(1, -1)

    @staticmethod
    def predict_level(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        mono_probs = []
        prev = probs[..., 0]
        mono_probs.append(prev)
        for idx in range(1, probs.size(-1)):
            prev = torch.minimum(probs[..., idx], prev)
            mono_probs.append(prev)
        return torch.stack(mono_probs, dim=-1).sum(dim=-1)


def parse_phq_thresholds(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("ordinal thresholds cannot be empty")
        thresholds = tuple(float(part) for part in parts)
    else:
        thresholds = tuple(float(part) for part in value)
    if sorted(thresholds) != list(thresholds):
        raise ValueError(f"ordinal thresholds must be sorted ascending, got {thresholds}")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError(f"ordinal thresholds must be unique, got {thresholds}")
    return thresholds


def build_phq_ordinal_targets(phq: torch.Tensor, thresholds: Sequence[float]) -> torch.Tensor:
    threshold_tensor = phq.new_tensor(tuple(thresholds), dtype=torch.float32)
    return (phq.float().unsqueeze(-1) >= threshold_tensor.view(1, -1)).float()


def phq_ordinal_loss(
    logits: torch.Tensor,
    phq: torch.Tensor,
    thresholds: Sequence[float],
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    targets = build_phq_ordinal_targets(phq, thresholds)
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
