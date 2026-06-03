from __future__ import annotations

import torch

from .depformer_text_mpdd import MPDDDepFormerTextFusion
from .heads_mpdd import PHQCoralHead, PHQRegressionHead


PHQ_MAX = 27.0


def predict_binary_from_coral(coral_logits: torch.Tensor) -> torch.Tensor:
    if coral_logits.ndim != 2 or coral_logits.size(-1) < 1:
        raise ValueError(f"coral_logits must be [B,K] with K>=1, got shape={tuple(coral_logits.shape)}")
    return (torch.sigmoid(coral_logits[:, 0]) > 0.5).long()


def predict_binary_ternary_from_coral(coral_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if coral_logits.ndim != 2 or coral_logits.size(-1) != 2:
        raise ValueError(f"coral_logits must be [B,2], got shape={tuple(coral_logits.shape)}")
    passed = (torch.sigmoid(coral_logits) > 0.5).long()
    q5 = passed[:, 0]
    q10 = passed[:, 1] & q5
    return q5, q5 + q10


class MPDDDepFormerTextCoral(MPDDDepFormerTextFusion):
    """
    Joint PHQ-threshold model on top of DepFormer-text sample fusion.

    Outputs one or two CORAL logits plus a bounded PHQ regression prediction
    in [0, 27]. One threshold is used for binary PHQ>=5; two thresholds are
    used for ternary PHQ>=5 and PHQ>=10.
    """

    def __init__(
        self,
        *args,
        coral_threshold_count: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if coral_threshold_count not in (1, 2):
            raise ValueError("MPDDDepFormerTextCoral expects one or two thresholds")
        fused_dim = self.depformer_d_model * 3
        hidden_dim = kwargs.get("hidden_dim", 64)
        dropout = kwargs.get("dropout", 0.3)
        self.coral_head = PHQCoralHead(
            d_in=fused_dim,
            hidden_dim=hidden_dim,
            n_thresholds=coral_threshold_count,
            dropout=dropout,
        )
        self.phq_regressor = PHQRegressionHead(
            d_in=fused_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def encode_subject(
        self,
        mfcc: torch.Tensor | None = None,
        opensmile: torch.Tensor | None = None,
        wav2vec: torch.Tensor | None = None,
        densenet: torch.Tensor | None = None,
        resnet: torch.Tensor | None = None,
        openface: torch.Tensor | None = None,
        personality: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        text_segments: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        return_sample_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        sample_repr = self.encode_sample_fusion(
            mfcc=mfcc,
            opensmile=opensmile,
            wav2vec=wav2vec,
            densenet=densenet,
            resnet=resnet,
            openface=openface,
            personality=personality,
            text_segments=text_segments,
            text_mask=text_mask,
            pair_mask=pair_mask,
        )
        fused, sample_attn = self.aggregate_sample_repr(sample_repr, pair_mask)
        if return_sample_attn:
            return fused, sample_attn
        return fused

    def forward(
        self,
        mfcc: torch.Tensor | None = None,
        opensmile: torch.Tensor | None = None,
        wav2vec: torch.Tensor | None = None,
        densenet: torch.Tensor | None = None,
        resnet: torch.Tensor | None = None,
        openface: torch.Tensor | None = None,
        gait: torch.Tensor | None = None,
        personality: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        text_segments: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del gait
        fused, sample_attn = self.encode_subject(
            mfcc=mfcc,
            opensmile=opensmile,
            wav2vec=wav2vec,
            densenet=densenet,
            resnet=resnet,
            openface=openface,
            personality=personality,
            pair_mask=pair_mask,
            text_segments=text_segments,
            text_mask=text_mask,
            return_sample_attn=True,
        )
        coral_logits = self.coral_head(fused)
        phq_pred = PHQ_MAX * torch.sigmoid(self.phq_regressor(fused))
        return {
            "coral_logits": coral_logits,
            "phq_pred": phq_pred,
            "sample_attn": sample_attn,
        }
