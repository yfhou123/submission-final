from __future__ import annotations

import torch
import torch.nn as nn

from .depformer_mpdd import MPDDDepFormerBaseline, _masked_avg_pool
from .personality_text_attention import PersonalityTextCrossAttention


class SampleAttentionPooling(nn.Module):
    """Learn a masked weighted average over per-sample representations."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        sample_repr: torch.Tensor,
        pair_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sample_repr.ndim != 3:
            raise ValueError(f"sample_repr must be [B,P,D], got shape={tuple(sample_repr.shape)}")

        batch_size, pair_count, _dim = sample_repr.shape
        if pair_mask is None:
            valid_mask = torch.ones(batch_size, pair_count, dtype=torch.bool, device=sample_repr.device)
        else:
            if tuple(pair_mask.shape) != (batch_size, pair_count):
                raise ValueError(f"pair_mask must be [B,P], got shape={tuple(pair_mask.shape)}")
            valid_mask = pair_mask.to(device=sample_repr.device).bool()

        scores = self.scorer(sample_repr).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * valid_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(sample_repr * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class MPDDDepFormerTextFusion(MPDDDepFormerBaseline):
    """
    HOPE-like sample-level fusion:
    each sample/pair first gets DepFormer A-V features and personality-text
    interaction features, then the fused sample features are aggregated to a
    subject-level representation.
    """

    def __init__(
        self,
        *args,
        text_dim: int = 1024,
        sample_pooling: str = "mean",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if sample_pooling not in {"mean", "attention"}:
            raise ValueError("sample_pooling must be 'mean' or 'attention'")
        self.text_dim = text_dim
        self.sample_pooling = sample_pooling
        self.text_interaction = PersonalityTextCrossAttention(
            personality_dim=kwargs.get("personality_dim", 1024),
            text_dim=text_dim,
            hidden_dim=self.depformer_d_model,
            num_heads=kwargs.get("depformer_heads", 2),
            dropout=kwargs.get("dropout", 0.3),
        )
        fused_dim = self.depformer_d_model * 3
        self.sample_attn_pool = SampleAttentionPooling(
            input_dim=fused_dim,
            hidden_dim=self.depformer_d_model,
            dropout=kwargs.get("dropout", 0.3),
        )

    @staticmethod
    def _mean_pooling_weights(
        sample_repr: torch.Tensor,
        pair_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, pair_count, _dim = sample_repr.shape
        if pair_mask is None:
            return torch.full(
                (batch_size, pair_count),
                1.0 / max(1, pair_count),
                dtype=sample_repr.dtype,
                device=sample_repr.device,
            )
        weights = pair_mask.to(device=sample_repr.device, dtype=sample_repr.dtype)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)

    def aggregate_sample_repr(
        self,
        sample_repr: torch.Tensor,
        pair_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sample_pooling == "attention":
            return self.sample_attn_pool(sample_repr, pair_mask)
        return self._aggregate_pairs(sample_repr, pair_mask), self._mean_pooling_weights(sample_repr, pair_mask)

    def encode_av_pairs(
        self,
        mfcc: torch.Tensor,
        opensmile: torch.Tensor,
        wav2vec: torch.Tensor,
        densenet: torch.Tensor,
        resnet: torch.Tensor,
        openface: torch.Tensor,
        pair_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, pair_count, seq_len, _ = mfcc.shape

        audio_parts = [
            self.audio_adapters["mfcc"](mfcc),
            self.audio_adapters["opensmile"](opensmile),
            self.audio_adapters["wav2vec"](wav2vec),
        ]
        video_parts = [
            self.video_adapters["densenet"](densenet),
            self.video_adapters["resnet"](resnet),
            self.video_adapters["openface"](openface),
        ]
        audio = self.audio_fusion(torch.cat(audio_parts, dim=-1))
        video = self.video_fusion(torch.cat(video_parts, dim=-1))

        audio_mask = self._sequence_mask(audio, pair_mask)
        video_mask = self._sequence_mask(video, pair_mask)
        flat_audio = audio.reshape(batch_size * pair_count, seq_len, -1)
        flat_video = video.reshape(batch_size * pair_count, seq_len, -1)
        flat_audio_mask = audio_mask.reshape(batch_size * pair_count, seq_len)
        flat_video_mask = video_mask.reshape(batch_size * pair_count, seq_len)

        audio_encoded = self.audio_encoder(flat_audio, flat_audio_mask)
        video_encoded = self.video_encoder(flat_video, flat_video_mask)
        audio_enhanced, video_enhanced = self.bct(
            audio_encoded,
            video_encoded,
            audio_mask=flat_audio_mask,
            video_mask=flat_video_mask,
        )

        audio_pooled = _masked_avg_pool(audio_enhanced, flat_audio_mask)
        video_pooled = _masked_avg_pool(video_enhanced, flat_video_mask)
        pair_repr = torch.cat([audio_pooled, video_pooled], dim=-1)
        pair_repr = pair_repr.view(batch_size, pair_count, -1)
        if pair_mask is not None:
            pair_repr = pair_repr * pair_mask.unsqueeze(-1).float()
        return pair_repr

    def encode_sample_fusion(
        self,
        mfcc: torch.Tensor | None = None,
        opensmile: torch.Tensor | None = None,
        wav2vec: torch.Tensor | None = None,
        densenet: torch.Tensor | None = None,
        resnet: torch.Tensor | None = None,
        openface: torch.Tensor | None = None,
        personality: torch.Tensor | None = None,
        text_segments: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if personality is None:
            raise ValueError("Missing required feature tensor: personality")
        if text_segments is None:
            raise ValueError("Missing required feature tensor: text_segments")

        mfcc = self._require_feature("mfcc", mfcc)
        opensmile = self._require_feature("opensmile", opensmile)
        wav2vec = self._require_feature("wav2vec", wav2vec)
        densenet = self._require_feature("densenet", densenet)
        resnet = self._require_feature("resnet", resnet)
        openface = self._require_feature("openface", openface)

        av_pair_repr = self.encode_av_pairs(
            mfcc=mfcc,
            opensmile=opensmile,
            wav2vec=wav2vec,
            densenet=densenet,
            resnet=resnet,
            openface=openface,
            pair_mask=pair_mask,
        )
        text_pair_repr = self.text_interaction(
            personality=torch.nan_to_num(personality, nan=0.0, posinf=0.0, neginf=0.0),
            text_segments=torch.nan_to_num(text_segments, nan=0.0, posinf=0.0, neginf=0.0),
            text_mask=text_mask,
        )
        sample_repr = torch.cat([av_pair_repr, text_pair_repr], dim=-1)
        if pair_mask is not None:
            sample_repr = sample_repr * pair_mask.unsqueeze(-1).float()
        return sample_repr

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del gait
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
        fused, _sample_attn = self.aggregate_sample_repr(sample_repr, pair_mask)
        logits = self.classifier(fused)
        if self.use_regression_head:
            return logits, self.regressor(fused).squeeze(-1)
        return logits
