from __future__ import annotations

import torch

from .depformer_text_coral_mpdd import MPDDDepFormerTextCoral, PHQ_MAX
from .depformer_text_mpdd import SampleAttentionPooling


class MPDDDepFormerTextChunkCoral(MPDDDepFormerTextCoral):
    """
    Elder chunk-cache variant of the text CORAL model.

    The original Elder fusion order is preserved:
    A/V chunk attention -> per-sample A/V representation,
    per-sample personality-text interaction, concat per sample,
    then subject-level sample pooling.
    """

    def __init__(
        self,
        *args,
        coral_threshold_count: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(*args, coral_threshold_count=coral_threshold_count, **kwargs)
        av_dim = self.depformer_d_model * 2
        self.chunk_attn_pool = SampleAttentionPooling(
            input_dim=av_dim,
            hidden_dim=self.depformer_d_model,
            dropout=kwargs.get("dropout", 0.3),
        )

    @staticmethod
    def _flatten_chunk_feature(name: str, value: torch.Tensor | None) -> tuple[torch.Tensor, int, int, int]:
        if value is None:
            raise ValueError(f"Missing required feature tensor: {name}")
        if value.ndim != 5:
            raise ValueError(f"{name} must be [B,P,C,T,D], got shape={tuple(value.shape)}")
        batch_size, pair_count, chunk_count, seq_len, feature_dim = value.shape
        flat = value.reshape(batch_size * pair_count, chunk_count, seq_len, feature_dim)
        return flat, batch_size, pair_count, chunk_count

    def encode_chunked_av(
        self,
        mfcc: torch.Tensor,
        opensmile: torch.Tensor,
        wav2vec: torch.Tensor,
        densenet: torch.Tensor,
        resnet: torch.Tensor,
        openface: torch.Tensor,
        pair_mask: torch.Tensor | None,
        chunk_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_mfcc, batch_size, pair_count, chunk_count = self._flatten_chunk_feature("mfcc", mfcc)
        flat_opensmile, *_ = self._flatten_chunk_feature("opensmile", opensmile)
        flat_wav2vec, *_ = self._flatten_chunk_feature("wav2vec", wav2vec)
        flat_densenet, *_ = self._flatten_chunk_feature("densenet", densenet)
        flat_resnet, *_ = self._flatten_chunk_feature("resnet", resnet)
        flat_openface, *_ = self._flatten_chunk_feature("openface", openface)

        if chunk_mask is None:
            flat_chunk_mask = torch.ones(
                batch_size * pair_count,
                chunk_count,
                dtype=flat_mfcc.dtype,
                device=flat_mfcc.device,
            )
        else:
            if tuple(chunk_mask.shape) != (batch_size, pair_count, chunk_count):
                raise ValueError(f"chunk_mask must be [B,P,C], got shape={tuple(chunk_mask.shape)}")
            flat_chunk_mask = chunk_mask.to(device=flat_mfcc.device, dtype=flat_mfcc.dtype).reshape(
                batch_size * pair_count,
                chunk_count,
            )

        if pair_mask is not None:
            if tuple(pair_mask.shape) != (batch_size, pair_count):
                raise ValueError(f"pair_mask must be [B,P], got shape={tuple(pair_mask.shape)}")
            flat_pair_mask = pair_mask.to(device=flat_mfcc.device, dtype=flat_mfcc.dtype).reshape(
                batch_size * pair_count,
                1,
            )
            flat_chunk_mask = flat_chunk_mask * flat_pair_mask

        chunk_repr = self.encode_av_pairs(
            mfcc=flat_mfcc,
            opensmile=flat_opensmile,
            wav2vec=flat_wav2vec,
            densenet=flat_densenet,
            resnet=flat_resnet,
            openface=flat_openface,
            pair_mask=flat_chunk_mask,
        )
        sample_av_flat, chunk_attn_flat = self.chunk_attn_pool(chunk_repr, flat_chunk_mask)
        sample_av = sample_av_flat.view(batch_size, pair_count, -1)
        chunk_attn = chunk_attn_flat.view(batch_size, pair_count, chunk_count)

        if pair_mask is not None:
            pair_weights = pair_mask.to(device=sample_av.device, dtype=sample_av.dtype).unsqueeze(-1)
            sample_av = sample_av * pair_weights
            chunk_attn = chunk_attn * pair_weights
        return sample_av, chunk_attn

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
        chunk_mask: torch.Tensor | None = None,
        return_chunk_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if personality is None:
            raise ValueError("Missing required feature tensor: personality")
        if text_segments is None:
            raise ValueError("Missing required feature tensor: text_segments")

        sample_av, chunk_attn = self.encode_chunked_av(
            mfcc=mfcc,
            opensmile=opensmile,
            wav2vec=wav2vec,
            densenet=densenet,
            resnet=resnet,
            openface=openface,
            pair_mask=pair_mask,
            chunk_mask=chunk_mask,
        )
        text_pair_repr = self.text_interaction(
            personality=torch.nan_to_num(personality, nan=0.0, posinf=0.0, neginf=0.0),
            text_segments=torch.nan_to_num(text_segments, nan=0.0, posinf=0.0, neginf=0.0),
            text_mask=text_mask,
        )
        sample_repr = torch.cat([sample_av, text_pair_repr], dim=-1)
        if pair_mask is not None:
            sample_repr = sample_repr * pair_mask.to(device=sample_repr.device, dtype=sample_repr.dtype).unsqueeze(-1)
        if return_chunk_attn:
            return sample_repr, chunk_attn
        return sample_repr

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
        chunk_mask: torch.Tensor | None = None,
        text_segments: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        return_sample_attn: bool = False,
        return_chunk_attn: bool = False,
    ):
        sample_repr, chunk_attn = self.encode_sample_fusion(
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
            chunk_mask=chunk_mask,
            return_chunk_attn=True,
        )
        fused, sample_attn = self.aggregate_sample_repr(sample_repr, pair_mask)
        if return_sample_attn and return_chunk_attn:
            return fused, sample_attn, chunk_attn
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
        chunk_mask: torch.Tensor | None = None,
        text_segments: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del gait
        fused, sample_attn, chunk_attn = self.encode_subject(
            mfcc=mfcc,
            opensmile=opensmile,
            wav2vec=wav2vec,
            densenet=densenet,
            resnet=resnet,
            openface=openface,
            personality=personality,
            pair_mask=pair_mask,
            chunk_mask=chunk_mask,
            text_segments=text_segments,
            text_mask=text_mask,
            return_sample_attn=True,
            return_chunk_attn=True,
        )
        coral_logits = self.coral_head(fused)
        phq_pred = PHQ_MAX * torch.sigmoid(self.phq_regressor(fused))
        return {
            "coral_logits": coral_logits,
            "phq_pred": phq_pred,
            "sample_attn": sample_attn,
            "chunk_attn": chunk_attn,
        }
