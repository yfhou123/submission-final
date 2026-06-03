from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _masked_avg_pool(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    weights = mask.unsqueeze(-1).float()
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


class FeatureAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_size * 2, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            out, _ = self.lstm(x)
        else:
            lengths = mask.long().sum(dim=1).clamp_min(1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        out = torch.tanh(self.proj(out))
        out = self.drop(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff or d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff or d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = None
        if kv_mask is not None:
            kv_mask = kv_mask.bool()
            empty_kv = ~kv_mask.any(dim=1)
            if empty_kv.any():
                kv_mask = kv_mask.clone()
                kv_mask[empty_kv] = True
            key_padding_mask = ~kv_mask

        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
        )
        query = self.norm1(query + self.drop(attn_out))
        out = self.norm2(query + self.ffn(query))
        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1).float()
        return out


class BimodalCollaborativeTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        n_heads: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.v2a_layers = nn.ModuleList(
            CrossAttentionBlock(d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        )
        self.a2v_layers = nn.ModuleList(
            CrossAttentionBlock(d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        )

    def forward(
        self,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = audio_feat
        video = video_feat
        for v2a, a2v in zip(self.v2a_layers, self.a2v_layers):
            audio_next = v2a(audio, video, query_mask=audio_mask, kv_mask=video_mask)
            video_next = a2v(video, audio, query_mask=video_mask, kv_mask=audio_mask)
            audio, video = audio_next, video_next
        return audio, video


class MPDDDepFormerBaseline(nn.Module):
    SUBTRACKS = {"A-V+P"}

    def __init__(
        self,
        subtrack: str = "A-V+P",
        num_classes: int = 3,
        is_regression: bool = False,
        use_regression_head: bool = True,
        mfcc_dim: int = 64,
        opensmile_dim: int = 65,
        wav2vec_dim: int = 768,
        densenet_dim: int = 1000,
        resnet_dim: int = 1000,
        openface_dim: int = 710,
        gait_dim: int = 9,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        encoder_type: str = "bilstm_mean",
        depformer_d_model: int = 256,
        depformer_adapter_dim: int = 128,
        depformer_lstm_layers: int = 1,
        depformer_bct_layers: int = 1,
        depformer_heads: int = 2,
        personality_dim: int = 1024,
    ) -> None:
        super().__init__()
        if subtrack not in self.SUBTRACKS:
            raise ValueError("MPDDDepFormerBaseline supports only A-V+P inputs")
        if depformer_d_model % depformer_heads != 0:
            raise ValueError(
                f"depformer_d_model={depformer_d_model} must be divisible by "
                f"depformer_heads={depformer_heads}"
            )
        if depformer_d_model < 2:
            raise ValueError("depformer_d_model must be >= 2")

        self.subtrack = subtrack
        self.is_regression = is_regression
        self.use_regression_head = use_regression_head
        self.depformer_d_model = depformer_d_model
        self.encoder_type = encoder_type
        self.gait_dim = gait_dim

        self.audio_adapters = nn.ModuleDict(
            {
                "mfcc": FeatureAdapter(mfcc_dim, depformer_adapter_dim, dropout),
                "opensmile": FeatureAdapter(opensmile_dim, depformer_adapter_dim, dropout),
                "wav2vec": FeatureAdapter(wav2vec_dim, depformer_adapter_dim, dropout),
            }
        )
        self.video_adapters = nn.ModuleDict(
            {
                "densenet": FeatureAdapter(densenet_dim, depformer_adapter_dim, dropout),
                "resnet": FeatureAdapter(resnet_dim, depformer_adapter_dim, dropout),
                "openface": FeatureAdapter(openface_dim, depformer_adapter_dim, dropout),
            }
        )
        self.audio_fusion = nn.Sequential(
            nn.LayerNorm(depformer_adapter_dim * 3),
            nn.Linear(depformer_adapter_dim * 3, depformer_d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.video_fusion = nn.Sequential(
            nn.LayerNorm(depformer_adapter_dim * 3),
            nn.Linear(depformer_adapter_dim * 3, depformer_d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        lstm_hidden = max(1, depformer_d_model // 2)
        self.audio_encoder = LSTMEncoder(
            depformer_d_model,
            hidden_size=lstm_hidden,
            num_layers=depformer_lstm_layers,
            dropout=dropout,
        )
        self.video_encoder = LSTMEncoder(
            depformer_d_model,
            hidden_size=lstm_hidden,
            num_layers=depformer_lstm_layers,
            dropout=dropout,
        )
        self.bct = BimodalCollaborativeTransformer(
            depformer_d_model,
            n_layers=depformer_bct_layers,
            n_heads=depformer_heads,
            dropout=dropout,
        )

        self.personality_proj = nn.Sequential(
            nn.LayerNorm(personality_dim),
            nn.Linear(personality_dim, depformer_d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fused_dim = depformer_d_model * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1 if is_regression else num_classes),
        )
        if use_regression_head:
            self.regressor = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        self._init_weights()

    @staticmethod
    def _require_feature(name: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"Missing required feature tensor: {name}")
        if value.ndim != 4:
            raise ValueError(f"Expected {name} shape [B, P, T, D], got {tuple(value.shape)}")
        return value

    @staticmethod
    def _sequence_mask(x: torch.Tensor, pair_mask: torch.Tensor | None) -> torch.Tensor:
        mask = torch.any(torch.abs(x) > 0, dim=-1)
        if pair_mask is not None:
            mask = mask & pair_mask.bool().unsqueeze(-1)
        return mask

    @staticmethod
    def _aggregate_pairs(pair_repr: torch.Tensor, pair_mask: torch.Tensor | None) -> torch.Tensor:
        if pair_mask is None:
            return pair_repr.mean(dim=1)
        weights = pair_mask.unsqueeze(-1).float()
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (pair_repr * weights).sum(dim=1) / denom

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode_av(
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
        return self._aggregate_pairs(pair_repr, pair_mask)

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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del gait
        if personality is None:
            raise ValueError("Missing required feature tensor: personality")

        mfcc = self._require_feature("mfcc", mfcc)
        opensmile = self._require_feature("opensmile", opensmile)
        wav2vec = self._require_feature("wav2vec", wav2vec)
        densenet = self._require_feature("densenet", densenet)
        resnet = self._require_feature("resnet", resnet)
        openface = self._require_feature("openface", openface)

        av_repr = self._encode_av(
            mfcc=mfcc,
            opensmile=opensmile,
            wav2vec=wav2vec,
            densenet=densenet,
            resnet=resnet,
            openface=openface,
            pair_mask=pair_mask,
        )
        personality_repr = self.personality_proj(
            torch.nan_to_num(personality, nan=0.0, posinf=0.0, neginf=0.0)
        )
        fused = torch.cat([av_repr, personality_repr], dim=-1)
        logits = self.classifier(fused)
        if self.use_regression_head:
            return logits, self.regressor(fused).squeeze(-1)
        return logits
