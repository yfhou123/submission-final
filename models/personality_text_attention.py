from __future__ import annotations

import torch
import torch.nn as nn


class PersonalityTextCrossAttention(nn.Module):
    """
    HOPE-style interaction: personality is Query, ASR text segments are Key/Value.

    Inputs:
        personality: [B, Dp]
        text_segments: [B, P, S, Dt], P is sample/pair count, S is text segment count
        text_mask: [B, P, S], 1 for valid text segments, 0 for padding

    Outputs:
        pairwise text features: [B, P, H]
    """

    def __init__(
        self,
        personality_dim: int = 1024,
        text_dim: int = 1024,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.hidden_dim = hidden_dim
        self.personality_proj = nn.Sequential(
            nn.LayerNorm(personality_dim),
            nn.Linear(personality_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        personality: torch.Tensor,
        text_segments: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if text_segments.ndim != 4:
            raise ValueError(f"text_segments must be [B,P,S,D], got shape={tuple(text_segments.shape)}")
        if personality.ndim != 2:
            raise ValueError(f"personality must be [B,D], got shape={tuple(personality.shape)}")

        batch_size, pair_count, segment_count, _text_dim = text_segments.shape
        if personality.shape[0] != batch_size:
            raise ValueError("personality and text_segments batch size mismatch")

        if text_mask is None:
            text_mask = torch.ones(
                batch_size,
                pair_count,
                segment_count,
                dtype=torch.bool,
                device=text_segments.device,
            )
        else:
            text_mask = text_mask.to(device=text_segments.device).bool()
            if tuple(text_mask.shape) != (batch_size, pair_count, segment_count):
                raise ValueError(f"text_mask must be [B,P,S], got shape={tuple(text_mask.shape)}")

        valid_pair_mask = text_mask.any(dim=-1)
        safe_text_mask = text_mask.clone().reshape(batch_size * pair_count, segment_count)
        empty_pairs = (~valid_pair_mask).reshape(batch_size * pair_count)
        if empty_pairs.any():
            safe_text_mask[empty_pairs, 0] = True

        text_hidden = self.text_proj(text_segments)
        flat_text = text_hidden.reshape(batch_size * pair_count, segment_count, self.hidden_dim)
        flat_mask = safe_text_mask

        personality_hidden = self.personality_proj(personality)
        pair_query = personality_hidden.unsqueeze(1).expand(batch_size, pair_count, self.hidden_dim)
        flat_query = pair_query.reshape(batch_size * pair_count, 1, self.hidden_dim)

        attn_out, weights = self.attn(
            query=flat_query,
            key=flat_text,
            value=flat_text,
            key_padding_mask=~flat_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        out = self.norm1(flat_query + attn_out)
        out = self.norm2(out + self.ffn(out)).squeeze(1)
        out = out.reshape(batch_size, pair_count, self.hidden_dim)
        out = out * valid_pair_mask.unsqueeze(-1).float()

        if not return_weights:
            return out

        weights = weights.squeeze(1).reshape(batch_size, pair_count, segment_count)
        weights = weights.masked_fill(~text_mask, 0.0)
        weights = weights * valid_pair_mask.unsqueeze(-1).float()
        return out, weights
