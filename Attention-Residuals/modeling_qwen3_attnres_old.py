"""
Qwen3 with Block Attention Residuals (AttnRes).

Replaces standard additive residual connections with softmax attention over
previous block representations, as described in:
  "Attention Residuals" (Kimi Team, arXiv:2603.15031)

Architecture change (per layer):
  Standard:  h = residual + sublayer(norm(residual))
  AttnRes:   h_in = attend(blocks + partial)   # selective aggregation
             h_out = partial + sublayer(norm(h_in))

Block AttnRes groups layers into N blocks, maintaining O(N*d) instead of
O(L*d) memory.  Within each block, outputs accumulate via standard residuals;
attention is applied only across block-level summary representations.
"""

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn

# Re-use Qwen3 components directly from the installed transformers package.
# We only override DecoderLayer and Model; everything else is unchanged.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "venv_pkgs"))

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3MLP,
    Qwen3Attention,
    Qwen3RotaryEmbedding,
    Qwen3PreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import can_return_tuple, auto_docstring
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


# ---------------------------------------------------------------------------
# Config extension
# ---------------------------------------------------------------------------

class Qwen3AttnResConfig(Qwen3Config):
    """Qwen3Config extended with Block AttnRes hyper-parameters."""

    model_type = "qwen3_attnres"

    def __init__(self, attnres_num_blocks: int = 8, **kwargs):
        super().__init__(**kwargs)
        # Number of depth-attention blocks.  ~8 recovers most Full-AttnRes gain
        # while keeping memory at O(N*d).
        self.attnres_num_blocks = attnres_num_blocks


# ---------------------------------------------------------------------------
# Core Block-AttnRes operation
# ---------------------------------------------------------------------------

def block_attn_res(
    blocks: list[torch.Tensor],   # completed blocks  [B, T, D] each
    partial_block: torch.Tensor,  # current intra-block partial sum  [B, T, D]
    proj: nn.Linear,              # learned pseudo-query weight  (d,)
    norm: Qwen3RMSNorm,           # RMSNorm applied to keys before scoring
) -> torch.Tensor:
    """
    Attend over all block representations + the current partial block.

    Returns a [B, T, D] tensor that serves as input to the next sublayer,
    replacing the standard residual stream.
    """
    # Stack everything: shape [N+1, B, T, D]
    V = torch.stack(blocks + [partial_block], dim=0)

    # Keys = normalised values
    K = norm(V)

    # Scalar logit per (block, batch, token) via the single learned query
    # proj.weight shape: (1, D) → squeeze to (D,)
    query = proj.weight.view(-1)                              # (D,)
    logits = torch.einsum("d, n b t d -> n b t", query, K)   # (N+1, B, T)

    # Softmax across block dimension
    weights = logits.softmax(dim=0)                           # (N+1, B, T)

    # Weighted sum of values
    h = torch.einsum("n b t, n b t d -> b t d", weights, V)  # (B, T, D)
    return h


# ---------------------------------------------------------------------------
# Modified decoder layer
# ---------------------------------------------------------------------------

class Qwen3AttnResDecoderLayer(GradientCheckpointingLayer):
    """
    Qwen3 decoder layer with Block AttnRes residuals.

    Compared to the standard layer:
      • receives (blocks, partial_block) instead of a single hidden_states
      • computes attended h_in = block_attn_res(...) before each sublayer
      • accumulates output into partial_block (not a running total)
      • returns updated (blocks, partial_block)

    Each layer owns two (proj, norm) pairs — one for the attention sublayer
    and one for the MLP sublayer.
    """

    def __init__(self, config: Qwen3AttnResConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Standard Qwen3 sublayers
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

        # AttnRes components — one pseudo-query Linear(d, 1) per sublayer.
        # We use bias=False and only rely on weight (d,) as the query vector.
        self.attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.attn_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Block boundary: how many transformer layers per block
        num_layers = config.num_hidden_layers
        num_blocks = getattr(config, "attnres_num_blocks", 8)
        # layers_per_block rounded up so all layers are covered
        self.layers_per_block = max(1, (num_layers + num_blocks - 1) // num_blocks)

    @property
    def is_block_boundary(self) -> bool:
        """True when this layer is the last in its block (0-indexed)."""
        return (self.layer_idx + 1) % self.layers_per_block == 0

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        # ---- Attention sublayer ----
        # Compute attended input for self-attention
        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        # Block boundary: save current partial_block, start fresh
        if self.is_block_boundary:
            blocks = blocks + [partial_block]   # non-destructive append
            partial_block = None

        # Self-attention (PreNorm)
        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(h),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        partial_block = attn_out if partial_block is None else partial_block + attn_out

        # ---- MLP sublayer ----
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

        mlp_out = self.mlp(self.post_attention_layernorm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


# ---------------------------------------------------------------------------
# Model backbone
# ---------------------------------------------------------------------------

class Qwen3AttnResModel(Qwen3PreTrainedModel):
    """Qwen3 backbone with Block AttnRes replacing standard residuals."""

    config_class = Qwen3AttnResConfig

    def __init__(self, config: Qwen3AttnResConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3AttnResDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = dict(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Block AttnRes state: list of completed block tensors + current partial
        # The token embedding acts as the first "block" (block 0).
        blocks: list[torch.Tensor] = [inputs_embeds]
        partial_block: torch.Tensor = inputs_embeds

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                blocks, partial_block = self._gradient_checkpointing_func(
                    layer.__call__,
                    blocks,
                    partial_block,
                    causal_mask_mapping[layer.attention_type],
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                blocks, partial_block = layer(
                    blocks=blocks,
                    partial_block=partial_block,
                    attention_mask=causal_mask_mapping[layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        hidden_states = self.norm(partial_block)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ---------------------------------------------------------------------------
# Causal LM head
# ---------------------------------------------------------------------------

class Qwen3AttnResForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Qwen3 causal LM with Block AttnRes residuals."""

    config_class = Qwen3AttnResConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen3AttnResConfig):
        super().__init__(config)
        self.model = Qwen3AttnResModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_idx, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
