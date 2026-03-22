"""Tests for standalone LoRA analytics module."""
from __future__ import annotations

import torch
import pytest

from serenityboard.lora_analytics import (
    _pair_lora_keys,
    analyze_lora_layer,
    analyze_lora_state_dict,
    diagnose,
)


# ---------------------------------------------------------------------------
# Key pairing
# ---------------------------------------------------------------------------

class TestKeyPairing:

    def test_diffusers_format(self) -> None:
        sd = {
            "transformer.block.0.attn.to_q.lora_A.weight": torch.randn(8, 512),
            "transformer.block.0.attn.to_q.lora_B.weight": torch.randn(512, 8),
        }
        pairs = _pair_lora_keys(sd)
        assert len(pairs) == 1
        assert "transformer.block.0.attn.to_q" in pairs

    def test_native_format(self) -> None:
        sd = {
            "block.0.lora_down.weight": torch.randn(8, 512),
            "block.0.lora_up.weight": torch.randn(512, 8),
        }
        pairs = _pair_lora_keys(sd)
        assert len(pairs) == 1
        assert "block.0" in pairs

    def test_simpletuner_format(self) -> None:
        sd = {
            "transformer.block.0.attn.to_q.lora.down.weight": torch.randn(8, 512),
            "transformer.block.0.attn.to_q.lora.up.weight": torch.randn(512, 8),
        }
        pairs = _pair_lora_keys(sd)
        assert len(pairs) == 1
        assert "transformer.block.0.attn.to_q" in pairs

    def test_skips_1d_tensors(self) -> None:
        sd = {
            "block.0.lora_down.weight": torch.randn(8, 512),
            "block.0.lora_up.weight": torch.randn(512, 8),
            "block.0.alpha": torch.tensor(8.0),  # 0D
            "block.0.bias": torch.randn(512),    # 1D
        }
        pairs = _pair_lora_keys(sd)
        assert len(pairs) == 1

    def test_unpaired_keys_skipped(self) -> None:
        sd = {
            "block.0.lora_down.weight": torch.randn(8, 512),
            # No matching lora_up
        }
        pairs = _pair_lora_keys(sd)
        assert len(pairs) == 0

    def test_multiple_layers(self) -> None:
        sd = {}
        for i in range(5):
            sd[f"block.{i}.lora_A.weight"] = torch.randn(4, 256)
            sd[f"block.{i}.lora_B.weight"] = torch.randn(256, 4)
        pairs = _pair_lora_keys(sd)
        assert len(pairs) == 5


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------

class TestAnalyzeLayer:

    def test_basic_metrics(self) -> None:
        a = torch.randn(8, 512) * 0.01
        b = torch.randn(512, 8) * 0.01
        m = analyze_lora_layer(a, b)

        assert "a_l1_norm" in m
        assert "a_l2_norm" in m
        assert "a_spectral_norm" in m
        assert "b_l1_norm" in m
        assert "b_spectral_norm" in m
        assert "effective_rank" in m
        assert "weight_magnitude" in m
        assert "ba_spectral_norm" in m
        assert "condition_number" in m
        assert "ab_ratio" in m

    def test_norms_positive(self) -> None:
        a = torch.randn(4, 128)
        b = torch.randn(128, 4)
        m = analyze_lora_layer(a, b)

        assert m["a_l1_norm"] > 0
        assert m["a_l2_norm"] > 0
        assert m["a_spectral_norm"] > 0
        assert m["b_spectral_norm"] > 0
        assert m["ba_spectral_norm"] > 0
        assert m["weight_magnitude"] > 0

    def test_effective_rank_bounded(self) -> None:
        """Effective rank should be between 1 and min(rank, out, in)."""
        rank = 8
        a = torch.randn(rank, 512)
        b = torch.randn(512, rank)
        m = analyze_lora_layer(a, b)

        assert 0 < m["effective_rank"] <= rank + 0.1

    def test_identity_like_condition(self) -> None:
        """Identity-like BA should have condition number near 1."""
        a = torch.eye(4, 4)
        b = torch.eye(4, 4)
        m = analyze_lora_layer(a, b)
        assert abs(m["condition_number"] - 1.0) < 0.01

    def test_zeros_handled(self) -> None:
        """All-zero matrices shouldn't crash."""
        a = torch.zeros(4, 128)
        b = torch.zeros(128, 4)
        m = analyze_lora_layer(a, b)
        assert m["a_spectral_norm"] == 0.0
        assert m["effective_rank"] == 0.0


# ---------------------------------------------------------------------------
# State dict analysis
# ---------------------------------------------------------------------------

class TestAnalyzeStateDict:

    def test_analyze_state_dict(self) -> None:
        sd = {}
        for i in range(3):
            sd[f"block.{i}.lora_A.weight"] = torch.randn(4, 128) * 0.01
            sd[f"block.{i}.lora_B.weight"] = torch.randn(128, 4) * 0.01
        metrics = analyze_lora_state_dict(sd)
        assert len(metrics) == 3
        for layer_metrics in metrics.values():
            assert "a_spectral_norm" in layer_metrics


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnose:

    def test_b_dominance_warning(self) -> None:
        """High ab_ratio across many layers triggers warning."""
        metrics = {}
        for i in range(10):
            metrics[f"block.{i}"] = {
                "a_l1_norm": 1.0, "a_l2_norm": 1.0, "a_spectral_norm": 0.1,
                "b_l1_norm": 10.0, "b_l2_norm": 10.0, "b_spectral_norm": 1.0,
                "effective_rank": 4.0, "weight_magnitude": 0.01,
                "ba_spectral_norm": 0.5, "condition_number": 10.0,
                "ab_ratio": 10.0,  # B dominates
            }
        warnings = diagnose(metrics)
        assert any("dominating" in w.lower() for w in warnings)

    def test_rank_collapse_warning(self) -> None:
        metrics = {
            "block.0.attn.q_proj": {
                "a_l1_norm": 1.0, "a_l2_norm": 1.0, "a_spectral_norm": 0.5,
                "b_l1_norm": 1.0, "b_l2_norm": 1.0, "b_spectral_norm": 0.5,
                "effective_rank": 1.1, "weight_magnitude": 0.01,
                "ba_spectral_norm": 0.5, "condition_number": 10.0,
                "ab_ratio": 1.0,
            }
        }
        warnings = diagnose(metrics)
        assert any("rank collapsed" in w.lower() for w in warnings)

    def test_high_condition_warning(self) -> None:
        metrics = {
            "block.0.mlp": {
                "a_l1_norm": 1.0, "a_l2_norm": 1.0, "a_spectral_norm": 0.5,
                "b_l1_norm": 1.0, "b_l2_norm": 1.0, "b_spectral_norm": 0.5,
                "effective_rank": 4.0, "weight_magnitude": 0.01,
                "ba_spectral_norm": 0.5, "condition_number": 5000.0,
                "ab_ratio": 1.0,
            }
        }
        warnings = diagnose(metrics)
        assert any("condition number" in w.lower() for w in warnings)

    def test_healthy_no_warnings(self) -> None:
        metrics = {
            "block.0": {
                "a_l1_norm": 1.0, "a_l2_norm": 1.0, "a_spectral_norm": 0.5,
                "b_l1_norm": 1.0, "b_l2_norm": 1.0, "b_spectral_norm": 0.5,
                "effective_rank": 4.0, "weight_magnitude": 0.01,
                "ba_spectral_norm": 0.5, "condition_number": 10.0,
                "ab_ratio": 1.0,
            }
        }
        warnings = diagnose(metrics)
        assert len(warnings) == 0

    def test_empty_metrics(self) -> None:
        assert diagnose({}) == []
