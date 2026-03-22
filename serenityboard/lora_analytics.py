"""Standalone LoRA weight analytics for SerenityBoard.

Lightweight copy of core metrics from serenity.lora.analytics that works
without Serenity installed. Only requires torch and safetensors.
"""
from __future__ import annotations

import torch

__all__ = [
    "analyze_lora_layer",
    "analyze_lora_state_dict",
    "analyze_lora_file",
    "compare_lora_files",
    "diagnose",
]


# ---------------------------------------------------------------------------
# Key name normalization
# ---------------------------------------------------------------------------

_A_SUFFIXES = (
    ".lora_down.weight", ".lora_a.weight", ".lora_A.weight",
    ".lora.down.weight",
)
_B_SUFFIXES = (
    ".lora_up.weight", ".lora_b.weight", ".lora_B.weight",
    ".lora.up.weight",
)


def _pair_lora_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Group A/B weight pairs by layer name."""
    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}

    for key in state_dict:
        if state_dict[key].ndim < 2:
            continue

        lower = key.lower()
        matched = False

        for suffix in _A_SUFFIXES:
            if lower.endswith(suffix.lower()):
                base = key[: len(key) - len(suffix)]
                a_keys[base] = key
                matched = True
                break

        if not matched:
            for suffix in _B_SUFFIXES:
                if lower.endswith(suffix.lower()):
                    base = key[: len(key) - len(suffix)]
                    b_keys[base] = key
                    break

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for base in sorted(set(a_keys) & set(b_keys)):
        pairs[base] = (state_dict[a_keys[base]], state_dict[b_keys[base]])
    return pairs


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------

def analyze_lora_layer(
    lora_a: torch.Tensor, lora_b: torch.Tensor,
) -> dict[str, float]:
    """Compute metrics for a single LoRA A/B pair."""
    a = lora_a.detach().float()
    b = lora_b.detach().float()

    a_l1 = a.abs().sum().item()
    a_l2 = a.norm().item()
    b_l1 = b.abs().sum().item()
    b_l2 = b.norm().item()

    a_sv = torch.linalg.svdvals(a)
    b_sv = torch.linalg.svdvals(b)
    a_spectral = a_sv[0].item()
    b_spectral = b_sv[0].item()

    # BA product — use Gram trick for large matrices
    ba = b @ a
    rank = min(a.shape[0], a.shape[1])
    if max(ba.shape) > 4096 and rank <= 128:
        if ba.shape[0] > ba.shape[1]:
            gram = ba.T @ ba
        else:
            gram = ba @ ba.T
        eigvals = torch.linalg.eigvalsh(gram)
        ba_sv = eigvals.clamp(min=0).sqrt().flip(0)
    else:
        ba_sv = torch.linalg.svdvals(ba)

    ba_spectral = ba_sv[0].item()
    nuclear = ba_sv.sum().item()
    effective_rank = nuclear / ba_spectral if ba_spectral > 1e-12 else 0.0
    weight_mag = ba.abs().mean().item()

    nonzero_sv = ba_sv[ba_sv > 1e-10]
    condition = (nonzero_sv[0] / nonzero_sv[-1]).item() if len(nonzero_sv) > 1 else 1.0
    ab_ratio = b_spectral / a_spectral if a_spectral > 1e-12 else float("inf")

    return {
        "a_l1_norm": a_l1,
        "a_l2_norm": a_l2,
        "a_spectral_norm": a_spectral,
        "b_l1_norm": b_l1,
        "b_l2_norm": b_l2,
        "b_spectral_norm": b_spectral,
        "effective_rank": effective_rank,
        "weight_magnitude": weight_mag,
        "ba_spectral_norm": ba_spectral,
        "condition_number": condition,
        "ab_ratio": ab_ratio,
    }


# ---------------------------------------------------------------------------
# State dict / file analysis
# ---------------------------------------------------------------------------

def analyze_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Analyze all LoRA layers in a state dict."""
    pairs = _pair_lora_keys(state_dict)
    return {name: analyze_lora_layer(a, b) for name, (a, b) in pairs.items()}


def analyze_lora_file(path: str) -> dict[str, dict[str, float]]:
    """Load a LoRA safetensors file and return per-layer analytics."""
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    return analyze_lora_state_dict(state)


def compare_lora_files(
    path_a: str, path_b: str,
) -> dict[str, dict]:
    """Compare two LoRA files, return per-layer metrics + diff percentages."""
    metrics_a = analyze_lora_file(path_a)
    metrics_b = analyze_lora_file(path_b)

    all_layers = sorted(set(metrics_a) | set(metrics_b))
    result: dict[str, dict] = {}

    for layer in all_layers:
        entry: dict = {}
        m_a = metrics_a.get(layer)
        m_b = metrics_b.get(layer)
        entry["lora1"] = m_a
        entry["lora2"] = m_b

        if m_a and m_b:
            for key in m_a:
                va, vb = m_a[key], m_b[key]
                if abs(va) > 1e-12:
                    entry[f"diff_{key}_pct"] = ((vb - va) / abs(va)) * 100.0
                else:
                    entry[f"diff_{key}_pct"] = 0.0 if abs(vb) < 1e-12 else float("inf")

        result[layer] = entry

    return result


def summary_stats(
    metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute aggregate summary from per-layer metrics."""
    if not metrics:
        return {}

    n = len(metrics)
    ab_ratios = [m["ab_ratio"] for m in metrics.values() if m["ab_ratio"] != float("inf")]
    eff_ranks = [m["effective_rank"] for m in metrics.values()]
    b_spectral = [m["b_spectral_norm"] for m in metrics.values()]

    return {
        "mean_ab_ratio": sum(ab_ratios) / len(ab_ratios) if ab_ratios else 0.0,
        "mean_effective_rank": sum(eff_ranks) / n,
        "max_b_spectral": max(b_spectral),
        "mean_b_spectral": sum(b_spectral) / n,
        "num_layers": n,
    }


def diagnose(metrics: dict[str, dict[str, float]]) -> list[str]:
    """Flag potential training issues from LoRA metrics."""
    warnings: list[str] = []
    if not metrics:
        return warnings

    n = len(metrics)

    b_dominant = sum(
        1 for m in metrics.values()
        if m["ab_ratio"] > 2.0 and m["ab_ratio"] != float("inf")
    )
    if b_dominant > n * 0.5:
        pct = b_dominant / n * 100
        warnings.append(
            f"B matrices dominating A across {pct:.0f}% of layers "
            f"(ab_ratio > 2.0) — possible overtraining or LR too high"
        )

    for name, m in metrics.items():
        if m["effective_rank"] < 1.5:
            short = name.split(".")[-1] if "." in name else name
            warnings.append(
                f"Effective rank collapsed to {m['effective_rank']:.1f} in "
                f"{short} — dead LoRA dimensions"
            )

    for name, m in metrics.items():
        if m["condition_number"] > 1000:
            short = name.split(".")[-1] if "." in name else name
            warnings.append(
                f"Condition number > 1000 ({m['condition_number']:.0f}) in "
                f"{short} — numerical instability risk"
            )

    return warnings
