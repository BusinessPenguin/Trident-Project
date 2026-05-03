from __future__ import annotations

from typing import Any, Dict, List, Optional


ALL_CHECK_CODES = [
    "CRITICAL_STALE_REQUIRED_MODALITY",
    "CRITICAL_LOW_CONFIDENCE",
    "CRITICAL_RISK_CLUSTER",
    "LOW_CONFIDENCE",
    "LOW_AGREEMENT",
    "LOW_HORIZON_ALIGNMENT",
    "MODEL_EDGE_WEAK",
    "HIGH_VOL_CHOP_LOW_CONF",
    "ELEVATED_EVENT_RISK_LOW_CONF",
    "RISK_CLUSTER_LOW_CONF",
    "LOW_BREADTH",
    "RELATIVE_WEAKNESS",
    "PRE_BREAKOUT_COMPRESSION",
]


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _normalize_code_list(raw: Any, fallback: List[str]) -> List[str]:
    if not isinstance(raw, list):
        return list(fallback)
    out: List[str] = []
    seen = set()
    valid = set(ALL_CHECK_CODES)
    for item in raw:
        code = str(item or "").strip().upper()
        if not code or code in seen or code not in valid:
            continue
        seen.add(code)
        out.append(code)
    return out if out else list(fallback)


def default_gate_policy() -> Dict[str, Any]:
    """
    Simplified baseline:
    - immutable hard blockers kept minimal
    - weighted blockers focused on confidence/edge + one horizon path
    - breadth/RS/compression become diagnostics (non-gating)
    - overlapping chop/event low-confidence checks merged into one path
    """
    return {
        "policy_version": "smart_adjust_v1_baseline",
        "enabled_hard_checks": [
            "CRITICAL_STALE_REQUIRED_MODALITY",
            "CRITICAL_LOW_CONFIDENCE",
        ],
        "enabled_weighted_checks": [
            "LOW_CONFIDENCE",
            "LOW_HORIZON_ALIGNMENT",
            "MODEL_EDGE_WEAK",
            "RISK_CLUSTER_LOW_CONF",
        ],
        "diagnostic_only_checks": [
            "LOW_AGREEMENT",
            "LOW_BREADTH",
            "RELATIVE_WEAKNESS",
            "PRE_BREAKOUT_COMPRESSION",
        ],
        "merged_overlap_mode": True,
    }


def _parse_gate_policy(overrides: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    policy = default_gate_policy()

    for source in [
        (context or {}).get("gate_policy"),
        (overrides or {}).get("gate_policy"),
        overrides,
    ]:
        if not isinstance(source, dict):
            continue
        if source.get("policy_version") is not None:
            policy["policy_version"] = str(source.get("policy_version") or policy["policy_version"])
        if source.get("enabled_hard_checks") is not None:
            policy["enabled_hard_checks"] = _normalize_code_list(
                source.get("enabled_hard_checks"),
                policy["enabled_hard_checks"],
            )
        if source.get("enabled_weighted_checks") is not None:
            policy["enabled_weighted_checks"] = _normalize_code_list(
                source.get("enabled_weighted_checks"),
                policy["enabled_weighted_checks"],
            )
        if source.get("diagnostic_only_checks") is not None:
            policy["diagnostic_only_checks"] = _normalize_code_list(
                source.get("diagnostic_only_checks"),
                policy["diagnostic_only_checks"],
            )
        if source.get("merged_overlap_mode") is not None:
            policy["merged_overlap_mode"] = bool(source.get("merged_overlap_mode"))

    # Prevent overlap between weighted and diagnostic sets.
    enabled_weighted = [c for c in policy["enabled_weighted_checks"] if c not in set(policy["diagnostic_only_checks"])]
    policy["enabled_weighted_checks"] = enabled_weighted
    return policy


def compute_gate_thresholds(regime_label: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    _ = regime_label
    thresholds = {
        "min_confidence": 0.33,
        "min_agreement": 0.52,
        "min_horizon_alignment": 0.60,
        "min_margin": 0.06,
        "min_confidence_high_vol_chop": 0.45,
        "min_confidence_elevated_event_risk": 0.40,
        "min_breadth_score": 0.45,
        "min_rs_vs_btc_7d": -0.01,
        "compression_vwap_abs_max": 0.01,
        "max_alignment_quality_penalty": 0.10,
        "alignment_penalty_per_point": 0.25,
        "critical_min_confidence": 0.25,
        "critical_risk_cluster_confidence": 0.30,
        "w_conf": 0.30,
        "w_agree": 0.18,
        "w_align": 0.12,
        "w_margin": 0.22,
        "w_chop": 0.12,
        "w_event": 0.06,
        "w_breadth": 0.08,
        "w_rs": 0.08,
        "w_compression": 0.04,
        "penalty_threshold": 0.45,
    }
    if overrides:
        for key, value in overrides.items():
            if key not in thresholds:
                continue
            parsed = _safe_float(value)
            if parsed is None:
                continue
            thresholds[key] = float(parsed)
    return thresholds


def evaluate_hard_blockers(
    effective_confidence: Optional[float],
    event_risk: str,
    regime_label: str,
    thresholds: Dict[str, float],
    context: Optional[Dict[str, Any]] = None,
    gate_policy: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    blockers: List[Dict[str, str]] = []
    conf = _safe_float(effective_confidence)
    risk = str(event_risk or "").lower()
    regime = str(regime_label or "")
    ctx = context or {}
    enabled_hard = set((gate_policy or {}).get("enabled_hard_checks") or [])

    required_ok = ctx.get("required_modality_ok")
    required_failures = ctx.get("required_modality_failures") or []
    if (
        "CRITICAL_STALE_REQUIRED_MODALITY" in enabled_hard
        and (required_ok is False or required_failures)
    ):
        detail = str(required_failures[0]) if required_failures else "required modality freshness failed"
        blockers.append(
            {
                "code": "CRITICAL_STALE_REQUIRED_MODALITY",
                "detail": detail,
            }
        )
    if conf is None:
        return blockers

    if "CRITICAL_LOW_CONFIDENCE" in enabled_hard and conf < thresholds["critical_min_confidence"]:
        blockers.append(
            {
                "code": "CRITICAL_LOW_CONFIDENCE",
                "detail": f"effective_confidence={conf:.3f} < {thresholds['critical_min_confidence']:.2f}",
            }
        )

    if (
        "CRITICAL_RISK_CLUSTER" in enabled_hard
        and regime == "high_vol_chop"
        and risk == "elevated"
        and conf < thresholds["critical_risk_cluster_confidence"]
    ):
        blockers.append(
            {
                "code": "CRITICAL_RISK_CLUSTER",
                "detail": (
                    "regime=high_vol_chop and event_risk=elevated and "
                    f"effective_confidence={conf:.3f} < {thresholds['critical_risk_cluster_confidence']:.2f}"
                ),
            }
        )
    return blockers


def _evaluate_reason_buckets(
    effective_confidence: Optional[float],
    agreement_score: Optional[float],
    horizon_alignment_score: Optional[float],
    margin_vs_second: Optional[float],
    event_risk: str,
    regime_label: str,
    thresholds: Dict[str, float],
    context: Optional[Dict[str, Any]] = None,
    gate_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    conf = _safe_float(effective_confidence)
    agree = _safe_float(agreement_score)
    align = _safe_float(horizon_alignment_score)
    margin = _safe_float(margin_vs_second)
    risk = str(event_risk or "").lower()
    regime = str(regime_label or "")

    t_conf = thresholds["min_confidence"]
    t_agree = thresholds["min_agreement"]
    t_align = thresholds["min_horizon_alignment"]
    t_margin = thresholds["min_margin"]
    t_chop = thresholds["min_confidence_high_vol_chop"]
    t_event = thresholds["min_confidence_elevated_event_risk"]
    t_breadth = thresholds["min_breadth_score"]
    t_rs = thresholds["min_rs_vs_btc_7d"]
    t_vwap_abs = thresholds["compression_vwap_abs_max"]

    ctx = context or {}
    breadth_score = _safe_float(ctx.get("breadth_score"))
    rs_vs_btc_7d = _safe_float(ctx.get("rs_vs_btc_7d"))
    vwap_distance_pct = _safe_float(ctx.get("vwap_distance_pct"))
    squeeze_on = bool(ctx.get("squeeze_on"))

    c_conf = 0.0
    if conf is not None and conf < t_conf:
        c_conf = clamp01((t_conf - conf) / t_conf)
    c_agree = 0.0
    if agree is not None and agree < t_agree:
        c_agree = clamp01((t_agree - agree) / t_agree)
    c_align = 0.0
    if align is not None and align < t_align:
        c_align = clamp01((t_align - align) / t_align)
    c_margin = 0.0
    if margin is not None and margin < t_margin:
        c_margin = clamp01((t_margin - margin) / t_margin)
    c_chop = 0.0
    if regime == "high_vol_chop" and conf is not None and conf < t_chop:
        c_chop = clamp01((t_chop - conf) / t_chop)
    c_event = 0.0
    if risk == "elevated" and conf is not None and conf < t_event:
        c_event = clamp01((t_event - conf) / t_event)
    c_breadth = 0.0
    if breadth_score is not None and breadth_score < t_breadth:
        c_breadth = clamp01((t_breadth - breadth_score) / max(t_breadth, 1e-9))
    c_rs = 0.0
    if rs_vs_btc_7d is not None and rs_vs_btc_7d < t_rs:
        denom = max(abs(t_rs), 0.01)
        c_rs = clamp01((t_rs - rs_vs_btc_7d) / denom)
    c_compression = 0.0
    if squeeze_on and vwap_distance_pct is not None and abs(vwap_distance_pct) < t_vwap_abs:
        c_compression = clamp01((t_vwap_abs - abs(vwap_distance_pct)) / max(t_vwap_abs, 1e-9))

    reason_components: Dict[str, Dict[str, Any]] = {
        "LOW_CONFIDENCE": {
            "weight": thresholds["w_conf"],
            "component_score": c_conf,
            "detail": (
                f"effective_confidence={conf:.3f} < {t_conf:.2f}"
                if conf is not None and c_conf > 0
                else "effective_confidence at or above threshold"
            ),
        },
        "LOW_AGREEMENT": {
            "weight": thresholds["w_agree"],
            "component_score": c_agree,
            "detail": (
                f"agreement_score={agree:.3f} < {t_agree:.2f}"
                if agree is not None and c_agree > 0
                else "agreement_score at or above threshold"
            ),
        },
        "LOW_HORIZON_ALIGNMENT": {
            "weight": thresholds["w_align"],
            "component_score": c_align,
            "detail": (
                f"horizon_alignment_score={align:.3f} < {t_align:.2f}"
                if align is not None and c_align > 0
                else "horizon_alignment_score at or above threshold"
            ),
        },
        "MODEL_EDGE_WEAK": {
            "weight": thresholds["w_margin"],
            "component_score": c_margin,
            "detail": (
                f"margin={margin:.3f} < {t_margin:.2f}"
                if margin is not None and c_margin > 0
                else "margin at or above threshold"
            ),
        },
        "HIGH_VOL_CHOP_LOW_CONF": {
            "weight": thresholds["w_chop"],
            "component_score": c_chop,
            "detail": (
                f"regime=high_vol_chop and effective_confidence={conf:.3f} < {t_chop:.2f}"
                if conf is not None and c_chop > 0
                else "high_vol_chop confidence check passed or not applicable"
            ),
        },
        "ELEVATED_EVENT_RISK_LOW_CONF": {
            "weight": thresholds["w_event"],
            "component_score": c_event,
            "detail": (
                f"event_risk=elevated and effective_confidence={conf:.3f} < {t_event:.2f}"
                if conf is not None and c_event > 0
                else "elevated event-risk confidence check passed or not applicable"
            ),
        },
        "LOW_BREADTH": {
            "weight": thresholds["w_breadth"],
            "component_score": c_breadth,
            "detail": (
                f"breadth_score={breadth_score:.3f} < {t_breadth:.2f}"
                if breadth_score is not None and c_breadth > 0
                else "breadth check passed or unavailable"
            ),
        },
        "RELATIVE_WEAKNESS": {
            "weight": thresholds["w_rs"],
            "component_score": c_rs,
            "detail": (
                f"rs_vs_btc_7d={rs_vs_btc_7d:.4f} < {t_rs:.4f}"
                if rs_vs_btc_7d is not None and c_rs > 0
                else "relative-strength check passed or unavailable"
            ),
        },
        "PRE_BREAKOUT_COMPRESSION": {
            "weight": thresholds["w_compression"],
            "component_score": c_compression,
            "detail": (
                f"squeeze_on={squeeze_on} and |vwap_distance_pct|={abs(vwap_distance_pct):.4f} < {t_vwap_abs:.4f}"
                if vwap_distance_pct is not None and c_compression > 0
                else "compression check passed or not applicable"
            ),
        },
    }

    policy = gate_policy or default_gate_policy()
    if bool(policy.get("merged_overlap_mode", True)):
        overlap = max(c_chop, c_event)
        overlap_weight = max(float(thresholds["w_chop"]), float(thresholds["w_event"]))
        overlap_detail = (
            "regime/event overlap confidence stress "
            f"(chop_component={c_chop:.3f}, event_component={c_event:.3f})"
        )
        reason_components.pop("HIGH_VOL_CHOP_LOW_CONF", None)
        reason_components.pop("ELEVATED_EVENT_RISK_LOW_CONF", None)
        reason_components["RISK_CLUSTER_LOW_CONF"] = {
            "weight": overlap_weight,
            "component_score": overlap,
            "detail": overlap_detail,
        }

    enabled_weighted = set(policy.get("enabled_weighted_checks") or [])
    diagnostic_only = set(policy.get("diagnostic_only_checks") or [])

    weighted_reasons: List[Dict[str, Any]] = []
    diagnostic_reasons: List[Dict[str, Any]] = []
    disabled_checks: List[str] = []

    for code, item in reason_components.items():
        component = float(item.get("component_score") or 0.0)
        if component <= 0.0:
            continue
        weight = float(item.get("weight") or 0.0)
        implied_penalty = clamp01(weight * component)
        payload = {
            "code": code,
            "weight": round(weight, 3),
            "component_score": round(component, 3),
            "penalty": round(implied_penalty, 3),
            "detail": str(item.get("detail") or ""),
        }
        if code in enabled_weighted:
            weighted_reasons.append(payload)
        elif code in diagnostic_only:
            diagnostic_reasons.append(payload)
        else:
            disabled_checks.append(code)

    weighted_reasons.sort(key=lambda x: float(x.get("penalty") or 0.0), reverse=True)
    diagnostic_reasons.sort(key=lambda x: float(x.get("penalty") or 0.0), reverse=True)
    disabled_checks = sorted(set(disabled_checks))

    return {
        "weighted_reasons": weighted_reasons,
        "diagnostic_reasons": diagnostic_reasons,
        "disabled_checks": disabled_checks,
    }


def evaluate_weighted_reasons(
    effective_confidence: Optional[float],
    agreement_score: Optional[float],
    horizon_alignment_score: Optional[float],
    margin_vs_second: Optional[float],
    event_risk: str,
    regime_label: str,
    thresholds: Dict[str, float],
    context: Optional[Dict[str, Any]] = None,
    gate_policy: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return _evaluate_reason_buckets(
        effective_confidence=effective_confidence,
        agreement_score=agreement_score,
        horizon_alignment_score=horizon_alignment_score,
        margin_vs_second=margin_vs_second,
        event_risk=event_risk,
        regime_label=regime_label,
        thresholds=thresholds,
        context=context,
        gate_policy=gate_policy,
    )["weighted_reasons"]


def build_gate_result(
    raw_confidence: Optional[float],
    agreement_score: Optional[float],
    horizon_alignment_score: Optional[float],
    margin_vs_second: Optional[float],
    event_risk: str,
    regime_label: str,
    is_high_vol: bool,
    context: Optional[Dict[str, Any]] = None,
    thresholds_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    thresholds = compute_gate_thresholds(regime_label, overrides=thresholds_override)
    gate_policy = _parse_gate_policy(overrides=thresholds_override, context=context)

    conf = _safe_float(raw_confidence)
    align = _safe_float(horizon_alignment_score)
    _ = is_high_vol

    alignment_quality_penalty = 0.0
    if align is not None and align < thresholds["min_horizon_alignment"]:
        deficit = thresholds["min_horizon_alignment"] - align
        alignment_quality_penalty = min(
            thresholds["max_alignment_quality_penalty"],
            max(0.0, deficit * thresholds["alignment_penalty_per_point"]),
        )
    effective_conf = None if conf is None else max(0.0, conf - alignment_quality_penalty)

    quality_adjustments: List[Dict[str, str]] = []
    if align is not None and alignment_quality_penalty > 0.0:
        quality_adjustments.append(
            {
                "code": "LOW_HORIZON_ALIGNMENT_QUALITY",
                "detail": (
                    f"horizon_alignment_score={align:.3f} < {thresholds['min_horizon_alignment']:.2f}; "
                    f"confidence penalty={alignment_quality_penalty:.3f}"
                ),
            }
        )

    hard_blockers = evaluate_hard_blockers(
        effective_confidence=effective_conf,
        event_risk=event_risk,
        regime_label=regime_label,
        thresholds=thresholds,
        context=context,
        gate_policy=gate_policy,
    )
    buckets = _evaluate_reason_buckets(
        effective_confidence=effective_conf,
        agreement_score=agreement_score,
        horizon_alignment_score=horizon_alignment_score,
        margin_vs_second=margin_vs_second,
        event_risk=event_risk,
        regime_label=regime_label,
        thresholds=thresholds,
        context=context,
        gate_policy=gate_policy,
    )
    weighted_reasons = buckets["weighted_reasons"]
    diagnostic_reasons = buckets["diagnostic_reasons"]
    disabled_checks = buckets["disabled_checks"]

    penalty_total = float(sum(float(r.get("penalty") or 0.0) for r in weighted_reasons))
    penalty_threshold = float(thresholds["penalty_threshold"])
    has_hard = bool(hard_blockers)
    active = bool(has_hard or penalty_total >= penalty_threshold)
    activation_basis = "hard_blocker" if has_hard else "penalty_threshold" if active else "none"

    reasons: List[Dict[str, str]] = []
    reasons.extend(hard_blockers)
    reasons.extend({"code": r["code"], "detail": r["detail"]} for r in weighted_reasons)

    active_blocking_checks = [str(r.get("code") or "") for r in reasons if r.get("code")]
    triggered_any = set(active_blocking_checks)
    triggered_any.update(str(r.get("code") or "") for r in diagnostic_reasons if r.get("code"))

    passed_checks = [
        code
        for code in ALL_CHECK_CODES
        if code not in triggered_any and code not in set(disabled_checks)
    ]

    return {
        "gate_model": "hybrid_weighted_v1",
        "gate_policy_version": str(gate_policy.get("policy_version") or "smart_adjust_v1_baseline"),
        "active": active,
        "activation_basis": activation_basis,
        "reasons": reasons,
        "hard_blockers": hard_blockers,
        "weighted_reasons": weighted_reasons,
        "penalty_total": round(penalty_total, 3),
        "penalty_threshold": round(penalty_threshold, 3),
        "penalty_ratio": round((penalty_total / penalty_threshold) if penalty_threshold > 0 else 0.0, 3),
        "passed_checks": passed_checks,
        "quality_adjustments": quality_adjustments,
        "effective_confidence": round(effective_conf, 3) if effective_conf is not None else None,
        "thresholds": thresholds,
        "active_blocking_checks": active_blocking_checks,
        "diagnostic_only_checks": diagnostic_reasons,
        "disabled_checks": disabled_checks,
        "enabled_hard_checks": list(gate_policy.get("enabled_hard_checks") or []),
        "enabled_weighted_checks": list(gate_policy.get("enabled_weighted_checks") or []),
        "merged_overlap_mode": bool(gate_policy.get("merged_overlap_mode", True)),
    }
