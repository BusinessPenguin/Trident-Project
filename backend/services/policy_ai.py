"""Policy event classifier and decision narrative helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from openai import OpenAI


def _json_safe_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        raw = json.dumps(payload, default=str, separators=(",", ":"))
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def classify_policy_event(
    title: str,
    summary: Optional[str],
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify whether a headline is a US federal crypto policy event.
    Returns a JSON dict with fixed keys.
    """
    client = OpenAI()

    system_msg = (
        "You are a classifier for US federal crypto policy events. "
        "Return strict JSON only, with keys: "
        "is_policy_event (bool), us_federal (bool), crypto_related (bool), "
        "stage (introduced|hearing|markup|vote|passed|signed|proposal|unknown), "
        "branch (senate|house|congress|executive|agency|court|unknown), "
        "confidence (float 0-1), rationale (short string). "
        "Use only the provided title/summary/source. "
        "If unsure, set confidence <= 0.4 and use stage/branch 'unknown'."
    )

    payload = {
        "title": title or "",
        "summary": summary or "",
        "source": source or "",
    }

    resp = client.chat.completions.create(
        model="gpt-5.2",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(payload)},
        ],
    )
    content = resp.choices[0].message.content
    data = json.loads(content)
    return data if isinstance(data, dict) else {}


def render_decision_narrative_gpt52(
    payload: Dict[str, Any],
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    Render human-readable decision narrative fields from deterministic payload.
    Returns dict with keys: summary, what_would_change, top_risks.
    """
    try:
        client = OpenAI()
        system_msg = (
            "You explain a deterministic trading decision using ONLY provided facts. "
            "Do not invent numbers or thresholds. "
            "Return strict JSON with keys: summary, what_would_change, top_risks. "
            "summary must be 2-5 sentences in plain language. "
            "what_would_change must be 3-6 concise strings. "
            "top_risks must be 2-5 concise strings."
        )
        schema = {
            "name": "decision_narrative",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string", "maxLength": 1200},
                    "what_would_change": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 6,
                        "items": {"type": "string", "maxLength": 180},
                    },
                    "top_risks": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 5,
                        "items": {"type": "string", "maxLength": 180},
                    },
                },
                "required": ["summary", "what_would_change", "top_risks"],
            },
        }
        user_msg = json.dumps(payload, separators=(",", ":"))
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_schema", "json_schema": schema},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        except TypeError:
            # Fallback path for SDKs without json_schema support.
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )

        content = resp.choices[0].message.content
        data = json.loads(content) if content else {}
        if not isinstance(data, dict):
            return {}
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return {}
        changes = data.get("what_would_change")
        risks = data.get("top_risks")
        if not isinstance(changes, list) or not isinstance(risks, list):
            return {}
        return {
            "summary": summary.strip(),
            "what_would_change": [str(x).strip() for x in changes if str(x).strip()],
            "top_risks": [str(x).strip() for x in risks if str(x).strip()],
        }
    except Exception:
        # Fail-safe for network/auth/runtime issues; caller will use deterministic fallback narrative.
        return {}


def render_trade_explanation_gpt52(
    payload: Dict[str, Any],
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    Explain deterministic trade-plan values in plain language.
    Returns dict with keys: rationale, risk_context, positioning_logic.
    """
    try:
        client = OpenAI()
        system_msg = (
            "You explain a deterministic trade plan using ONLY provided facts. "
            "Do not invent numbers, thresholds, or calculations. "
            "If no_trade_gate.active is false, do not describe quality flags as blockers; describe them as caution only. "
            "When present, explicitly reference stop_source, atr_mode, anchor_age_hours, anchor_stale, and escalation recommendation. "
            "Return strict JSON with keys: rationale, risk_context, positioning_logic. "
            "Each field should be concise plain-English text."
        )
        schema = {
            "name": "trade_explanation",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rationale": {"type": "string", "maxLength": 900},
                    "risk_context": {"type": "string", "maxLength": 900},
                    "positioning_logic": {"type": "string", "maxLength": 900},
                },
                "required": ["rationale", "risk_context", "positioning_logic"],
            },
        }
        user_msg = json.dumps(payload, separators=(",", ":"))
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_schema", "json_schema": schema},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        except TypeError:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )

        content = resp.choices[0].message.content
        data = json.loads(content) if content else {}
        if not isinstance(data, dict):
            return {}
        rationale = data.get("rationale")
        risk_context = data.get("risk_context")
        positioning_logic = data.get("positioning_logic")
        if not isinstance(rationale, str) or not rationale.strip():
            return {}
        if not isinstance(risk_context, str) or not risk_context.strip():
            return {}
        if not isinstance(positioning_logic, str) or not positioning_logic.strip():
            return {}
        return {
            "rationale": rationale.strip(),
            "risk_context": risk_context.strip(),
            "positioning_logic": positioning_logic.strip(),
        }
    except Exception:
        return {}


def render_paper_report_summary_gpt52(
    payload: Dict[str, Any],
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """Generate a concise report summary from deterministic paper stats."""
    try:
        client = OpenAI()
        safe_payload = _json_safe_payload(payload)
        if not safe_payload:
            return {"error": "payload_not_json_serializable"}
        system_msg = (
            "You summarize deterministic paper-trading report stats in plain language. "
            "Do not invent numbers. Use only provided data. "
            "If prediction_summary or top_patterns are present, include them concisely. "
            "Return strict JSON with keys: summary."
        )
        schema = {
            "name": "paper_report_summary",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string", "maxLength": 1200},
                },
                "required": ["summary"],
            },
        }
        user_msg = json.dumps(safe_payload, separators=(",", ":"))
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_schema", "json_schema": schema},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        except TypeError:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        content = resp.choices[0].message.content
        data = json.loads(content) if content else {}
        if not isinstance(data, dict):
            return {"error": "invalid_response_object"}
        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return {"error": "missing_summary"}
        return {"summary": summary.strip()}
    except Exception as exc:
        return {"error": str(exc)}


def render_paper_learning_summary_gpt52(
    payload: Dict[str, Any],
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """Generate concise learning summary from deterministic learning payload."""
    try:
        client = OpenAI()
        system_msg = (
            "You summarize deterministic paper-trading learning output. "
            "Do not invent numbers or parameter values. "
            "If prediction/pattern diagnostics are present, reference them briefly. "
            "Return strict JSON with keys: summary, what_changed, key_risks."
        )
        schema = {
            "name": "paper_learning_summary",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string", "maxLength": 1200},
                    "what_changed": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 6,
                        "items": {"type": "string", "maxLength": 200},
                    },
                    "key_risks": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 6,
                        "items": {"type": "string", "maxLength": 200},
                    },
                },
                "required": ["summary", "what_changed", "key_risks"],
            },
        }
        safe_payload = _json_safe_payload(payload)
        if not safe_payload:
            return {}
        user_msg = json.dumps(safe_payload, separators=(",", ":"))
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_schema", "json_schema": schema},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        except TypeError:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        content = resp.choices[0].message.content
        data = json.loads(content) if content else {}
        if not isinstance(data, dict):
            return {}
        summary = data.get("summary")
        what_changed = data.get("what_changed")
        key_risks = data.get("key_risks")
        if not isinstance(summary, str) or not summary.strip():
            return {}
        if not isinstance(what_changed, list) or not isinstance(key_risks, list):
            return {}
        return {
            "summary": summary.strip(),
            "what_changed": [str(x).strip() for x in what_changed if str(x).strip()],
            "key_risks": [str(x).strip() for x in key_risks if str(x).strip()],
        }
    except Exception:
        return {}


def render_paper_run_explanation_gpt52(
    payload: Dict[str, Any],
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """Explain what happened in a paper:run cycle and current trade state."""
    try:
        client = OpenAI()
        system_msg = (
            "You explain a deterministic paper-trading run. "
            "Do not invent numbers or facts. Use only provided payload fields. "
            "Return strict JSON with keys: summary, current_trade_status."
        )
        schema = {
            "name": "paper_run_explanation",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string", "maxLength": 1200},
                    "current_trade_status": {"type": "string", "maxLength": 1200},
                },
                "required": ["summary", "current_trade_status"],
            },
        }
        safe_payload = _json_safe_payload(payload)
        if not safe_payload:
            return {}
        user_msg = json.dumps(safe_payload, separators=(",", ":"))
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_schema", "json_schema": schema},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        except TypeError:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )

        content = resp.choices[0].message.content
        data = json.loads(content) if content else {}
        if not isinstance(data, dict):
            return {}
        summary = data.get("summary")
        current_trade_status = data.get("current_trade_status")
        if not isinstance(summary, str) or not summary.strip():
            return {}
        if not isinstance(current_trade_status, str) or not current_trade_status.strip():
            return {}
        return {
            "summary": summary.strip(),
            "current_trade_status": current_trade_status.strip(),
        }
    except Exception:
        return {}


def render_aggression_profile_gpt52(
    payload: Dict[str, Any],
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    Select an aggression profile for paper trading from deterministic context.
    Returns dict with keys:
      tier, override_weighted_gate, quick_exit_bias, reasoning_summary, confidence
    """
    try:
        client = OpenAI()
        safe_payload = _json_safe_payload(payload)
        if not safe_payload:
            return {}
        system_msg = (
            "You choose an aggression profile for paper trading using ONLY provided facts. "
            "Do not make raw trade decisions and do not invent numbers. "
            "Output strict JSON with keys: tier, override_weighted_gate, quick_exit_bias, reasoning_summary, confidence. "
            "tier must be one of: very_defensive, defensive, balanced, assertive, very_aggressive. "
            "quick_exit_bias must be one of: none, take_profit, cut_loss."
        )
        schema = {
            "name": "aggression_profile",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tier": {
                        "type": "string",
                        "enum": ["very_defensive", "defensive", "balanced", "assertive", "very_aggressive"],
                    },
                    "override_weighted_gate": {"type": "boolean"},
                    "quick_exit_bias": {
                        "type": "string",
                        "enum": ["none", "take_profit", "cut_loss"],
                    },
                    "reasoning_summary": {"type": "string", "maxLength": 800},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "tier",
                    "override_weighted_gate",
                    "quick_exit_bias",
                    "reasoning_summary",
                    "confidence",
                ],
            },
        }
        user_msg = json.dumps(safe_payload, separators=(",", ":"))
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_schema", "json_schema": schema},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        except TypeError:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        content = resp.choices[0].message.content
        data = json.loads(content) if content else {}
        if not isinstance(data, dict):
            return {}
        tier = data.get("tier")
        quick = data.get("quick_exit_bias")
        reasoning = data.get("reasoning_summary")
        conf = data.get("confidence")
        if tier not in {"very_defensive", "defensive", "balanced", "assertive", "very_aggressive"}:
            return {}
        if quick not in {"none", "take_profit", "cut_loss"}:
            return {}
        if not isinstance(reasoning, str) or not reasoning.strip():
            return {}
        try:
            conf_f = float(conf)
        except Exception:
            return {}
        conf_f = max(0.0, min(1.0, conf_f))
        return {
            "tier": tier,
            "override_weighted_gate": bool(data.get("override_weighted_gate")),
            "quick_exit_bias": quick,
            "reasoning_summary": reasoning.strip(),
            "confidence": round(conf_f, 6),
        }
    except Exception:
        return {}


def render_paper_learning_policy_proposal_gpt52(
    payload: Dict[str, Any],
    model: str = "gpt-5.2",
) -> Dict[str, Any]:
    """
    Propose bounded policy deltas for paper learning.
    Numeric application is still deterministic downstream.
    """
    try:
        client = OpenAI()
        safe_payload = _json_safe_payload(payload)
        if not safe_payload:
            return {}
        system_msg = (
            "You propose conservative parameter deltas for paper-trading learning from provided outcomes. "
            "Do not invent facts. Return strict JSON only using the specified schema and bounds."
        )
        schema = {
            "name": "paper_learning_policy_proposal",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "deltas": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "min_confidence": {"type": "number", "minimum": -0.02, "maximum": 0.02},
                            "stop_distance_atr_mult_pct": {"type": "number", "minimum": -0.05, "maximum": 0.05},
                            "penalty_high_vol_chop_pct": {"type": "number", "minimum": -0.05, "maximum": 0.05},
                            "penalty_elevated_event_risk_pct": {
                                "type": "number",
                                "minimum": -0.05,
                                "maximum": 0.05,
                            },
                            "entry_min_score": {"type": "number", "minimum": -0.02, "maximum": 0.02},
                            "entry_min_effective_confidence": {"type": "number", "minimum": -0.02, "maximum": 0.02},
                            "entry_min_agreement": {"type": "number", "minimum": -0.02, "maximum": 0.02},
                            "entry_min_margin": {"type": "number", "minimum": -0.02, "maximum": 0.02},
                            "aggression_risk_mult_pct": {"type": "number", "minimum": -0.05, "maximum": 0.05},
                            "aggression_stop_mult_pct": {"type": "number", "minimum": -0.05, "maximum": 0.05},
                            "aggression_hold_mult_pct": {"type": "number", "minimum": -0.05, "maximum": 0.05},
                            "aggression_exposure_cap_pct": {"type": "number", "minimum": -0.05, "maximum": 0.05},
                        },
                        "required": [
                            "min_confidence",
                            "stop_distance_atr_mult_pct",
                            "penalty_high_vol_chop_pct",
                            "penalty_elevated_event_risk_pct",
                            "entry_min_score",
                            "entry_min_effective_confidence",
                            "entry_min_agreement",
                            "entry_min_margin",
                            "aggression_risk_mult_pct",
                            "aggression_stop_mult_pct",
                            "aggression_hold_mult_pct",
                            "aggression_exposure_cap_pct",
                        ],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string", "maxLength": 1200},
                    "focus_labels": {
                        "type": "array",
                        "maxItems": 6,
                        "items": {
                            "type": "string",
                            "enum": [
                                "STOP_TOO_TIGHT",
                                "STOP_TOO_WIDE",
                                "REGIME_MISCLASSIFIED",
                                "EVENT_RISK_UNDERWEIGHTED",
                                "SLIPPAGE_TOO_OPTIMISTIC",
                                "CONFIDENCE_CALIBRATION",
                                "SIGNAL_DISAGREEMENT",
                                "TIMING_BAD",
                                "NEWS_REVERSAL",
                            ],
                        },
                    },
                },
                "required": ["deltas", "confidence", "rationale", "focus_labels"],
            },
        }
        user_msg = json.dumps(safe_payload, separators=(",", ":"))
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_schema", "json_schema": schema},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        except TypeError:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                top_p=1.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
        content = resp.choices[0].message.content
        data = json.loads(content) if content else {}
        if not isinstance(data, dict):
            return {}
        deltas = data.get("deltas")
        if not isinstance(deltas, dict):
            return {}
        confidence = data.get("confidence")
        rationale = data.get("rationale")
        focus_labels = data.get("focus_labels")
        if not isinstance(rationale, str) or not rationale.strip():
            return {}
        if not isinstance(focus_labels, list):
            return {}
        try:
            confidence_f = float(confidence)
        except Exception:
            return {}
        confidence_f = max(0.0, min(1.0, confidence_f))
        return {
            "deltas": deltas,
            "confidence": round(confidence_f, 6),
            "rationale": rationale.strip(),
            "focus_labels": [str(x) for x in focus_labels if str(x)],
        }
    except Exception:
        return {}
