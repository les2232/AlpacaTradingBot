"""Model explanation, output validation, and the no-model fallback.

The model receives only the evidence package for the selected question.  Its
output must follow a strict JSON schema; every finding must cite evidence IDs
that exist in that same package.  When the package says the comparison is
unavailable, sentences that assert comparative conclusions are removed.

Without an API key (or when the call fails) the operator still gets a
deterministic "calculated diagnostics" answer built from the same package.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from tradeos.ask.evidence import (
    QUESTION_COMPARE,
    QUESTION_EXPLAIN,
    QUESTION_SYMBOL,
    EvidencePackage,
)
from tradeos.ask.settings import AskSettings

MAX_OUTPUT_TOKENS = 4000
MAX_FINDINGS = 20
MAX_ANSWER_CHARS = 4000
MAX_STATEMENT_CHARS = 1200
SCHEMA_NAME = "tradeos_session_answer"

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "findings", "missing_evidence", "comparison_status"],
    "properties": {
        "answer": {"type": "string", "description": "Direct plain-English answer for the operator."},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement", "kind", "citations"],
                "properties": {
                    "statement": {"type": "string"},
                    "kind": {"type": "string", "enum": ["observation", "hypothesis"]},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "comparison_status": {"type": "string", "enum": ["available", "unavailable", "not_applicable"]},
    },
}

INSTRUCTIONS = """You are Ask TradeOS, a read-only analyst explaining one historical trading session to its operator.

Rules:
1. Use only the evidence package in the user message. Do not use outside knowledge about markets, symbols, or the strategy beyond what the package states.
2. Every number you state must appear in the package. Do not compute new statistics; the package already contains all calculated values.
3. Every finding must cite one or more evidence IDs from the package "evidence" list (for example "M1", "E4", "CFG1"). Cite only IDs that exist.
4. Label each finding "observation" (directly shown by the cited records) or "hypothesis" (a plausible explanation the records do not prove). Phrase hypotheses tentatively.
5. If comparison.status is "unavailable", say "Comparison unavailable" and give the listed reasons. Do not state or imply underperformance, outperformance, drift, improvement, or deterioration relative to any backtest or expectation. Observed session metrics may still be reported.
6. Set comparison_status to exactly the package's comparison.status for the comparison question; for other questions use "not_applicable" unless you discuss the comparison, in which case copy the package value.
7. List in missing_evidence what cannot be established from the records (use the package limitations and data_coverage). Do not invent missing data.
8. Respect the scope: an "unbound_log_day" is not a single session and has no known configuration. Never assume the current configuration applied to a historical session.
9. This tool is read-only. Do not recommend placing orders, changing trading rules or risk limits, repairing state, or promoting configurations.
10. Be concise and specific."""

_ID_PATTERN = re.compile(r"\[([A-Z]{1,3}\d{1,4})\]")
_COMPARATIVE = re.compile(
    r"(under-?perform|out-?perform|\bdrift|improv|deteriorat|regress|worse than|better than|"
    r"(above|below|short of|ahead of|exceed\w*|beat|lag\w*|trail\w*)\s+(the\s+)?(backtest|baseline|expectation|expected)|"
    r"(relative|compared)\s+(to|with)\s+(the\s+)?(backtest|baseline))",
    re.IGNORECASE,
)
_UNAVAILABLE_MARKERS = re.compile(
    r"(unavailable|cannot|can't|can not|not possible|not established|no valid|insufficient|unable|"
    r"not be (determined|assessed|made|established)|invalid|suppressed|withheld|not available|no basis)",
    re.IGNORECASE,
)
_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")
_FORBIDDEN_KEYS = ("account_fingerprint", "expected_account_fingerprint", "account_id", "order_id", "client_order_id", "api_key")


class ResponsesClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Finding:
    statement: str
    kind: str
    citations: tuple[str, ...]


@dataclass
class AnalystAnswer:
    mode: str  # "model" | "calculated"
    answer: str
    findings: list[Finding]
    missing_evidence: list[str]
    comparison_status: str
    warnings: list[str] = field(default_factory=list)
    model: str | None = None
    response_id: str | None = None

    @property
    def cited_ids(self) -> list[str]:
        seen: list[str] = []
        for finding in self.findings:
            for cid in finding.citations:
                if cid not in seen:
                    seen.append(cid)
        for cid in _ID_PATTERN.findall(self.answer):
            if cid not in seen:
                seen.append(cid)
        return seen


@dataclass
class AskResult:
    answer: AnalystAnswer
    error: str | None
    request_body: dict[str, Any] | None  # exactly what was (or would be) sent, never the key


class ModelOutputError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Request construction and data-minimisation guard
# ---------------------------------------------------------------------------

def build_request(package: EvidencePackage, settings: AskSettings) -> dict[str, Any]:
    payload = package.to_model_payload()
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert_no_secrets(text, settings)
    return {
        "model": settings.model,
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
        "text": {"format": {"type": "json_schema", "name": SCHEMA_NAME, "schema": ANSWER_SCHEMA, "strict": True}},
        "store": False,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def assert_no_secrets(text: str, settings: AskSettings | None = None) -> None:
    # Exact-key check for realistic key lengths (very short test keys would match ordinary text).
    if settings is not None and settings.api_key and len(settings.api_key) >= 8 and settings.api_key in text:
        raise RuntimeError("Refusing to send: the request body contains the API key.")
    if _SECRET_PATTERN.search(text):
        raise RuntimeError("Refusing to send: the request body contains a credential-like value.")
    for key in _FORBIDDEN_KEYS:
        if f'"{key}"' in text:
            raise RuntimeError(f"Refusing to send: the request body contains the field {key!r}.")


def _sanitize(message: str, settings: AskSettings | None) -> str:
    text = _SECRET_PATTERN.sub("[redacted]", message)
    if settings is not None and settings.api_key and len(settings.api_key) >= 8:
        text = text.replace(settings.api_key, "[redacted]")
    return text[:300]


def _error_message(exc: Exception, settings: AskSettings) -> str:
    try:
        import openai
    except ImportError:  # pragma: no cover
        openai = None  # type: ignore[assignment]
    status = getattr(exc, "status_code", None)
    if openai is not None:
        if isinstance(exc, openai.AuthenticationError):
            return "OpenAI rejected the API key (authentication failed). Check OPENAI_API_KEY."
        if isinstance(exc, openai.PermissionDeniedError):
            return f"This API key is not permitted to use model {settings.model!r}."
        if isinstance(exc, openai.NotFoundError):
            return f"Model {settings.model!r} was not found for this API key. Set TRADEOS_ASK_MODEL to an available model."
        if isinstance(exc, openai.RateLimitError):
            return "OpenAI rate limit or quota reached. Try again later."
        if isinstance(exc, openai.APITimeoutError):
            return f"The model request timed out after {settings.timeout_seconds:.0f}s."
        if isinstance(exc, openai.APIConnectionError):
            return "Could not reach the OpenAI API (network error)."
        if isinstance(exc, openai.BadRequestError):
            return "OpenAI rejected the request as invalid (HTTP 400)."
        if isinstance(exc, openai.APIStatusError):
            return f"OpenAI returned HTTP {status}."
    return f"Model request failed ({type(exc).__name__})."


def default_client(settings: AskSettings) -> ResponsesClient:
    from openai import OpenAI

    return OpenAI(api_key=settings.api_key, timeout=settings.timeout_seconds, max_retries=1).responses


# ---------------------------------------------------------------------------
# Output parsing and validation
# ---------------------------------------------------------------------------

def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def extract_output_text(response: Any) -> str:
    status = _get(response, "status")
    if status not in (None, "completed"):
        details = _get(response, "incomplete_details")
        reason = _get(details, "reason") if details is not None else None
        error = _get(response, "error")
        if error is not None:
            raise ModelOutputError(f"Model response failed ({_get(error, 'code') or 'error'}).")
        raise ModelOutputError(f"Model response incomplete ({reason or status}).")
    for item in _get(response, "output") or []:
        if _get(item, "type") != "message":
            continue
        for content in _get(item, "content") or []:
            if _get(content, "type") == "refusal":
                raise ModelOutputError("The model declined to answer: " + str(_get(content, "refusal") or "")[:200])
    text = _get(response, "output_text")
    if not isinstance(text, str):
        texts = []
        for item in _get(response, "output") or []:
            if _get(item, "type") == "message":
                for content in _get(item, "content") or []:
                    if _get(content, "type") == "output_text" and isinstance(_get(content, "text"), str):
                        texts.append(_get(content, "text"))
        text = "".join(texts)
    if not text.strip():
        raise ModelOutputError("Model returned no text output.")
    return text


def _strip_comparative_sentences(text: str) -> tuple[str, int]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept, removed = [], 0
    for sentence in sentences:
        if _COMPARATIVE.search(sentence) and not _UNAVAILABLE_MARKERS.search(sentence):
            removed += 1
            continue
        kept.append(sentence)
    return " ".join(kept).strip(), removed


def validate_model_output(raw_text: str, package: EvidencePackage) -> AnalystAnswer:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"Model output is not valid JSON ({exc.msg}).") from None
    if not isinstance(data, dict):
        raise ModelOutputError("Model output is not a JSON object.")
    expected_keys = set(ANSWER_SCHEMA["required"])
    if set(data) != expected_keys:
        raise ModelOutputError(
            "Model output fields do not match the schema "
            f"(missing={sorted(expected_keys - set(data))}, unexpected={sorted(set(data) - expected_keys)})."
        )
    answer = data["answer"]
    findings_raw = data["findings"]
    missing = data["missing_evidence"]
    status = data["comparison_status"]
    if not isinstance(answer, str) or not answer.strip():
        raise ModelOutputError("Model answer is empty or not a string.")
    if not isinstance(findings_raw, list) or not isinstance(missing, list):
        raise ModelOutputError("Model findings or missing_evidence is not a list.")
    if status not in ("available", "unavailable", "not_applicable"):
        raise ModelOutputError("Model comparison_status is invalid.")

    package_status = package.comparison_status
    if package.question_type == QUESTION_COMPARE and status != package_status:
        raise ModelOutputError(
            f"Model reported comparison {status!r} but the evidence says {package_status!r}."
        )
    if package_status != "available" and status == "available":
        raise ModelOutputError("Model reported an available comparison that the evidence does not support.")

    valid_ids = package.item_ids
    warnings: list[str] = []
    comparison_blocked = package_status != "available"

    findings: list[Finding] = []
    for index, raw in enumerate(findings_raw[:MAX_FINDINGS], start=1):
        if not isinstance(raw, dict) or set(raw) != {"statement", "kind", "citations"}:
            warnings.append(f"Finding {index} was malformed and was dropped.")
            continue
        statement, kind, citations = raw["statement"], raw["kind"], raw["citations"]
        if not isinstance(statement, str) or not statement.strip() or kind not in ("observation", "hypothesis"):
            warnings.append(f"Finding {index} was malformed and was dropped.")
            continue
        if not isinstance(citations, list) or not citations or any(not isinstance(c, str) for c in citations):
            warnings.append(f"Finding {index} had no citations and was dropped.")
            continue
        unknown = [c for c in citations if c not in valid_ids]
        if unknown:
            warnings.append(f"Finding {index} cited evidence not in the request ({', '.join(unknown[:5])}) and was dropped.")
            continue
        if comparison_blocked and _COMPARATIVE.search(statement) and not _UNAVAILABLE_MARKERS.search(statement):
            warnings.append(f"Finding {index} made a comparative claim while comparison is unavailable and was suppressed.")
            continue
        findings.append(Finding(statement.strip()[:MAX_STATEMENT_CHARS], kind, tuple(dict.fromkeys(citations))))
    if len(findings_raw) > MAX_FINDINGS:
        warnings.append(f"Only the first {MAX_FINDINGS} findings were kept.")

    answer_text = answer.strip()[:MAX_ANSWER_CHARS]
    unknown_inline = sorted({cid for cid in _ID_PATTERN.findall(answer_text) if cid not in valid_ids})
    if unknown_inline:
        answer_text = _ID_PATTERN.sub(lambda m: m.group(0) if m.group(1) in valid_ids else "", answer_text)
        warnings.append("Removed inline citation(s) not present in the request: " + ", ".join(unknown_inline))
    if comparison_blocked:
        answer_text, removed = _strip_comparative_sentences(answer_text)
        if removed:
            warnings.append(f"Removed {removed} comparative sentence(s) because the comparison is unavailable.")
        if package.question_type == QUESTION_COMPARE and "comparison unavailable" not in answer_text.lower():
            answer_text = "Comparison unavailable. " + answer_text
    if not findings and not answer_text:
        raise ModelOutputError("No valid, cited content remained after validation.")
    if not findings:
        warnings.append("The model returned no valid cited findings.")

    return AnalystAnswer(
        mode="model",
        answer=answer_text,
        findings=findings,
        missing_evidence=[str(m).strip()[:400] for m in missing if isinstance(m, str) and m.strip()][:20],
        comparison_status=status,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Deterministic calculated diagnostics (no model)
# ---------------------------------------------------------------------------

def _join(reasons: list[str], limit: int) -> str:
    return "; ".join(str(r).rstrip(". ") for r in reasons[:limit])


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}{suffix}"
    return f"{value}{suffix}"


def calculated_answer(package: EvidencePackage) -> AnalystAnswer:
    findings: list[Finding] = []
    cfg_id = package.configuration.get("evidence_id")
    cov_id = package.data_coverage.get("evidence_id")
    cmp_id = package.comparison.get("evidence_id")
    session = package.session
    decisions = package.data_coverage.get("decisions", {})

    def add(statement: str, *ids: str | None) -> None:
        cited = tuple(i for i in ids if i)
        if cited:
            findings.append(Finding(statement, "observation", cited))

    scope = "Unbound log day" if session.get("kind") != "bound" else f"Session {session.get('session_id')}"
    rng = session.get("time_range_et", {})
    add(
        f"{scope}: records from {rng.get('start') or 'n/a'} to {rng.get('end') or 'n/a'}; "
        f"{decisions.get('distinct_decisions_in_entry_window', 0)} distinct decisions in the entry window "
        f"(expected {_fmt(decisions.get('expected_decisions'))}).",
        cov_id,
    )
    add(f"Configuration status: {package.configuration.get('status')}. {package.configuration.get('detail', '')}", cfg_id)

    if package.question_type == QUESTION_SYMBOL:
        m = package.metrics.get("symbol", {})
        mid = m.get("evidence_id")
        add(
            f"{m.get('symbol')}: {m.get('evaluations', 0)} evaluations; actions {m.get('actions') or {}}; "
            f"hold reasons {m.get('hold_reasons') or {}}.",
            mid,
        )
        if m.get("risk_blocks_by_action_and_reason"):
            add(f"Risk blocks: {m['risk_blocks_by_action_and_reason']}.", mid)
        add(
            f"Orders submitted {m.get('orders_submitted_by_side') or {}}, fill events {m.get('fill_events', 0)} "
            f"({_fmt(m.get('fill_events_unique_orders'))} unique orders), "
            f"position closes {m.get('position_close_events', 0)}, realized P&L {_fmt(m.get('realized_pnl_usd'), ' USD')}.",
            mid,
        )
        add(f"Last logged action: {m.get('last_action') or 'n/a'} (hold reason {m.get('last_rejection') or 'none'}).", mid)
        headline = (
            f"{m.get('symbol')} had {m.get('evaluations', 0)} logged evaluations in this scope with actions "
            f"{m.get('actions') or {}}."
        )
    else:
        s = package.metrics.get("session", {})
        sig, ex, pos, risk = s.get("signals", {}), s.get("execution", {}), s.get("positions", {}), s.get("risk", {})
        add(
            f"Signals: {sig.get('buy', 0)} BUY, {sig.get('sell', 0)} SELL, {sig.get('hold', 0)} HOLD of "
            f"{sig.get('evaluations', 0)} evaluations; top hold reasons {sig.get('hold_reasons') or {}}.",
            sig.get("evidence_id"),
        )
        add(
            f"Orders submitted: {ex.get('orders_submitted_buy', 0)} buy, {ex.get('orders_submitted_sell', 0)} sell; "
            f"average slippage {_fmt(ex.get('avg_slippage_bps'), ' bps')}.",
            ex.get("evidence_id"),
        )
        win = _fmt(pos.get("win_rate"))
        if not pos.get("win_rate_sample_sufficient"):
            win += " (sample too small to interpret)"
        add(
            f"Positions: {pos.get('opened', 0)} opened, {pos.get('close_events', 0)} close events; "
            f"total P&L {_fmt(pos.get('total_pnl_usd'), ' USD')}; win rate {win}.",
            pos.get("evidence_id"),
        )
        if risk.get("buy_blocks_by_reason") or risk.get("kill_switch_triggered"):
            add(
                f"Risk: kill switch triggered={risk.get('kill_switch_triggered')}; buy blocks {risk.get('buy_blocks_by_reason') or {}}.",
                risk.get("evidence_id"),
            )
        headline = (
            f"{scope} logged {sig.get('evaluations', 0)} signal evaluations, "
            f"{ex.get('orders_submitted_buy', 0) + ex.get('orders_submitted_sell', 0)} submitted orders and "
            f"{pos.get('close_events', 0)} position closes."
        )

    status = package.comparison_status
    reasons = package.comparison.get("reasons") or []
    if status == "available":
        add("Comparison available: all validity checks passed.", cmp_id)
        for item in package.items:
            if item.kind == "comparison" and item.id != cmp_id:
                add(item.title + ".", item.id)
    else:
        add("Comparison unavailable: " + _join(reasons, 4) + ".", cmp_id)
    if package.question_type == QUESTION_COMPARE:
        headline = (
            "Comparison available: all validity checks passed."
            if status == "available"
            else "Comparison unavailable. " + _join(reasons, 3) + "."
        )
    return AnalystAnswer(
        mode="calculated",
        answer=headline,
        findings=findings,
        missing_evidence=list(package.limitations),
        comparison_status=status if package.question_type == QUESTION_COMPARE else "not_applicable",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def ask(
    package: EvidencePackage,
    settings: AskSettings,
    *,
    client: ResponsesClient | None = None,
) -> AskResult:
    """Explain ``package``.  Never raises for model/credential problems."""

    try:
        request = build_request(package, settings)
    except RuntimeError as exc:
        return AskResult(calculated_answer(package), str(exc), None)
    if not settings.api_key and client is None:
        return AskResult(calculated_answer(package), None, request)
    try:
        responses = client or default_client(settings)
        response = responses.create(**request, timeout=settings.timeout_seconds)
    except Exception as exc:
        return AskResult(calculated_answer(package), _sanitize(_error_message(exc, settings), settings), request)
    try:
        answer = validate_model_output(extract_output_text(response), package)
    except ModelOutputError as exc:
        return AskResult(calculated_answer(package), _sanitize(str(exc), settings), request)
    answer.model = settings.model
    answer.response_id = _get(response, "id")
    return AskResult(answer, None, request)


__all__ = [
    "ANSWER_SCHEMA",
    "AnalystAnswer",
    "AskResult",
    "Finding",
    "ModelOutputError",
    "QUESTION_EXPLAIN",
    "ask",
    "build_request",
    "calculated_answer",
    "validate_model_output",
]
