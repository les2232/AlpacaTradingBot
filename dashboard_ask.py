"""Streamlit tab for Ask TradeOS (read-only historical session analyst)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from tradeos.ask import (
    PROMOTED_BASELINE,
    QUESTION_COMPARE,
    QUESTION_SYMBOL,
    QUESTIONS,
    AskSettings,
    SessionRef,
    ask,
    build_evidence_package,
    discover_sessions,
    list_comparison_targets,
    load_ask_settings,
)
from tradeos.ask.records import load_day
from tradeos.runtime_paths import (
    TRADEOS_REPOSITORY_ROOT,
    resolve_database_path,
    resolve_explicit_database_path,
    resolve_log_root,
)

_RESULT_KEY = "ask_tradeos_result"
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def _md(text: object) -> str:
    """Escape model/log text so it renders literally (no links, images, or HTML)."""

    return _MD_SPECIAL.sub(r"\\\1", str(text))


def _paths() -> tuple[Path, Path | None, Path]:
    log_root = resolve_log_root()
    try:
        db_path = resolve_explicit_database_path() or resolve_database_path()
    except ValueError:
        db_path = None
    if db_path is not None and not db_path.is_file():
        db_path = None
    return log_root, db_path, TRADEOS_REPOSITORY_ROOT / "results" / "experiment_log.jsonl"


@st.cache_data(ttl=60, show_spinner=False)
def _cached_sessions(log_root: str, db_path: str | None) -> list[SessionRef]:
    return discover_sessions(Path(log_root), Path(db_path) if db_path else None)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_symbols(log_root: str, session_key: str, session_id: str | None, days: tuple[str, ...]) -> list[str]:
    symbols: set[str] = set()
    for day in days:
        for record in load_day(Path(log_root), day).records:
            if record.symbol and record.session_id == session_id:
                symbols.add(record.symbol)
    return sorted(symbols)


def _render_answer(package: Any, result: Any) -> None:
    answer = result.answer
    if answer.mode == "model":
        st.success(f"Model explanation · {answer.model}")
    else:
        st.info("Calculated diagnostics · computed in Python from the session records; no model was used.")
    if result.error:
        st.error(f"Model explanation unavailable: {result.error} Showing calculated diagnostics instead.")

    status = package.comparison_status
    reasons = package.comparison.get("reasons") or []
    if package.question_type != QUESTION_COMPARE:
        if status != "available":
            st.caption(
                "Backtest comparison unavailable for this session, so underperformance, drift, and improvement "
                "conclusions are suppressed. Ask the comparison question for the reasons."
            )
    else:
        if status == "available":
            st.success("Comparison available: every validity check passed.")
        else:
            st.warning(
                "**Comparison unavailable.** Underperformance, drift, and improvement conclusions are suppressed.\n\n"
                + "\n".join(f"- {_md(reason)}" for reason in reasons[:8])
            )

    st.markdown("#### Answer")
    st.markdown(_md(answer.answer))

    if answer.findings:
        st.markdown("#### Findings")
        for finding in answer.findings:
            badge = ":blue-background[Observation]" if finding.kind == "observation" else ":orange-background[Hypothesis]"
            cites = " ".join(f"`{cid}`" for cid in finding.citations)
            st.markdown(f"{badge} {_md(finding.statement)}")
            st.caption(f"Evidence: {cites}")
    for warning in answer.warnings:
        st.caption(f"Validation: {_md(warning)}")

    missing = answer.missing_evidence or package.limitations
    if missing:
        with st.expander(f"What cannot be established ({len(missing)})", expanded=False):
            for item in missing:
                st.markdown(f"- {_md(item)}")

    cited = [package.item(cid) for cid in answer.cited_ids]
    cited = [item for item in cited if item is not None]
    with st.expander(f"Supporting evidence ({len(cited)} cited)", expanded=False):
        if not cited:
            st.caption("No evidence items were cited.")
        for item in cited:
            st.markdown(f"**{item.id}** · {_md(item.title)}  \nSource: `{item.source}`")
            st.json(dict(item.data), expanded=False)

    with st.expander(f"All evidence in the request ({len(package.items)} items)", expanded=False):
        rows = [{"id": i.id, "kind": i.kind, "title": i.title, "source": i.source} for i in package.items]
        st.dataframe(rows, hide_index=True)

    if result.request_body is not None:
        with st.expander("Exact model request body (no credentials)", expanded=False):
            body = dict(result.request_body)
            try:
                body["input"] = json.loads(body["input"][0]["content"][0]["text"])
            except (KeyError, IndexError, TypeError, ValueError):
                pass
            st.caption(
                "This is what is sent when a model is configured: the evidence package only, with store=false."
            )
            st.json(body, expanded=False)


def render_ask_tradeos_tab(
    *,
    settings_loader: Callable[[], AskSettings] = load_ask_settings,
    client: Any | None = None,
) -> None:
    st.markdown("### Ask TradeOS")
    st.caption(
        "Read-only analyst for historical sessions. It explains calculated results from saved logs and "
        "history; it never submits orders or changes configuration, risk limits, or state."
    )
    settings = settings_loader()
    if settings.model_configured or client is not None:
        st.caption(_md(settings.describe()))
    else:
        st.info(
            "Model explanations are off because `OPENAI_API_KEY` is not set. Calculated diagnostics "
            "still work. To turn explanations on, add `OPENAI_API_KEY=...` to the repository `.env` "
            "(optionally `TRADEOS_ASK_MODEL`, default `gpt-6.1-sol`) and restart the dashboard."
        )

    log_root, db_path, experiment_log_path = _paths()
    try:
        sessions = _cached_sessions(str(log_root), str(db_path) if db_path else None)
    except Exception as exc:
        st.error(f"Could not read session history ({type(exc).__name__}).")
        return
    if not sessions:
        st.info(f"No historical sessions were found under `{_md(log_root.name)}/`.")
        return

    by_key = {s.key: s for s in sessions}
    left, middle = st.columns([3, 2])
    with left:
        session_key = st.selectbox(
            "Session",
            options=list(by_key),
            format_func=lambda key: by_key[key].label,
            key="ask_session",
            help="Bound sessions carry a session id. Unbound log days hold records without one.",
        )
    session = by_key[session_key]
    with middle:
        question = st.radio(
            "Question",
            options=list(QUESTIONS),
            format_func=lambda key: QUESTIONS[key],
            key="ask_question",
        )

    symbol: str | None = None
    target = PROMOTED_BASELINE
    if question == QUESTION_SYMBOL:
        symbols = _cached_symbols(str(log_root), session.key, session.session_id, session.days)
        if not symbols:
            st.warning("No symbol-level records exist for this session.")
            return
        symbol = st.selectbox("Symbol", options=symbols, key="ask_symbol")
    elif question == QUESTION_COMPARE:
        targets = dict(list_comparison_targets(experiment_log_path))
        target = st.selectbox("Backtest", options=list(targets), format_func=targets.get, key="ask_target")
        if not experiment_log_path.is_file():
            st.caption("results/experiment_log.jsonl is absent, so only the promoted baseline is listed.")

    if session.kind != "bound":
        st.caption(
            "This log day has records without a session id. Results describe the whole day and cannot be "
            "tied to one session or launch configuration."
        )

    selection = (session.key, question, symbol, target)
    label = "Ask TradeOS" if (settings.model_configured or client is not None) else "Show calculated diagnostics"
    if st.button(label, type="primary", key="ask_submit"):
        try:
            with st.spinner("Assembling evidence from session records…"):
                package = build_evidence_package(
                    session,
                    question,
                    log_root=log_root,
                    repo_root=TRADEOS_REPOSITORY_ROOT,
                    db_path=db_path,
                    symbol=symbol,
                    comparison_target=target,
                    experiment_log_path=experiment_log_path,
                )
            spinner = (
                f"Asking {settings.model}…" if (settings.model_configured or client is not None) else "Calculating…"
            )
            with st.spinner(spinner):
                result = ask(package, settings, client=client)
        except Exception as exc:
            st.session_state.pop(_RESULT_KEY, None)
            st.error(f"Ask TradeOS could not build an answer ({type(exc).__name__}: {_md(str(exc)[:200])}).")
            return
        st.session_state[_RESULT_KEY] = (selection, package, result)

    stored = st.session_state.get(_RESULT_KEY)
    if stored and stored[0] == selection:
        _render_answer(stored[1], stored[2])
    elif stored:
        st.caption("Selection changed. Press the button to answer the new question.")
