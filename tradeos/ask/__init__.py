"""Ask TradeOS: read-only, evidence-backed explanations of historical sessions.

Layers:
* ``records``    - provenance-preserving loading and the session catalog
* ``evidence``   - session/symbol isolation and the citable evidence package
* ``comparison`` - fail-closed backtest comparison validity
* ``analyst``    - Responses API call, output validation, no-model fallback
* ``settings``   - server-side model/credential configuration

Nothing here submits orders, writes runtime state, or changes configuration.
"""

from tradeos.ask.analyst import AnalystAnswer, AskResult, ask, calculated_answer
from tradeos.ask.comparison import PROMOTED_BASELINE, list_comparison_targets
from tradeos.ask.evidence import (
    QUESTION_COMPARE,
    QUESTION_EXPLAIN,
    QUESTION_SYMBOL,
    QUESTIONS,
    EvidencePackage,
    build_evidence_package,
)
from tradeos.ask.records import SessionRef, discover_sessions
from tradeos.ask.settings import AskSettings, load_ask_settings

__all__ = [
    "AnalystAnswer",
    "AskResult",
    "AskSettings",
    "EvidencePackage",
    "PROMOTED_BASELINE",
    "QUESTIONS",
    "QUESTION_COMPARE",
    "QUESTION_EXPLAIN",
    "QUESTION_SYMBOL",
    "SessionRef",
    "ask",
    "build_evidence_package",
    "calculated_answer",
    "discover_sessions",
    "list_comparison_targets",
    "load_ask_settings",
]
