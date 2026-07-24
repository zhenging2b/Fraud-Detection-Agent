"""
Tools backing the two subagents: SHAP explanation and similar-fraud lookup.

Runtime dependencies (the fitted SHAPExplainer and the confirmed-fraud
dataframe) aren't available at import time — they're produced by the
training pipeline. Call set_tool_context() once at startup (see agents.py)
before invoking either tool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from langchain.tools import tool

from src.model.explain import SHAPExplainer


@dataclass
class ToolContext:
    explainer: SHAPExplainer
    fraud_cases: pd.DataFrame       # rows confirmed isFraud == 1, indexed by TransactionID
    similarity_features: list[str]  # numeric feature columns used for cosine similarity
    all_transactions: pd.DataFrame  # full feature set (train+val), indexed by TransactionID
    model: object                   # fitted classifier with .predict_proba, for fraud_prob lookup


_ctx: ToolContext | None = None


def set_tool_context(
    explainer: SHAPExplainer,
    fraud_cases: pd.DataFrame,
    similarity_features: list[str],
    all_transactions: pd.DataFrame,
    model: object,
) -> None:
    """Inject runtime dependencies. Call once at startup before invoking tools."""
    global _ctx
    _ctx = ToolContext(
        explainer=explainer,
        fraud_cases=fraud_cases,
        similarity_features=similarity_features,
        all_transactions=all_transactions,
        model=model,
    )


def _lookup_row(transaction_id: str) -> pd.Series:
    ctx = _require_context()
    try:
        tid = int(transaction_id)
    except ValueError:
        raise ValueError(f"transaction_id must be numeric, got {transaction_id!r}")
    if tid not in ctx.all_transactions.index:
        raise KeyError(f"TransactionID {tid} not found")
    return ctx.all_transactions.loc[tid]


def _require_context() -> ToolContext:
    if _ctx is None:
        raise RuntimeError("Tool context not set — call set_tool_context() before running agents.")
    return _ctx


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between vector `a` and each row of matrix `b`."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b_norm @ a_norm


def find_similar_fraud(row: pd.Series, top_k: int = 5) -> list[dict]:
    """Find the top_k confirmed-fraud transactions most similar to `row`.

    TODO(user): decide whether similarity_features should be raw features,
    SHAP-weighted features, or an embedding — this determines what "similar"
    means for the investigation narrative.
    """
    ctx = _require_context()
    feats = ctx.similarity_features
    query_vec = row[feats].fillna(0).to_numpy(dtype=float)
    corpus = ctx.fraud_cases[feats].fillna(0).to_numpy(dtype=float)

    sims = _cosine_similarity(query_vec, corpus)
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_idx:
        case = ctx.fraud_cases.iloc[idx]
        results.append({
            "transaction_id": int(case.name),
            "similarity": float(sims[idx]),
            **{f: (None if pd.isna(case[f]) else float(case[f])) for f in feats},
        })
    return results


def get_shap_explanation(row: pd.Series, fraud_prob: float, top_n: int = 10) -> str:
    """Return the formatted SHAP explanation text for a single transaction."""
    ctx = _require_context()
    shap_dicts = ctx.explainer.explain_transaction(row, top_n=top_n)
    return ctx.explainer.format_for_llm(shap_dicts, fraud_prob)


@tool("find_similar_fraud_cases", description="Find confirmed fraud transactions similar to the current one, to justify a fraud prediction")
def find_similar_fraud_cases_tool(transaction_id: str) -> list[dict]:
    row = _lookup_row(transaction_id)
    return find_similar_fraud(row)


@tool("read_shap_values", description="Read SHAP feature contributions for a transaction to explain why the model flagged it as fraud")
def read_shap_values_tool(transaction_id: str) -> str:
    ctx = _require_context()
    row = _lookup_row(transaction_id)
    fraud_prob = float(ctx.model.predict_proba(row.to_frame().T)[:, 1][0])
    return get_shap_explanation(row, fraud_prob)
