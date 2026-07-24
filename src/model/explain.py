"""
SHAP-based explainability for the LightGBM model.

The SHAP Formatter converts raw SHAP vectors into a structured business-labeled
dict that the Investigation Agent can reason about directly. The LLM never sees
raw feature names like "V217" — only human-readable labels with direction and magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# Map anonymized column names to business concepts.
# Extend this as you identify more column meanings during EDA.
FEATURE_LABELS: dict[str, str] = {
    "TransactionAmt": "Transaction amount",
    "ProductCD": "Product code (C S H R or W)",
    "card4": "Card network (discover, mastercard, visa, americna express)",
    "card6": "Card type (debit/credit/charge)",
    "P_emaildomain_binned": "How common is purchaser email domain",
    "R_emaildomain_binned": "How common is recipient email domain",
    "dist1": "Distance between billing and purchaser address",
    "hour_sin": "Hour of day (cyclic sine encoding)",
    "hour_cos": "Hour of day (cyclic cosine encoding)",
    "amt_log": "Transaction amount (log-scaled)",
    "amt_cents": "Cents portion of the transaction amount",
    "amt_is_round": "Transaction amount is a round dollar figure",
    "email_match": "Purchaser and recipient email domains match",
    # "P_provider": "Purchaser email provider family (Google/Yahoo/Microsoft/other)",
    "P_emaildomain_binned": "Purchaser email domain (rare domains grouped as Others)",
    "R_emaildomain_binned": "Recipient email domain (rare domains grouped as Others)",
    "uid_count": "How many prior transactions from this client (synthetic client id)",
    "uid_amt_mean": "This client's average transaction amount",
    "uid_amt_std": "This client's transaction amount variability",
    "uid_recency": "Seconds since this client's previous transaction",
    "uid_age": "Seconds since this client's first seen transaction",
    "DeviceInfo_freq": "How common this device string is in the training data",
}


# Generic per-family descriptions, from README.md's column breakdown — used as a
# fallback for any Cn/Dn/Mn/Vn/id_n column not specifically labeled in FEATURE_LABELS
# above (there are too many, e.g. 339 V-columns, to hand-label individually).
FAMILY_DESCRIPTIONS: dict[str, str] = {
    "C": "Masked counting info",
    "D": "Time delta (e.g. days since a previous transaction)",
    "M": "Match flag (e.g. name on card matches billing name)",
    "V": "Engineered feature (ranking, counting, etc.)",
    "id_": "Network connection / digital signature info",
    "card": "Card payment information, card type "
}


def _generic_label(base_name: str) -> str | None:
    """Return a generic '<family> <number>' label for Cn/Dn/Mn/Vn/id_n, else None."""
    if base_name.startswith("id_"):
        return f"{FAMILY_DESCRIPTIONS['id_']} {base_name}"
    if base_name.startswith("card"):
        return f"{FAMILY_DESCRIPTIONS['card']} {base_name}"
    if base_name[:1] in ("C", "D", "M", "V") and base_name[1:].isdigit():
        family = base_name[0]
        return f"{FAMILY_DESCRIPTIONS[family]} {base_name}"
    return None


def get_feature_label(feature_name: str) -> str:
    """Return a human-readable label for a feature, or the name itself if unknown.

    Specifically-labeled columns in FEATURE_LABELS take priority. Otherwise, any
    Cn/Dn/Mn/Vn/id_n column (optionally with a "_freq" suffix from frequency encoding)
    falls back to a generic per-family description instead of the raw column name.
    """
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]

    is_freq = feature_name.endswith("_freq")
    base_name = feature_name[: -len("_freq")] if is_freq else feature_name

    generic = _generic_label(base_name)
    if generic is None:
        return feature_name

    if is_freq:
        return f"How common this value is in the training data ({generic})"
    return generic


class SHAPExplainer:
    """Wraps shap.TreeExplainer for LightGBM with business-label formatting."""

    def __init__(
        self,
        model,
        feature_names: list[str],
        category_maps: dict[str, dict[int, str]] | None = None,
    ) -> None:
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names
        # {column: {code: original_category}} for columns that were integer-encoded
        # (e.g. card6 2 -> "debit") — lets explain_transaction show the real category
        # instead of the encoded number. Optional: omitted for numeric-only pipelines.
        self.category_maps = category_maps or {}

    def _decode_value(self, feature: str, value: float | None):
        """Map an encoded category code back to its original label, if known."""
        col_map = self.category_maps.get(feature)
        if col_map is None or value is None:
            return value
        return col_map.get(int(value), f"<unseen category {int(value)}>")

    def explain_global(self, X: pd.DataFrame, max_display: int = 20) -> None:
        """SHAP (beeswarm) — shows global feature importance on dataset X."""
        shap_values = self.explainer(X)
        shap.plots.beeswarm(shap_values)

    def explain_transaction(
        self,
        row: pd.Series | pd.DataFrame,
        top_n: int = 10,
    ) -> list[dict]:
        """Compute SHAP values for a single transaction and return formatted dicts.

        Returns a list of dicts sorted by |SHAP contribution| descending:
        [
            {
                "feature": "D1",
                "business_label": "Days since last transaction on this card",
                "value": 3.0,
                "shap_contribution": 0.08,
                "direction": "increases_fraud_risk",
            },
            ...
        ]
        """
        if isinstance(row, pd.Series):
            row = row.to_frame().T

        shap_values = self.explainer(row)
        # shap_values is a shap.Explanation; .values pulls out the plain numpy array —
        # zipping over the Explanation object itself yields more Explanation slices,
        # not floats, which is what caused "float() argument ... not 'Explanation'".
        if isinstance(shap_values, list):
            # XGBoost/LightGBM may return [neg_class, pos_class] (legacy API convention)
            shap_vals = np.asarray(shap_values[1][0])
        else:
            vals = shap_values.values[0]
            # binary classifier: new SHAP API can return (n_features, n_classes) per row —
            # take the positive-class column in that case, else it's already 1D
            shap_vals = vals[:, 1] if vals.ndim == 2 else vals

        feature_vals = row.iloc[0]
        results = []
        for feat, shap_val, feat_val in zip(self.feature_names, shap_vals, feature_vals):
            raw_value = float(feat_val) if pd.notna(feat_val) else None
            results.append({
                "feature": feat,
                "business_label": get_feature_label(feat),
                "value": self._decode_value(feat, raw_value),
                "shap_contribution": float(shap_val),
                "direction": "increases_fraud_risk" if shap_val > 0 else "decreases_fraud_risk",
            })

        results.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)
        return results[:top_n]

    def plot_waterfall(self, row: pd.Series | pd.DataFrame, top_n: int = 15) -> None:
        """SHAP waterfall plot for a single transaction."""
        if isinstance(row, pd.Series):
            row = row.to_frame().T
        shap_values = self.explainer(row)
        shap.plots.waterfall(shap_values[0], max_display=top_n, show=True)

    def format_for_llm(self, shap_dicts: list[dict], fraud_prob: float) -> str:
        """Format SHAP explanation as a structured text block for the LLM agent prompt."""
        lines = [
            f"Fraud probability: {fraud_prob:.3f} ({fraud_prob*100:.1f}%)",
            "",
            "Top contributing factors:",
        ]
        for i, d in enumerate(shap_dicts, 1):
            direction = "↑ fraud" if d["direction"] == "increases_fraud_risk" else "↓ fraud"
            if d["value"] is None:
                val_str = "NaN"
            elif isinstance(d["value"], (int, float)):
                val_str = f"{d['value']:.2f}"
            else:
                val_str = str(d["value"])  # decoded category label, e.g. "debit"
            lines.append(
                f"  {i}. {d['business_label']} = {val_str} "
                f"[{direction}, SHAP={d['shap_contribution']:+.4f}]"
            )
        return "\n".join(lines)
