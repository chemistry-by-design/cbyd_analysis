"""Complexity-score helpers for the target, product and key-step analyses. Scores come from ``complexity_features``; -9999 / -9990 marks a failure, and SPS is unnormalised, so additive.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SCORES = ["sascore", "npscore", "sps", "boettcher"]
SCORE_NAMES = {"sascore": "SAScore", "npscore": "NPScore", "sps": "SPS",
               "boettcher": "Böttcher's score"}


def clean_score(v):
    """NaN for a missing value or a scorer failure (<= -9990)."""
    return np.nan if (pd.isna(v) or v <= -9990) else v


def pool_and_normalise(frames: dict, numeric=("mw", "n_stereo")) -> pd.DataFrame:
    """Stack ``{dataset: complexity table}`` into one frame with a ``dataset`` column, min-max normalise each score over the POOLED data and add ``overall_product_complexity`` = mean of the four normalised scores (0-1)."""
    out = []
    for name, df in frames.items():
        d = df.copy(); d["dataset"] = name
        for c in SCORES + list(numeric):
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        for c in SCORES:
            if c in d:
                d.loc[d[c] <= -9990, c] = np.nan
        out.append(d)
    pooled = pd.concat(out, ignore_index=True)
    for c in SCORES:
        lo, hi = pooled[c].min(), pooled[c].max()
        pooled[c + "_norm"] = (pooled[c] - lo) / (hi - lo)
    pooled["overall_product_complexity"] = pooled[[c + "_norm" for c in SCORES]].mean(axis=1)
    return pooled


def by_dataset(pooled: pd.DataFrame, col: str) -> dict:
    """``{dataset: non-NaN values of col}`` from a :func:`pool_and_normalise` frame."""
    return {name: pooled.loc[pooled["dataset"] == name, col].dropna().values
            for name in pooled["dataset"].unique()}
