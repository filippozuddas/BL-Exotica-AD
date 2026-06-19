"""
One-class anomaly scorer over ViT-MAE encoder embeddings.

The ``embedding`` anomaly score (default for the ViT-MAE backbone): fit a
one-class classifier (Isolation Forest / One-Class SVM, scikit-learn) on the
encoder embeddings of *normal* training data, then score new snippets by how far
they fall outside that learned "normality" boundary. This separates
representation learning (the ViT encoder) from anomaly scoring, so detection
does not depend on reconstruction magnitude — the failure mode diagnosed for
plain reconstruction-error scoring on narrowband data.

Shape-agnostic: operates purely on ``(B, embed_dim)`` embeddings produced by
``model.encode`` (see ``src/models/vit_mae.py``), so the same scorer works for
every GBT product. Persisted with ``joblib`` alongside the model checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import torch

__all__ = ["OneClassScorer"]


class OneClassScorer:
    """Wrap a scikit-learn one-class estimator around ``model.encode`` embeddings.

    Higher ``score`` = more anomalous. ``estimator`` is one of
    ``"isolation_forest"`` (default) or ``"ocsvm"``. The anomaly score is the
    negated ``decision_function`` (sklearn convention: higher decision_function
    = more normal), so larger scores rank as more anomalous, consistent with the
    reconstruction/InfoNCE scores.
    """

    def __init__(self, estimator: str = "isolation_forest", **kwargs):
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM

        self.estimator_name = estimator
        if estimator == "isolation_forest":
            self.estimator = IsolationForest(random_state=kwargs.pop("random_state", 42), **kwargs)
        elif estimator == "ocsvm":
            self.estimator = OneClassSVM(**kwargs)
        else:
            raise ValueError(f"Unknown estimator '{estimator}'. Use 'isolation_forest' or 'ocsvm'.")
        self._fitted = False

    @staticmethod
    def _embed_batches(model, loader: Iterable, device: Union[str, torch.device]) -> np.ndarray:
        """Collect ``model.encode`` embeddings over an iterable of NCHW batches."""
        model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device)
                embeddings.append(model.encode(x).cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def fit(self, model, loader: Iterable, device: Union[str, torch.device] = "cpu") -> "OneClassScorer":
        """Fit on encoder embeddings of normal training data."""
        feats = self._embed_batches(model, loader, device)
        self.estimator.fit(feats)
        self._fitted = True
        return self

    def fit_embeddings(self, embeddings: np.ndarray) -> "OneClassScorer":
        """Fit directly on precomputed ``(N, embed_dim)`` embeddings."""
        self.estimator.fit(np.asarray(embeddings))
        self._fitted = True
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Per-sample anomaly score (higher = more anomalous) from ``(B, embed_dim)``."""
        if not self._fitted:
            raise RuntimeError("OneClassScorer must be fit before scoring.")
        return -self.estimator.decision_function(np.asarray(embeddings))

    def save(self, path: Union[str, Path]) -> None:
        import joblib

        joblib.dump({"estimator": self.estimator, "name": self.estimator_name, "fitted": self._fitted}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OneClassScorer":
        import joblib

        blob = joblib.load(path)
        obj = cls.__new__(cls)
        obj.estimator = blob["estimator"]
        obj.estimator_name = blob["name"]
        obj._fitted = blob["fitted"]
        return obj
