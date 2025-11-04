"""
Kernel Change Point Sample Scorer Learner (`kcp_ss_learner`)

This module provides a Scikit-Learn-style wrapper for performing anomaly scoring
compatible with the TADkit formalism, based on the Kernel Change Point (KCP) detection
methodology.

The KCP algorithm identifies a discrete list of time indices at which change-points
are likely to occur. This wrapper translates those discrete detections into a continuous
score for *every* time index, with values in [0, 1], where higher values indicate a
greater likelihood of an anomaly occurring at that point.

The score decays exponentially with distance from each detected change-point,
controlled by the `decay_param` parameter.
"""

from __future__ import annotations
from typing import Any, Dict, Literal, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin

from kcpdi.kcp_ds import kcp_ds


class KcpLearner(BaseEstimator, OutlierMixin):
    """
    Scikit-Learn–style learner that converts change-point detections
    into dense anomaly scores across all time indices.

    Attributes
    ----------
    kernel : {"linear", "cosine", "rbf"}
        Kernel type used for the change-point detection.
    params : dict, optional
        Parameters passed to the kernel change-point detector.
    max_n_time_points : int
        Maximum allowed number of time points in a sequence.
    min_n_time_points : int
        Minimum allowed number of time points in a sequence.
    expected_frac_anomaly : float
        Expected proportion of anomaly points in the dataset.
    decay_param : float
        Controls how rapidly scores decay away from detected change-points.
        Higher values → faster decay.
    """

    def __init__(
        self,
        kernel: Literal["linear", "cosine", "rbf"] = "linear",
        params: Optional[Dict[str, Any]] = None,
        max_n_time_points: int = 2000,
        min_n_time_points: int = 10,
        expected_frac_anomaly: float = 1 / 1000,
        decay_param: float = 1.0,
    ):
        if decay_param < 0:
            raise ValueError("decay_param must be non-negative.")
        self.kernel = kernel
        self.params = params
        self.max_n_time_points = max_n_time_points
        self.min_n_time_points = min_n_time_points
        self.expected_frac_anomaly = expected_frac_anomaly
        self.decay_param = decay_param

    def fit(self, X, y=None):
        """No fitting required; returns self for compatibility with Scikit-Learn."""
        return self

    @staticmethod
    def _kcp_ss(
        detected_change_points: Sequence[int],
        n_time_points: int,
        decay_param: float = 1.0,
    ) -> np.ndarray:
        """
        Transform detected change-points into a dense anomaly score.

        Parameters
        ----------
        detected_change_points : sequence of int
            Indices of detected change-points from `kcp_ds`.
        n_time_points : int
            Total number of time indices in the sequence.
        decay_param : float, default=1.0
            Controls the exponential decay rate away from each change-point.

        Returns
        -------
        np.ndarray
            Array of shape (n_time_points,) containing anomaly scores ∈ [0, 1].
        """
        if n_time_points <= 0:
            raise ValueError("n_time_points must be a positive integer.")
        if decay_param < 0:
            raise ValueError("decay_param must be non-negative.")

        detected_change_points = sorted(set(detected_change_points))
        scores = np.zeros(n_time_points, dtype=float)

        if not detected_change_points:
            return scores

        # Create an array of distances to the nearest change-point for each index
        indices = np.arange(n_time_points)
        distances = np.full(n_time_points, np.inf)

        for cp in detected_change_points:
            # Update with smaller distances as we go
            new_distances = np.abs(indices - cp)
            distances = np.minimum(distances, new_distances)

        # Compute decayed scores: 1 at cp, decaying exponentially with distance
        scores = (0.5) ** (distances * decay_param)
        scores[detected_change_points] = 1.0  # ensure exact peaks

        return scores

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute negative anomaly scores for each time index in X.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) of negative anomaly scores.
        """
        if hasattr(X, "values"):  # handle pandas DataFrame
            X = X.values

        detected_change_points, _ = kcp_ds(
            data=X,
            kernel=self.kernel,
            params=self.params,
            max_n_time_points=self.max_n_time_points,
            min_n_time_points=self.min_n_time_points,
            expected_frac_anomaly=self.expected_frac_anomaly,
        )

        scores = self._kcp_ss(
            detected_change_points=detected_change_points,
            n_time_points=len(X),
            decay_param=self.decay_param,
        )

        # Return negative scores to align with OutlierMixin conventions
        return -np.array(scores, dtype=float)
