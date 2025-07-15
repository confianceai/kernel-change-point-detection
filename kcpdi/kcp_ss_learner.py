"""
This module, `kcp_ss_learner` (Kernel Change Point Sample Scorer Learner), includes a wrapper to
perform anomaly scoring as understood in the TADkit formalism, based on this library's change-point
detection methodology.

The kernel change-point detection algorithm outputs a final list of time indices at which it thinks
that "something happened". 

Other methods output a score at __all__ time indices, where the higher the score is, the more we
believe that "something happened there". 

In order to allow the kernel change-point method to be integrated into a Python package based
around score samples, we have implemented a function __kcp_ss__ which takes the __kcp_ds__ output
list of change-points and turns them into scores at __all time indices__.

Remember that the kernel change-point detection algorithm truly believes that it has found the true
and only set of change-points. However, due to random noise, it could have been that a true
change-point was immediately before or immediately after a detected change-point, i.e., 1 or 2 or
maybe even 3 time indices before or after.

We give a score of 1 at each detected change-point. Other points are assigned a score that
decreases as they move away from the detected change-point, in a way parameterized by a decay
parameter (gamma).
"""
from typing import Any, Dict, List, Literal, Optional, Sequence

from sklearn.base import BaseEstimator, OutlierMixin

from kcpdi import kcp_ds

import numpy as np


class KcpLearner(BaseEstimator, OutlierMixin):
    required_properties = []

    params_description = {
        "decay_param": {
            "description": "How fast the score decays around the change-points.",
            "family": "postprocessing",
            "value_type": "log_range",
            "log_start": -2,
            "log_step": 0.1,
            "log_stop": 2,
            "default": 1,
        }
    }

    def __init__(
            self,
            kernel: Literal["linear", "cosine", "rbf"] = "linear",
            params: Optional[Dict[str, Any]] = None,
            max_n_time_points: int = 2000,
            min_n_time_points: int = 10,
            expected_frac_anomaly: float = 1 / 1000,
            decay_param: float = 1.,
    ):
        self.kernel = kernel
        self.params = params
        self.max_n_time_points = max_n_time_points
        self.min_n_time_points = min_n_time_points
        self.expected_frac_anomaly = expected_frac_anomaly
        self.decay_param = decay_param

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _kcp_ss(
            detected_change_points: Sequence[int],
            n_time_points: int,
            decay_param: float = 1.,
    ) -> List[float]:
        """Take the output list of detected change-points from the
        function kcp_ds() and transform them into a score at each time
        point from 0 (low probability of an anomaly at the time point)
        to 1 (high probability of an anomaly at the time point).

        Args:   
            detected_change_points: The detected change-points. It
                should be the first output of the kcp_ds function.
            n_time_points: The number of time points in the input data. 
                Usually this will be something like: np.shape(data)[0].
            decay_param: Parameter that defines how fast the score
                decays around the change-points. The larger this value,
                the faster it decays.
        """
        if decay_param < 0:
            raise ValueError(
                "decay_param has to be positive."
            )

        # Create list of 0s of length n_time_points
        kcp_scores = [0] * n_time_points

        if not len(detected_change_points):
            return kcp_scores

        for i in range(n_time_points):

            # Cases:
            # 1) i is in the list of change-point indices
            # 2) i is before the first change-point index
            # 3) i is between two change-point indices
            # 4) i is after the last change-point index

            # Create a local version of the input variable detected_change_points
            # and append i to it:
            local_detected_change_points = detected_change_points.copy()
            local_detected_change_points.append(i)

            # A couple of useful values:
            sum_first = sum(i < j for j in detected_change_points)
            sum_last = sum(i > j for j in detected_change_points)

            # Case (1)
            if len(set(local_detected_change_points)) == len(detected_change_points):
                kcp_scores[i] = 1

            # Case (2)
            elif sum_first == len(detected_change_points):
                # We only have the distance to the first change-point index to deal with:
                curr_dist = abs(i - detected_change_points[0])
                kcp_scores[i] = (1 / 2) ** (curr_dist * decay_param)

            # Case (4)
            elif sum_last == len(detected_change_points):
                # We only have the distance to the last change-point index to deal with:
                curr_dist = abs(i - detected_change_points[-1])
                kcp_scores[i] = (1 / 2) ** (curr_dist * decay_param)

            # Case (3)
            else:
                n_left = sum(i > j for j in detected_change_points)
                curr_dist_left = abs(i - detected_change_points[n_left - 1])
                curr_dist_right = abs(i - detected_change_points[n_left])
                kcp_scores[i] = (1 / 2) * (
                        (1 / 2) ** (curr_dist_left * decay_param) +
                        (1 / 2) ** (curr_dist_right * decay_param)
                )

        return kcp_scores

    def score_samples(self, X):
        if hasattr(X, "values"):
            # small hack to handle DataFrame s
            X = X.values
        detected_change_points, _ = kcp_ds(
            data=X,
            kernel=self.kernel,
            params=self.params,
            max_n_time_points=self.max_n_time_points,
            min_n_time_points=self.min_n_time_points,
            expected_frac_anomaly=self.expected_frac_anomaly,
        )

        return np.array([- score for score in self._kcp_ss(
            detected_change_points=detected_change_points,
            n_time_points=len(X),
            decay_param=self.decay_param
        )])
