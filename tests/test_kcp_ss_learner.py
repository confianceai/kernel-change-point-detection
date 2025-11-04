import numpy as np
import pytest
from kcpdi.kcp_ss_learner import KcpLearner

# Mock kcp_ds for deterministic testing
import kcpdi.kcp_ss_learner as ksl
ksl.kcp_ds = lambda **kwargs: ([10, 30, 70], None)

@pytest.fixture
def mock_data():
    # Simple time series with 100 samples, 1 feature
    np.random.seed(0)
    return np.random.randn(100, 1)

def test_score_samples_shape(mock_data):
    learner = KcpLearner()
    scores = learner.score_samples(mock_data)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (mock_data.shape[0],)

def test_scores_range(mock_data):
    learner = KcpLearner(decay_param=1.0)
    scores = -learner.score_samples(mock_data)  # scores are negative internally
    assert np.all(scores >= 0) and np.all(scores <= 1)

def test_fit_sets_offset(mock_data):
    learner = KcpLearner(expected_frac_anomaly=0.05)
    learner.fit(mock_data)
    assert hasattr(learner, "offset_")
    # offset should be a scalar float (possibly negative)
    assert isinstance(learner.offset_, (float, np.floating))

def test_predict_output(mock_data):
    learner = KcpLearner(expected_frac_anomaly=0.05)
    learner.fit(mock_data)
    preds = learner.predict(mock_data)
    assert set(preds) <= {-1, 1}
    assert preds.shape == (mock_data.shape[0],)

def test_lower_scores_are_more_anomalous(mock_data):
    learner = KcpLearner(expected_frac_anomaly=0.05)
    learner.fit(mock_data)
    scores = learner.score_samples(mock_data)
    preds = learner.predict(mock_data)
    # Ensure that predicted outliers have lower decision_function values
    assert np.mean(scores[preds == -1]) < np.mean(scores[preds == 1])

def test_decision_function_and_predict(mock_data):
    learner = KcpLearner(expected_frac_anomaly=0.05)
    learner.fit(mock_data)

    decision = learner.decision_function(mock_data)
    preds = learner.predict(mock_data)

    assert decision.shape == (mock_data.shape[0],)
    assert set(preds) <= {-1, 1}
    # Outliers should correspond to smaller (more negative) decision values
    assert np.mean(decision[preds == -1]) < np.mean(decision[preds == 1])
