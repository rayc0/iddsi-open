import numpy as np

from eval.calibration import calibrate_abstention_threshold, fit_temperature, softmax
from eval.metrics import evaluate_predictions
from eval.report import render_markdown_report


def test_calibration_and_metrics_include_unclear_and_intervals():
    labels = np.tile(np.arange(5), 3)
    logits = np.full((15, 5), -2.0)
    logits[np.arange(15), labels] = 2.0
    logits[0] = 0.0
    temperature, _ = fit_temperature(logits, labels, steps=9)
    probabilities = softmax(logits / temperature)
    selection = calibrate_abstention_threshold(probabilities, labels, 0.2, 0.4)
    result = evaluate_predictions(
        probabilities, labels, selection["threshold"], bootstrap_samples=20, bootstrap_seed=4
    )
    assert len(result["confusion_matrix"]) == 6
    assert len(result["confusion_matrix"][0]) == 6
    assert "macro_f1_with_abstentions" in result["confidence_intervals_95"]
    assert result["risk_coverage_curve"]
    report = render_markdown_report(result, title="Test", synthetic=True, calibration={"temperature": temperature, **selection})
    assert "SYNTHETIC DATA" in report
    assert "unclear" in report

