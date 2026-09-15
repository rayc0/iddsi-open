from pathlib import Path

from fixtures.generate import generate_fixture
from train.engine import run_experiment


def test_cpu_pipeline_end_to_end(tmp_path: Path):
    manifest = generate_fixture(tmp_path / "fixture", per_level_per_split=1, size=24)
    cfg = {
        "experiment": {"name": "e2e", "output_dir": str(tmp_path / "runs"), "seeds": [3]},
        "data": {
            "manifest": str(manifest), "image_size": 24, "num_workers": 0, "batch_size": 5,
            "level_field": "level", "split_field": "split", "group_field": "group_id", "synthetic": True,
        },
        "model": {"kind": "tiny_cnn", "name": "test", "dropout": 0.0, "pretrained": False},
        "training": {
            "weight_decay": 0.0, "class_balance_beta": 0.99, "gradient_clip_norm": 1.0,
            "amp": False, "device": "cpu",
            "phases": [{"name": "head", "epochs": 1, "lr": 0.001, "unfreeze_last_blocks": 0}],
        },
        "calibration": {
            "target_under_fnr": 0.4, "max_abstention": 0.4,
            "temperature_min": 0.5, "temperature_max": 2.0, "temperature_steps": 5,
        },
        "evaluation": {"bootstrap_samples": 5, "bootstrap_seed": 2},
    }
    summaries = run_experiment(cfg)
    output = Path(summaries[0]["output_dir"])
    assert (output / "best.pt").exists()
    assert (output / "test_predictions.npz").exists()
    assert (output / "test_structured_outputs.json").exists()
    assert "NaN" not in (output / "metrics.json").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in (output / "report.md").read_text(encoding="utf-8")
