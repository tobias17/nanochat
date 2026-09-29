"""
Test the local metrics log of base_train: a fresh run starts a new file, a resumed run drops what the
previous attempt logged from the resume step onwards (it logs those steps again).

python -m pytest tests/test_metrics_log.py -v
"""

import json

from nanochat.common import MetricsLog, DummyWandb


def _steps(path):
    return [json.loads(line).get("step") for line in open(path)]


def test_metrics_log_fresh_and_resume(tmp_path):
    path = tmp_path / "run" / "metrics.jsonl"
    log = MetricsLog(str(path), DummyWandb())
    log.log({"config": {"depth": 4}}, to_wandb=False)
    for step in range(8):
        log.log({"step": step, "train/loss": 1.0})
    log.finish()
    assert _steps(path) == [None] + list(range(8))

    log = MetricsLog(str(path), DummyWandb(), resume_from_step=5)  # crashed after step 7, resumed from the step 5 checkpoint
    log.log({"config": {"depth": 4}}, to_wandb=False)
    for step in range(5, 10):
        log.log({"step": step, "train/loss": 1.0})
    log.finish()
    assert _steps(path) == [None] + list(range(5)) + [None] + list(range(5, 10))

    MetricsLog(str(path), DummyWandb()).finish()  # a fresh run with the same tag starts over
    assert _steps(path) == []
