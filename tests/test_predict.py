"""End-to-end tests for the online workpiece decision feature.

The inspection rule under test:

* a decision pins the task id, the last CONFIRMED round and its confirmed
  weights at acceptance time, then asks BOTH independent workers to score the
  same two-feature workpiece against that pinned version;
* only when task id, round and probability agree on both sides is a hard
  classification at the 0.5 threshold returned;
* any unreachable side, version/task mismatch or probability disagreement
  degrades to ``needs_review`` -- a single-side conclusion is never returned
  and training progress is never touched;
* a worker that persisted an unconfirmed next round, or restarted and has not
  been re-prepared yet, still serves predictions from the accepted snapshot.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from app import lrmath
from test_integration import (cluster, _create_and_start, _dataset_csv,
                              _wait_status)

FEATURES = [1.25, -0.4]


def _predict(base, x1=FEATURES[0], x2=FEATURES[1]):
    return httpx.post(base + "/api/predict", json={"x1": x1, "x2": x2},
                      timeout=15)


def _both_sides_ok(res) -> bool:
    return all(res["workers"][w]["ok"] for w in ("worker1", "worker2"))


# ----------------------------------------------------------------- happy paths
def test_predict_classified_when_both_sides_agree(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    st = _wait_status(base, lambda d: d["status"] == "completed")

    r = _predict(base)
    r.raise_for_status()
    res = r.json()

    assert res["verdict"] == "classified"
    assert res["classification"] in (0, 1)
    assert res["confirmed_round"] == st["confirmed_round"]
    assert 0.0 <= res["probability"] <= 1.0
    # Both workers reported the same task, version and probability.
    w1, w2 = res["workers"]["worker1"], res["workers"]["worker2"]
    assert w1["ok"] and w2["ok"]
    assert w1["task_id"] == w2["task_id"] == res["task_id"]
    assert w1["round"] == w2["round"] == res["confirmed_round"]
    assert w1["probability"] == w2["probability"] == res["probability"]
    # Threshold rule: p >= 0.5 -> defect (1), otherwise normal (0).
    assert res["classification"] == lrmath.classify(res["probability"], 0.5)
    # The served probability is exactly the confirmed-weight computation.
    expected = lrmath.predict_probability(FEATURES, st["w"])
    assert res["probability"] == expected

    # Prediction is read-only: training state is untouched.
    st_after = httpx.get(base + "/api/status", timeout=5).json()
    assert st_after["confirmed_round"] == st["confirmed_round"]
    assert st_after["status"] == "completed"
    assert st_after["loss_history"] == st["loss_history"]


def test_predict_works_with_zero_round_confirmed_model(cluster, tmp_path):
    """Right after task creation (round 0 weights, all-zero) the last
    confirmed model is the initial model and must already be usable."""
    base = cluster["coord"]
    csv_path = _dataset_csv(tmp_path)
    with open(csv_path, "rb") as f:
        httpx.post(base + "/api/job?rounds=6",
                   files={"file": ("data.csv", f, "text/csv")},
                   timeout=15).raise_for_status()
    res = _predict(base).json()
    assert res["verdict"] == "classified"
    assert res["confirmed_round"] == 0
    # sigmoid(0) = 0.5 -> classified as defect at the inclusive threshold.
    assert res["probability"] == 0.5 and res["classification"] == 1


# ------------------------------------------------- snapshot pinning while busy
def test_predict_during_training_pins_confirmed_snapshot(cluster, tmp_path):
    """A worker may already have persisted the unconfirmed next round
    (local_round ahead). The decision must still use the accepted confirmed
    version, never the local one. The ahead state is driven deterministically
    via the worker commit protocol rather than hoped-for timing."""
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=20)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 3)

    # Pause at a stable confirmed point, then push worker1 one unconfirmed
    # round ahead on its durable volume (mimics a commit landing just before
    # a crash; the coordinator never confirmed that round).
    deadline = time.time() + 15
    while time.time() < deadline:
        if httpx.post(base + "/api/pause", timeout=5).status_code == 200:
            break
        time.sleep(0.3)
    st = _wait_status(base, lambda d: d["status"] == "paused")
    confirmed = st["confirmed_round"]
    task_id, gen = st["task_id"], st["gen"]
    w1_port = cluster["w1"].port
    url = f"http://127.0.0.1:{w1_port}"
    g = httpx.post(url + "/gradient",
                   json={"task_id": task_id, "gen": gen,
                         "round": confirmed + 1}, timeout=5).json()
    c = httpx.post(url + "/commit",
                   json={"task_id": task_id, "gen": gen,
                         "round": confirmed + 1,
                         "grad": [0.001, -0.002, 0.003]}, timeout=5).json()
    assert c["round"] == confirmed + 1  # worker now one unconfirmed round ahead
    h = httpx.get(url + "/health", timeout=5).json()
    assert h["round"] == confirmed + 1

    # The decision stays on the confirmed snapshot even though worker1's
    # durable local round is ahead, and both sides still agree exactly.
    res = _predict(base).json()
    assert res["verdict"] == "classified"
    assert res["confirmed_round"] == confirmed
    w1side = res["workers"]["worker1"]
    assert w1side["round"] == confirmed          # version used for the score
    assert w1side["local_round"] == confirmed + 1  # local advance ignored
    expected = lrmath.predict_probability(FEATURES, st["w"])
    assert res["probability"] == expected
    assert res["workers"]["worker2"]["probability"] == expected

    # Cross-check against the immutable confirmed-round artifact on disk.
    artifact = Path(cluster["paths"]["coord"]) / "weights" / (
        f"weights_round_{confirmed:04d}.json")
    doc = json.loads(artifact.read_text())
    w = [doc["weights"]["bias"], doc["weights"]["w1"], doc["weights"]["w2"]]
    assert res["probability"] == lrmath.predict_probability(FEATURES, w)

    # Resume: /prepare rolls the worker back; training completes exactly.
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    final = _wait_status(base, lambda d: d["status"] == "completed")
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    assert [p["loss"] for p in final["loss_history"]] == ref["loss"]


# ------------------------------------------------------------- failure handling
def test_predict_needs_review_when_worker_down_no_single_side(cluster, tmp_path):
    base = cluster["coord"]
    # Slow rounds and make the run long enough that training is still in
    # flight when the worker is killed.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=40)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)
    assert httpx.get(base + "/api/status").json()["status"] == "running"
    cluster["w1"].hard_kill()
    _wait_status(base, lambda d: d["status"] == "paused")

    res = _predict(base).json()
    assert res["verdict"] == "needs_review"
    # No single-side conclusion is ever exposed.
    assert res["classification"] is None and res["probability"] is None
    assert res["reason"] and "worker1" in res["reason"]
    assert res["workers"]["worker1"]["ok"] is False
    # The healthy side did compute an answer, but it is withheld.
    assert res["workers"]["worker2"]["ok"] is True
    assert isinstance(res["workers"]["worker2"]["probability"], float)

    # Training progress is unchanged by the inspection request.
    st2 = httpx.get(base + "/api/status", timeout=5).json()
    assert st2["confirmed_round"] == st["confirmed_round"]
    assert st2["status"] == "paused"


def test_predict_worker_restart_during_request_then_recovers(cluster, tmp_path):
    """Kill a worker with a prediction in flight (the request must fail safe
    to needs_review), then restart it: the fresh generation-neutral process
    serves the next prediction from the confirmed snapshot without /prepare."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=10)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 3)

    cluster["w1"].hard_kill()
    # Request issued while worker1 is gone: must be needs_review, not a raise.
    res = _predict(base).json()
    assert res["verdict"] == "needs_review"

    # Bring the worker back as a fresh process (gen stays None until prepare).
    cluster["w1"].start()
    deadline = time.time() + 20
    while time.time() < deadline:
        h = httpx.get(f"http://127.0.0.1:{cluster['w1'].port}/health",
                      timeout=2)
        if h.status_code == 200:
            assert h.json()["gen"] is None  # not re-prepared yet
            break
        time.sleep(0.2)

    res = _predict(base).json()
    assert res["verdict"] == "classified"
    assert res["confirmed_round"] == st["confirmed_round"]
    assert _both_sides_ok(res)


def test_predict_uses_last_confirmed_model_while_paused_and_after_resume(
        cluster, tmp_path):
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.3}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=14)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)

    # Pause between rounds and keep predicting the last confirmed model.
    deadline = time.time() + 15
    while True:
        r = httpx.post(base + "/api/pause", timeout=5)
        if r.status_code == 200:
            break
        time.sleep(0.3)
    paused = _wait_status(base, lambda d: d["status"] == "paused")
    res = _predict(base).json()
    assert res["verdict"] == "classified"
    assert res["confirmed_round"] == paused["confirmed_round"]
    assert res["probability"] == lrmath.predict_probability(
        FEATURES, paused["w"])

    # Resume and complete; prediction now serves the final confirmed model.
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    final = _wait_status(base, lambda d: d["status"] == "completed")
    res = _predict(base).json()
    assert res["verdict"] == "classified"
    assert res["confirmed_round"] == final["confirmed_round"]
    assert res["probability"] == lrmath.predict_probability(FEATURES, final["w"])


def test_predict_task_mismatch_after_worker_kept_stale_shard(cluster, tmp_path):
    """If a worker missed the /load of a new task (it was down), it still
    holds the old shard: its task mismatch must force needs_review."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=4)
    _wait_status(base, lambda d: d["status"] == "completed")

    cluster["w1"].hard_kill()
    httpx.post(base + "/api/reset", timeout=5).raise_for_status()
    with open(_dataset_csv(tmp_path), "rb") as f:
        r = httpx.post(base + "/api/job?rounds=4",
                       files={"file": ("data.csv", f, "text/csv")},
                       timeout=15)
    assert r.status_code == 202  # shard load to worker1 failed -> paused

    cluster["w1"].start()  # still holds the OLD task's shard
    time.sleep(1.0)
    res = _predict(base).json()
    assert res["verdict"] == "needs_review"
    assert res["workers"]["worker1"]["ok"] is False
    assert res["workers"]["worker2"]["ok"] is True
    assert "worker1" in res["reason"]


# ------------------------------------------------------------------ guard tests
def test_predict_without_task_returns_404(cluster):
    r = _predict(cluster["coord"])
    assert r.status_code == 404


def test_predict_rejects_non_finite_input(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=3)
    _wait_status(base, lambda d: d["status"] == "completed")
    # Strict JSON clients refuse to even encode NaN; send raw tokens to
    # verify the server boundary rejects non-finite values with a clean 400
    # rather than a 500 from error-response serialization.
    r = httpx.post(base + "/api/predict",
                   content=b'{"x1": NaN, "x2": 1.0}',
                   headers={"Content-Type": "application/json"}, timeout=5)
    assert r.status_code == 400
    r = httpx.post(base + "/api/predict",
                   content=b'{"x1": 1.0, "x2": Infinity}',
                   headers={"Content-Type": "application/json"}, timeout=5)
    assert r.status_code == 400
    r = httpx.post(base + "/api/predict",
                   content=b'{"x1": "oops", "x2": 1.0}',
                   headers={"Content-Type": "application/json"}, timeout=5)
    assert r.status_code == 400
    r = httpx.post(base + "/api/predict", content=b'not-json',
                   headers={"Content-Type": "application/json"}, timeout=5)
    assert r.status_code == 400


def test_worker_predict_is_read_only_and_gen_neutral(cluster, tmp_path):
    base = cluster["coord"]
    # Slow rounds so the pause reliably lands between rounds; once paused,
    # state.json is stable and read-only-ness can be checked byte-for-byte.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=10)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 2)
    deadline = time.time() + 15
    while time.time() < deadline:
        if httpx.post(base + "/api/pause", timeout=5).status_code == 200:
            break
        time.sleep(0.3)
    st = _wait_status(base, lambda d: d["status"] == "paused")
    w1_port = cluster["w1"].port
    url = f"http://127.0.0.1:{w1_port}"
    state_path = cluster["paths"]["w1"] / "state.json"
    before = state_path.read_bytes()

    # Direct worker call: no generation field, so it also proves a restarted,
    # not-yet-prepared worker can score an accepted confirmed snapshot.
    r = httpx.post(url + "/predict", json={
        "task_id": st["task_id"], "round": st["confirmed_round"],
        "w": st["w"], "features": FEATURES}, timeout=5)
    r.raise_for_status()
    doc = r.json()
    assert doc["round"] == st["confirmed_round"]
    assert doc["probability"] == lrmath.predict_probability(FEATURES, st["w"])
    for _ in range(3):
        httpx.post(url + "/predict", json={
            "task_id": st["task_id"], "round": st["confirmed_round"],
            "w": st["w"], "features": FEATURES}, timeout=5)
    assert state_path.read_bytes() == before  # durable state never touched

    # Wrong task is rejected, and malformed feature payloads are 4xx.
    r = httpx.post(url + "/predict", json={
        "task_id": "stale", "round": 0, "w": st["w"],
        "features": FEATURES}, timeout=5)
    assert r.status_code == 410
    r = httpx.post(url + "/predict", json={
        "task_id": st["task_id"], "round": 0, "w": st["w"],
        "features": [1.0]}, timeout=5)
    assert r.status_code == 422


# ------------------------------------------------------- coordinator unit branches
def test_run_prediction_review_branches(monkeypatch):
    """Exercise the disagreement / version-mismatch paths that deterministic
    identical workers cannot produce on the wire."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    import coordinator

    snapshot = {"task_id": "t1", "confirmed_round": 4,
                "w": [0.0, 0.1, -0.2]}

    async def fake_disagree(worker_id, path, payload, tolerate=False):
        if worker_id == "worker1":
            return {"worker_id": "worker1", "task_id": "t1", "round": 4,
                    "probability": 0.3, "local_round": 4}
        return {"worker_id": "worker2", "task_id": "t1", "round": 4,
                "probability": 0.31, "local_round": 4}

    monkeypatch.setattr(coordinator, "worker_call", fake_disagree)
    res = asyncio.run(coordinator._run_prediction(snapshot, FEATURES))
    assert res["verdict"] == "needs_review"
    assert res["classification"] is None and res["probability"] is None
    assert "概率不一致" in res["reason"]

    async def fake_version(worker_id, path, payload, tolerate=False):
        if worker_id == "worker1":
            return {"worker_id": "worker1", "task_id": "t1", "round": 4,
                    "probability": 0.3, "local_round": 4}
        return {"worker_id": "worker2", "task_id": "t1", "round": 5,
                "probability": 0.3, "local_round": 5}

    monkeypatch.setattr(coordinator, "worker_call", fake_version)
    res = asyncio.run(coordinator._run_prediction(snapshot, FEATURES))
    assert res["verdict"] == "needs_review"
    assert "版本不符" in res["reason"]

    async def fake_unreachable(worker_id, path, payload, tolerate=False):
        return {"_error": f"{worker_id} connection refused"}

    monkeypatch.setattr(coordinator, "worker_call", fake_unreachable)
    res = asyncio.run(coordinator._run_prediction(snapshot, FEATURES))
    assert res["verdict"] == "needs_review"
    assert all(not s["ok"] for s in res["workers"].values())
