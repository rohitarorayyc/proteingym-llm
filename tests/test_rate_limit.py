import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src import run
from src.rate_limit import RollingTokenLimiter


class FakeClock:
    def __init__(self) -> None:
        self._value = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


class SignalingCondition(threading.Condition):
    def __init__(self) -> None:
        super().__init__()
        self.wait_started = threading.Event()

    def wait(self, timeout=None):
        self.wait_started.set()
        return super().wait(timeout)


def _wake(condition: threading.Condition) -> None:
    with condition:
        condition.notify_all()


def test_rolling_limiter_gates_threads_and_expires_reservations_in_sequence():
    clock = FakeClock()
    condition = SignalingCondition()
    limiter = RollingTokenLimiter(100, _clock=clock, _condition=condition)

    limiter.reserve(60)
    clock.advance(10)
    limiter.reserve(40)

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_waiter = executor.submit(limiter.reserve, 50)
        assert condition.wait_started.wait(timeout=1)
        assert not first_waiter.done()

        clock.advance(50)
        _wake(condition)
        first_waiter.result(timeout=1)

        with condition:
            assert limiter._reserved_tokens == 90

        condition.wait_started.clear()
        second_waiter = executor.submit(limiter.reserve, 11)
        assert condition.wait_started.wait(timeout=1)
        assert not second_waiter.done()

        clock.advance(10)
        _wake(condition)
        second_waiter.result(timeout=1)

        with condition:
            assert limiter._reserved_tokens == 61


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "100"])
def test_rolling_limiter_requires_a_positive_non_bool_integer(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        RollingTokenLimiter(invalid)


def test_rolling_limiter_rejects_one_request_larger_than_its_budget():
    limiter = RollingTokenLimiter(100)
    with pytest.raises(ValueError, match="request reserves 101 tokens"):
        limiter.reserve(101)


def test_run_assay_reserves_prompt_plus_output_immediately_before_chat_and_skips_dry_run(
    tmp_path, monkeypatch
):
    spec = {
        "provider": "openai-compatible",
        "api_style": "responses",
        "model_id": "test-model",
        "api_key_env": "LAB_API_KEY",
        "base_url_env": "LAB_BASE_URL",
        "reasoning": "max",
        "send_reasoning": True,
        "max_tokens": 128_000,
        "ctx": 1_000_000,
    }
    metadata = {
        "assay": {
            "reference_sequence": "WT",
            "fitness_description": "canonical description",
        }
    }
    data_bundle = {
        "bundle_version": "test-v1",
        "manifest_sha256": "1" * 64,
        "selection": {"sizes": [50], "seeds": [1]},
    }
    subset = [("v1", "A", 1.0), ("v2", "B", 0.0)]
    events = []

    class SpyLimiter:
        def reserve(self, tokens):
            events.append(("reserve", tokens))

    monkeypatch.setattr(run, "shared_subset", lambda *_args: subset)
    monkeypatch.setattr(
        run.client,
        "_env",
        lambda: {
            "LAB_API_KEY": "test-key",
            "LAB_BASE_URL": "https://inference.test/v1",
        },
    )

    def fake_chat(*_args, **_kwargs):
        events.append(("chat", None))
        return {
            "text": '{"ranking":["M01","M02"]}',
            "response_model_id": "test-model",
            "provider_response": {"id": "response"},
            "status": "completed",
            "error": None,
        }

    monkeypatch.setattr(run.client, "chat", fake_chat)
    live = run.run_assay(
        "test-model",
        spec,
        50,
        1,
        "assay",
        metadata,
        tmp_path / "live-results",
        tmp_path / "live-attempts",
        data_bundle=data_bundle,
        token_limiter=SpyLimiter(),
    )

    assert events == [
        ("reserve", live["prompt_tokens_estimate"] + spec["max_tokens"]),
        ("chat", None),
    ]

    class FailIfCalledLimiter:
        def reserve(self, _tokens):
            raise AssertionError("dry-run tried to reserve TPM")

    dry = run.run_assay(
        "test-model",
        spec,
        50,
        1,
        "assay",
        metadata,
        tmp_path / "dry-results",
        tmp_path / "dry-attempts",
        data_bundle=data_bundle,
        dry_run=True,
        token_limiter=FailIfCalledLimiter(),
    )
    assert dry["dry_run"] is True


def test_main_uses_one_limiter_across_sequential_seeds(tmp_path, monkeypatch):
    data_bundle = {
        "bundle_version": "test-v1",
        "manifest_sha256": "1" * 64,
        "selection": {"sizes": [50], "seeds": [1, 2]},
    }
    spec = {
        "provider": "openai-compatible",
        "api_style": "responses",
        "model_id": "test-model",
        "api_key_env": "LAB_API_KEY",
        "base_url_env": "LAB_BASE_URL",
        "reasoning": "max",
        "send_reasoning": True,
        "max_tokens": 128_000,
        "ctx": 1_000_000,
        "display_name": "Test model",
    }
    created = []
    seen = []

    class SentinelLimiter:
        def __init__(self, budget):
            self.budget = budget
            created.append(self)

    def fake_run_assay(*args, token_limiter=None, **_kwargs):
        seen.append((args[3], token_limiter))
        return {"dry_run": True, "error": None, "truncated": False, "overflow": False}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pgllm-run",
            "--registry",
            "models.json",
            "--models",
            "test-model",
            "--assays",
            "assay",
            "--sizes",
            "50",
            "--seeds",
            "1",
            "2",
            "--concurrency",
            "1",
            "--tokens-per-minute",
            "100",
            "--dry-run",
        ],
    )
    monkeypatch.setattr(run, "load_model_registry", lambda _path: {"test-model": spec})
    monkeypatch.setattr(run, "condition_record", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run, "authenticate_data_bundle", lambda: data_bundle)
    monkeypatch.setattr(run, "ensure_run_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run, "load_assay_meta", lambda: {"assay": {}})
    monkeypatch.setattr(run, "RollingTokenLimiter", SentinelLimiter)
    monkeypatch.setattr(run, "run_assay", fake_run_assay)
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run, "ATTEMPTS", tmp_path / "attempts")

    assert run.main() == 0
    assert len(created) == 1
    assert seen == [(1, created[0]), (2, created[0])]
