"""Dense pinned H2D overlap and its observable transfer contract."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nova_bf import compute
from nova_bf.compute import DenseCorpusBatch


def test_dense_transfer_mode_defaults_to_overlap_and_validates(monkeypatch):
    monkeypatch.delenv("NOVA_BF_PINNED", raising=False)
    assert compute._dense_transfer_mode() == 2
    for value, expected in (("0", 0), ("2", 2)):
        monkeypatch.setenv("NOVA_BF_PINNED", value)
        assert compute._dense_transfer_mode() == expected
    for value in ("on", "-1", "1", "3"):
        monkeypatch.setenv("NOVA_BF_PINNED", value)
        with pytest.raises(ValueError, match="must be 0 .* or 2"):
            compute._dense_transfer_mode()


def test_dense_prefetch_keeps_transfer_seam_and_issues_next_first(monkeypatch):
    """Every slice crosses `transfer`, and i+1 is issued before i is scored."""
    calls = []
    stream = object()

    class Event:
        def record(self, got_stream):
            assert got_stream is stream

        def synchronize(self):
            pass

    class Slice:
        def __init__(self, index):
            self.index = index

        def wait_on(self, got_stream):
            assert got_stream is stream
            calls.append(("wait", self.index))

    class Batch:
        def prefetch(self, r0, r1, device):
            assert device == "cuda:7"
            calls.append(("prefetch", r0 // 4, r0, r1))

        def transfer(self, r0, r1, device):
            assert device == "cuda:7"
            index = r0 // 4
            calls.append(("transfer", index, r0, r1))
            return Slice(index)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    ranges = [(0, 4), (4, 8), (8, 10)]
    compute._run_dense_prefetch(
        Batch(), ranges, "cuda:7",
        lambda r0, r1, sl: calls.append(("score", sl.index, r0, r1)),
    )
    assert calls == [
        ("prefetch", 0, 0, 4),
        ("prefetch", 1, 4, 8),
        ("transfer", 0, 0, 4),
        ("wait", 0),
        ("score", 0, 0, 4),
        ("prefetch", 2, 8, 10),
        ("transfer", 1, 4, 8),
        ("wait", 1),
        ("score", 1, 4, 8),
        ("transfer", 2, 8, 10),
        ("wait", 2),
        ("score", 2, 8, 10),
    ]


def test_dense_prefetch_bounds_delayed_compute_across_batches(monkeypatch):
    """Fast DMA cannot leave an entire batch's device tensors awaiting compute.

    Model a GPU that makes no compute progress until the host waits on a
    recorded event. DMA is instant, so waiting on DMA alone cannot pass.
    """
    submitted = completed = allocated = peak = 0
    stream = object()

    class Event:
        def record(self, got_stream):
            assert got_stream is stream
            self.through = submitted

        def synchronize(self):
            nonlocal completed
            completed = max(completed, self.through)

    class Slice:
        def wait_on(self, got_stream):
            assert got_stream is stream  # device-side wait makes no host progress

    class Batch:
        def prefetch(self, r0, r1, device):
            nonlocal allocated, peak
            allocated += 1
            peak = max(peak, allocated - completed)
            assert allocated - completed <= 3, "unbounded device allocation backlog"

        def transfer(self, r0, r1, device):
            return Slice()

    def score(*args):
        nonlocal submitted
        submitted += 1
        assert submitted - completed <= 2, "compute submission is not throttled"

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    # Repeated batches also catch a final event left undrained on each return.
    for count in (128, 1, 37):
        compute._run_dense_prefetch(
            Batch(), [(i, i + 1) for i in range(count)], "cuda", score,
        )
        assert submitted == completed == allocated
    assert peak == 3, "the test must exercise concurrent compute and lookahead"


def test_transfer_configuration_uses_resolved_step_not_survivor_count():
    batch = DenseCorpusBatch(np.zeros((3, 5), dtype=np.float16))
    batch.configure_transfer(4096, 2)
    assert batch._pinned_rows == 4096
    assert batch._transfer_mode == 2


def test_copy_stream_slice_waits_before_recording_compute_ownership():
    calls = []

    class Ready:
        pass

    class Tensor:
        shape = (4, 3)

        def record_stream(self, stream):
            calls.append(("record", stream))

    class Stream:
        def wait_event(self, ready):
            calls.append(("wait", ready))

    ready, stream = Ready(), Stream()
    sl = compute.DenseBatchSlice(Tensor(), ready=ready)
    sl.wait_on(stream)
    assert calls == [("wait", ready), ("record", stream)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_pinned_overlap_is_bitwise_identical_across_ring_wraps(monkeypatch, dtype):
    """Exercise a ragged tail and repeated ring reuse with real CUDA streams."""
    from nova_bf.config import SearchSpec
    from nova_bf.tiebreak import sentinel_key

    rng = np.random.default_rng(7)
    corpus = np.ascontiguousarray(rng.standard_normal((35, 12)).astype(dtype))
    queries = np.ascontiguousarray(rng.standard_normal((7, 12)).astype(np.float32))
    spec = SearchSpec(name="dense", vector_type="dense", metric="cosine", k=5)

    # Keep this test about transfer correctness. Kernel and two-pass parity have
    # their own suites, and allowing either here would make a failure ambiguous.
    monkeypatch.setenv("NOVA_BF_NO_TOPK_KERNEL", "1")
    monkeypatch.setenv("NOVA_BF_NO_FOLD_KERNEL", "1")

    def run(mode):
        monkeypatch.setenv("NOVA_BF_PINNED", str(mode))
        q = torch.tensor(queries, dtype=torch.float32, device="cuda")
        qn = q.norm(dim=1).clamp_min(1e-12)
        top_key = [sentinel_key((len(queries), spec.k), "cuda")]
        top_enc = [torch.zeros(
            (len(queries), spec.k), dtype=torch.int64, device="cuda"
        )]
        threshold = [sentinel_key((len(queries),), "cuda")]
        batch = DenseCorpusBatch(corpus)
        compute._process_batch_group(
            batch, [0], [spec], [q], [qn], top_key, top_enc, threshold,
            4, 0, "cuda", None,
            lambda _m, rows, _true_rows, _cache: (rows, None, None),
            [None], [None], [qn.contiguous()], two_pass=False,
        )
        torch.cuda.synchronize()
        return (
            top_key[0].cpu().numpy().copy(),
            top_enc[0].cpu().numpy().copy(),
            batch,
        )

    ref_key, ref_enc, _ = run(0)
    got_key, got_enc, batch = run(2)
    np.testing.assert_array_equal(got_key, ref_key)
    np.testing.assert_array_equal(got_enc, ref_enc)
    assert len(batch._ring) == 2
    assert {slot.shape[0] for slot in batch._ring} == {4}
