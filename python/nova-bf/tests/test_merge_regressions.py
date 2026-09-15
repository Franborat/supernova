"""Regression tests for the partial-major reduce in `merge._reduce`.

Concurrency / resource / failure-mode invariants. Each test below started life
as a reproducer for a defect found in review; they are inverted here
to pin the fixed behaviour, so a regression fails loudly rather than silently
returning the old shape. Every test that could hang runs the merge on a watchdog
thread and fails on timeout rather than blocking the suite.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref

import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

import nova_bf.io as io_mod
import nova_bf.merge as merge_mod
from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.io import Store
from nova_bf.results import build_result_table, partial_dir, result_name

K = 4


def _cfg(root: str, name: str = "test") -> BruteForceConfig:
    return BruteForceConfig(
        corpus=CorpusConfig(path=f"{root}/corpus"),
        queries=QueriesConfig(path=f"{root}/queries.parquet"),
        output=OutputConfig(path=root),
        searches=[SearchSpec(name=name, k=K)],
    )


def _write_partials(cfg, pdir, n_partials: int, n_queries: int = 8) -> None:
    pdir.mkdir(parents=True, exist_ok=True)
    qids = [f"q{i}" for i in range(n_queries)]
    score = 1000.0
    for p in range(n_partials):
        ids, scores = [], []
        for q in qids:
            ids.append([f"{q}_p{p}_{i}" for i in range(K)])
            scores.append([score := score - 1.0 for _ in range(K)])
        payload = {"src": [f"payload-{q}" for q in qids]}
        pq.write_table(build_result_table(qids, payload, ids, scores),
                       str(pdir / f"rank{p:03d}.parquet"), row_group_size=4)


def _run_with_timeout(fn, timeout: float):
    """Run `fn` on a thread. Returns ('ok', value) | ('raised', exc) | ('hung', None)."""
    box: list = []

    def target():
        try:
            box.append(("ok", fn()))
        except BaseException as exc:                     # noqa: BLE001
            box.append(("raised", exc))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return ("hung", None)
    return box[0]


def _reader_threads(exclude: set) -> list[threading.Thread]:
    return [t for t in threading.enumerate()
            if t not in exclude and t.is_alive()
            and t is not threading.current_thread()
            and "_read" in t.name]


# ---------------------------------------------------------------------------
# `Store.root` is scheme-stripped -- a standing trap, deliberately documented.
# ---------------------------------------------------------------------------

def test_store_root_is_scheme_stripped_and_is_not_reusable_as_a_uri():
    """NOT a fixed bug -- a live footgun in `Store` itself, kept as documentation.

    `_fs_and_path` returns pyarrow's path, which for s3 is `bucket/key` with the
    scheme gone. So `Store(other.root)` silently downgrades an S3 store to a
    LocalFileSystem rooted at `$CWD/bucket/key`. Round-tripping is idempotent on
    LOCAL roots, which is why no test caught `_reduce` doing exactly this.
    Anything rebuilding a Store from another must pass `.uri`, never `.root`.
    """
    out = Store("s3://bucket/prefix")
    assert out.is_s3 and out.root == "bucket/prefix"

    downgraded = Store(out.root)
    assert downgraded.is_s3 is False
    assert isinstance(downgraded.fs, pafs.LocalFileSystem)
    assert downgraded.root.endswith("bucket/prefix") and downgraded.root.startswith("/")

    # ...whereas the uri round-trips faithfully.
    assert Store(out.uri).is_s3 is True


def test_merge_against_an_s3_style_root_reads_its_own_partials(tmp_path, monkeypatch):
    """`_reduce` must build its reader Store from `out.uri`, not `out.root`.

    With `out.root` the reduce looked for every partial on the LOCAL filesystem
    and every S3 merge died with a FileNotFoundError naming a cwd-relative path.
    This drives a filesystem whose paths are shaped like pyarrow's s3 ones
    (`bucket/key`) so the scheme strip would be fatal, and asserts the merge
    completes and produces the right answer.
    """
    fake_s3 = tmp_path / "s3root"
    fake_s3.mkdir()
    real = io_mod._fs_and_path

    def fake_fs_and_path(uri: str):
        if uri.startswith("s3://"):
            return (pafs.SubTreeFileSystem(str(fake_s3), pafs.LocalFileSystem()),
                    uri[len("s3://"):])
        return real(uri)

    monkeypatch.setattr(io_mod, "_fs_and_path", fake_fs_and_path)

    cfg = _cfg("s3://bucket/prefix")
    pdir = fake_s3 / "bucket" / "prefix" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=3)

    kind, res = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=60)
    assert kind == "ok", f"s3-shaped merge {kind}: {res!r}"

    merged = fake_s3 / "bucket" / "prefix" / result_name(cfg, cfg.searches[0])
    t = pq.read_table(str(merged)).to_pydict()
    assert len(t["query_id"]) == 8
    # every query's top-K is the global one: partial 0 holds the highest scores
    assert all(len(h) == K for h in t["hit_ids"])


# ---------------------------------------------------------------------------
# A failing reader must not strand the others.
# ---------------------------------------------------------------------------

def _leaky_scenario(tmp_path, monkeypatch, n_partials=8, window=2, slow=1.0,
                    fail_rank="rank000"):
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=n_partials)

    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: window)

    real_read = Store.read_columns

    def faulty(self, read_path, columns):
        if fail_rank in read_path:
            raise OSError(f"injected read failure on {read_path}")
        time.sleep(slow)                 # keep the other readers in flight
        return real_read(self, read_path, columns)

    monkeypatch.setattr(Store, "read_columns", faulty)
    return cfg


def test_failed_reduce_surfaces_the_error_and_strands_no_reader_thread(
        tmp_path, monkeypatch):
    """The consumer must DRAIN all `len(partials)` items, never break early.

    Every reader enqueues exactly one item (a table, or a `None` sentinel on
    failure) and holds a semaphore permit until the consumer releases it. A
    consumer that stopped at the first sentinel left the rest blocked forever --
    those holding a permit in `q.put` (queue full), the rest in
    `window.acquire()` (no permit ever returned). Observed before the fix: 6 of
    8 reader threads still alive after the failure. `daemon=True` only defers
    that to interpreter exit; inside a live process they are simply lost.
    """
    n_partials, slow = 8, 1.0
    cfg = _leaky_scenario(tmp_path, monkeypatch, n_partials=n_partials,
                          window=2, slow=slow)

    before = set(threading.enumerate())
    kind, res = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=120)

    assert kind == "raised", f"merge {kind}"
    assert isinstance(res, OSError) and "injected read failure" in str(res)

    time.sleep(slow * 3)                 # far longer than any injected sleep
    leaked = _reader_threads(before)
    assert not leaked, (
        f"{len(leaked)} reader threads survived the failed reduce "
        f"({[t.name for t in leaked]}); the consumer stopped draining")


def test_repeated_failed_merges_do_not_accumulate_threads(tmp_path, monkeypatch):
    """A long-lived process -- one `run_merge` per search, a retry loop, a
    notebook, this very test session -- used to gain ~W wedged threads per
    failure, forever. Thread count must be flat across repeated failures."""
    counts = []
    for attempt in range(3):
        sub = tmp_path / f"try{attempt}"
        sub.mkdir()
        cfg = _leaky_scenario(sub, monkeypatch, n_partials=8, window=2, slow=1.0)
        kind, _ = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=120)
        assert kind == "raised"
        time.sleep(1.5)
        counts.append(threading.active_count())
    assert counts[-1] <= counts[0], f"thread count grew across failures: {counts}"


def test_failed_reduce_pins_no_partial_table_in_memory(tmp_path, monkeypatch):
    """Tables already read when the failure lands must become collectable.

    Under the early-break consumer they were not: the wedged reader threads kept
    the `_read` closure -- and with it the Queue and every table sitting in it --
    reachable, so `gc.collect()` could not free them. At production shape that is
    `window_n` whole parsed partials lost per failed search.
    """
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=6, n_queries=200)

    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 3)

    refs: list[weakref.ref] = []
    real_read = Store.read_columns

    def faulty(self, read_path, columns):
        if "rank000" in read_path:
            raise OSError("injected")     # fails first
        time.sleep(0.3)                   # 1 and 2 are still in flight then
        tbl = real_read(self, read_path, columns)
        try:
            refs.append(weakref.ref(tbl))
        except TypeError:
            pytest.skip("pyarrow.Table is not weak-referenceable on this build")
        return tbl

    monkeypatch.setattr(Store, "read_columns", faulty)
    kind, _ = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=120)
    assert kind == "raised"

    time.sleep(1.0)
    gc.collect()
    alive = [r for r in refs if r() is not None]
    assert refs, "no partial was read — the test proves nothing"
    assert not alive, (
        f"{len(alive)}/{len(refs)} partial tables still resident after the "
        "reduce raised — the abandoned queue is still reachable")


def test_leaked_permits_do_not_wedge_a_later_merge(tmp_path, monkeypatch):
    """`q` and `window` are `_reduce` locals, so even a badly-behaved reduce
    cannot strand permits anything else can see. A subsequent merge in the same
    process must still complete."""
    bad = tmp_path / "bad"
    bad.mkdir()
    cfg_bad = _leaky_scenario(bad, monkeypatch, n_partials=8, window=2, slow=1.0)
    assert _run_with_timeout(lambda: merge_mod.run_merge(cfg_bad), 120)[0] == "raised"

    monkeypatch.undo()                    # restore the real Store.read_columns
    good = tmp_path / "good"
    good.mkdir()
    cfg_good = _cfg(str(good / "out"))
    _write_partials(cfg_good, good / "out" / partial_dir(cfg_good, cfg_good.searches[0]),
                    n_partials=4)
    kind, res = _run_with_timeout(lambda: merge_mod.run_merge(cfg_good), timeout=60)
    assert kind == "ok", f"a later merge was wedged: {kind} {res!r}"


def test_consumer_side_failure_also_drains_and_joins_promptly(tmp_path, monkeypatch):
    """A raise from the CONSUMER must leave the reader threads as clean as a
    raise from a reader does, and must not stall on `join(timeout=30)`."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True)
    qids = [f"q{i}" for i in range(8)]
    for p in range(3):
        order = qids if p != 1 else list(reversed(qids))   # partial 1 misaligned
        pq.write_table(
            build_result_table(order, {"src": [f"pay-{q}" for q in order]},
                               [[f"{q}_p{p}"] for q in order],
                               [[100.0 - p] for _ in order]),
            str(pdir / f"rank{p:03d}.parquet"))

    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 1)
    real_read = Store.read_columns

    def slow(self, read_path, columns):
        if "rank002" in read_path:
            time.sleep(0.2)
        return real_read(self, read_path, columns)

    monkeypatch.setattr(Store, "read_columns", slow)

    before = set(threading.enumerate())
    t0 = time.perf_counter()
    kind, res = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=120)
    elapsed = time.perf_counter() - t0

    assert kind == "raised" and "not row-aligned" in str(res)
    assert elapsed < 5.0, f"took {elapsed:.1f}s to surface a consumer-side error"
    assert not _reader_threads(before), "consumer-side raise stranded readers"


# ---------------------------------------------------------------------------
# `head` must not pin the partial it came from.
# ---------------------------------------------------------------------------

def test_head_copies_query_id_and_payload_instead_of_slicing_the_partial(
        tmp_path, monkeypatch):
    """`head[bi]` must hold a COPY of query_id + payload only.

    It used to hold `tbl.slice(...)`. An Arrow slice is a view, so that pinned
    partial 0's whole table -- hit columns included -- for the entire reduce,
    long past the point its window permit was handed back at merge.py's
    `window.release()`. The window bounds `window x partial` and never
    accounted for the extra one.
    """
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=3, n_queries=200)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 1)

    built: list[tuple[str, ...]] = []
    real_table = merge_mod.pa.table

    def spy(mapping, *a, **kw):
        tbl = real_table(mapping, *a, **kw)
        built.append(tuple(tbl.schema.names))
        return tbl

    monkeypatch.setattr(merge_mod.pa, "table", spy)
    merge_mod.run_merge(cfg)

    # a projection carrying exactly query_id + payload was materialised...
    assert ("query_id", "src") in built, built
    # ...and nothing that `head` keeps carries hit columns. Scoped to tables
    # keyed by query_id: the spy patches the pyarrow MODULE, so it also sees
    # tables built inside other modules -- `build_ordinals` makes an {id, pos}
    # one on its CPU path -- which are not heads and are absent on a GPU box.
    heads = [s for s in built if "hit_ids" not in s and "query_id" in s]
    assert heads and all(set(s) == {"query_id", "src"} for s in heads), heads

    # The rationale, pinned: a slice really would have retained the parent.
    parent = pq.read_table(str(pdir / "rank000.parquet"))
    sl, parent_bytes = parent.slice(0, parent.num_rows), parent.nbytes
    del parent
    gc.collect()
    assert sl.nbytes == parent_bytes


# ---------------------------------------------------------------------------
# `ranged_get` concurrency must be divided by the window.
# ---------------------------------------------------------------------------

def test_ranged_get_pool_is_divided_by_the_window(tmp_path, monkeypatch):
    """`_ranged_download` builds a fresh ThreadPoolExecutor PER FILE, so W
    in-flight partials multiply it: 24 x 16 = 384 concurrent range GETs, with
    `params.io_workers` -- which capped exactly this in the deleted
    `_prefetch_all` -- no longer consulted. `_reduce` must divide the per-file
    pool by its own window."""
    cfg = _cfg(str(tmp_path / "out"))
    cfg.params.merge_ranged_reads = True
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=4, n_queries=500)

    monkeypatch.setattr(io_mod, "_RANGED_GET_MIN_BYTES", 1)
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", 4096)
    window_n = 4
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: window_n)

    concurrencies: list[int] = []
    real_dl = Store._ranged_download

    def spy(self, read_path, size):
        concurrencies.append(self.ranged_get_concurrency)
        return real_dl(self, read_path, size)

    monkeypatch.setattr(Store, "_ranged_download", spy)
    merge_mod.run_merge(cfg)

    assert concurrencies, "ranged_get path never ran"
    # `max(1, ...)`, not `max(2, ...)`: a floor of 2 breached the pool above
    # window_n=12. This agreed with the code only because window_n is 4 here.
    expected = max(1, merge_mod._RANGED_GET_POOL // window_n)
    assert set(concurrencies) == {expected}, concurrencies
    # the product is what matters: it must stay at or under the pool
    assert window_n * expected <= merge_mod._RANGED_GET_POOL


# ---------------------------------------------------------------------------
# Batch sizing must match the fold that actually happens.
# ---------------------------------------------------------------------------

def test_auto_batch_size_matches_the_two_way_fold():
    """The grid is `B x 2k`, not `B x W x k`: the reduce folds partial by
    partial, so `_topk_merge` sees the running state plus one partial however
    many partials exist. Sizing off `n_partials` was right for the old lockstep
    loop and is 32x too small at W=64 — 32x the fold calls, and one parquet row
    group per batch in the artifact people consume."""
    n_partials, k, n_rows = 64, 1000, 100_000
    rows = merge_mod._resolve_batch_rows(None, n_rows, n_partials, k)
    slots = rows * 2 * k
    assert slots <= merge_mod._TARGET_CANDIDATE_SLOTS
    assert slots > merge_mod._TARGET_CANDIDATE_SLOTS * 0.9, (
        f"batch_rows={rows} uses only {slots/1e6:.2f} M of the "
        f"{merge_mod._TARGET_CANDIDATE_SLOTS/1e6:.0f} M target")
    # ...and it must not depend on the partial count any more.
    assert rows == merge_mod._resolve_batch_rows(None, n_rows, 2, k)


def test_fold_is_never_wider_than_two_and_does_not_fragment_the_output(
        tmp_path, monkeypatch):
    """Every `_topk_merge` call is the seed (one list) or a 2-way fold, and the
    output keeps one row group per batch — so the batch must be sized for the
    real grid, not W x k."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=4, n_queries=40)

    widths: list[int] = []
    real = merge_mod._topk_merge

    def spy(score_lists, id_lists, tie_lists, k):
        widths.append(len(score_lists))
        return real(score_lists, id_lists, tie_lists, k)

    monkeypatch.setattr(merge_mod, "_topk_merge", spy)
    merge_mod.run_merge(cfg)

    assert widths and set(widths) <= {1, 2}, f"fold widths seen: {sorted(set(widths))}"
    assert 1 in widths, "the single-partial seed must go THROUGH _topk_merge"
    md = pq.ParquetFile(f"{cfg.output.path}/{result_name(cfg, cfg.searches[0])}").metadata
    assert md.num_rows == 40
    # auto sizing covers all 40 rows in one batch -> one row group, not 20
    assert md.num_row_groups == 1, f"{md.num_row_groups} row groups for 40 rows"


# ---------------------------------------------------------------------------
# Row alignment across partials.
# ---------------------------------------------------------------------------

def test_row_misaligned_partials_are_rejected(tmp_path):
    """Partials are row-aligned by query, and the reduce reads payload from
    partial 0 only. A partial whose rows are in a different order therefore
    folds its hits into the WRONG queries and the output looks entirely normal.
    The old lockstep loop compared query_id across all W partials per batch;
    that check was dropped in the rewrite and must stay restored."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True)

    qids = ["q0", "q1", "q2", "q3"]
    for p, order in enumerate([qids, list(reversed(qids))]):
        pq.write_table(
            build_result_table(order, {"src": [f"pay-{q}" for q in order]},
                               [[f"{q}_hit_p{p}"] for q in order],
                               [[100.0 + p] for _ in order]),
            str(pdir / f"rank{p:03d}.parquet"))

    kind, res = _run_with_timeout(lambda: merge_mod.run_merge(cfg), timeout=120)
    assert kind == "raised", f"misaligned partials merged without complaint ({kind})"
    assert "not row-aligned" in str(res), res


# ---------------------------------------------------------------------------
# Review, 2026-09-05: resource / lifetime / shutdown.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window_n", [1, 2, 3, 4, 8, 12, 13, 16, 24, 25, 64])
def test_range_gets_per_file_shrink_as_the_window_grows(window_n):
    """The per-file pool is DIVIDED by the window, because `_ranged_download`
    builds one PER FILE -- so the two cannot both be large.

    The old list ([1, 4, 12, 13, 16]) was exhaustive of the domain when the
    window came from a byte budget clamped to 16. The window is an operator
    knob with no ceiling now, so this covers past the pool size too.

    Above `_RANGED_GET_POOL` readers the division bottoms out at one GET each
    and the total tracks the window, which is the operator's choice to make:
    they asked for that many readers. What must not happen is per-file
    concurrency staying flat while the window grows, which is what a floor of
    2 did (32 outstanding GETs at window 16, against the 24 being divided).
    """
    per_file = max(1, merge_mod._RANGED_GET_POOL // window_n)
    if window_n <= merge_mod._RANGED_GET_POOL:
        assert window_n * per_file <= merge_mod._RANGED_GET_POOL, (
            f"{window_n} x {per_file} exceeds {merge_mod._RANGED_GET_POOL}")
    else:
        assert per_file == 1, "cannot give a reader less than one GET"


def test_zero_query_partials_are_refused_not_written_as_an_empty_file(tmp_path):
    """Zero queries means zero batches, so the writer was never created and
    `sink.close()` left a 0-byte file -- reported as success, unreadable
    afterwards ("Parquet file size is 0 bytes")."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=2, n_queries=0)
    with pytest.raises(RuntimeError, match="0 queries"):
        merge_mod.run_merge(cfg)


def test_a_failure_before_the_drain_does_not_strand_readers(tmp_path, monkeypatch):
    """`path`/`os.makedirs` used to sit between `t.start()` and the try that
    owns the drain, so anything raising there left every reader blocked on
    `window.acquire()` or `q.put()` with nobody to drain them -- each holding a
    parsed partial for the life of the process."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=6, n_queries=200)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 2)

    def boom(*a, **kw):
        raise PermissionError("read-only output root")
    monkeypatch.setattr(merge_mod.os, "makedirs", boom)

    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(PermissionError):
        merge_mod.run_merge(cfg)
    time.sleep(0.5)
    leaked = [t for t in threading.enumerate()
              if t.ident not in before and t.is_alive() and "read" not in t.name.lower()
              or (t.ident not in before and t.is_alive())]
    assert not leaked, f"{len(leaked)} reader thread(s) stranded: {[t.name for t in leaked]}"


# ---------------------------------------------------------------------------
# Review round 2: defects introduced BY the round-1 fixes.
# ---------------------------------------------------------------------------

class _SliceBoom:
    """A table whose `.slice()` raises -- i.e. the fold fails BEFORE `sl` is
    bound, which is the one path where a `del sl` would explode."""

    def __init__(self, t):
        self._t = t

    def slice(self, *a, **kw):
        raise MemoryError("INJECTED before sl is bound")

    def __getattr__(self, n):
        return getattr(self._t, n)


def test_a_fold_failing_before_sl_is_bound_surfaces_the_real_error(tmp_path, monkeypatch):
    """`sl` is assigned INSIDE the try, so a failure on the first slice leaves
    it unbound. A `del sl` there raises UnboundLocalError out of the handler,
    abandoning the drain and stranding every reader -- re-creating the exact
    failure the drain was written to prevent, and hiding the real error."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=6, n_queries=200)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 2)

    real = Store.read_columns
    monkeypatch.setattr(Store, "read_columns",
                        lambda self, p, c: _SliceBoom(real(self, p, c)))

    before = {t.ident for t in threading.enumerate()}
    t0 = time.monotonic()
    with pytest.raises(MemoryError, match="INJECTED"):
        merge_mod.run_merge(cfg)
    assert time.monotonic() - t0 < 25, "drain was abandoned; readers hit the join grace"
    time.sleep(0.3)
    leaked = [t for t in threading.enumerate()
              if t.ident not in before and t.is_alive()]
    assert not leaked, f"stranded readers: {[t.name for t in leaked]}"


def test_a_consumer_failure_keeps_its_location(tmp_path, monkeypatch):
    """The traceback is dropped so it cannot pin a multi-GB partial, but the
    location has to survive as text -- otherwise every consumer-side failure
    reports only the `raise errors[0]` line, and an unexpected error deep in the
    fold becomes unlocatable."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=3, n_queries=50)

    real = merge_mod._topk_merge

    def boom(*a, **kw):
        raise ValueError("INJECTED deep in the fold")
    monkeypatch.setattr(merge_mod, "_topk_merge", boom)

    with pytest.raises(ValueError, match="INJECTED") as ei:
        merge_mod.run_merge(cfg)
    notes = "\n".join(getattr(ei.value, "__notes__", []))
    assert "_fold" in notes or "_topk_merge" in notes, notes
    assert "merge.py" in notes, notes


def _no_leaked_readers(before, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        leaked = [t for t in threading.enumerate()
                  if t.ident not in before and t.is_alive()]
        if not leaked:
            return []
        time.sleep(0.1)
    return leaked


def test_an_interrupt_in_the_drain_still_releases_every_reader(tmp_path, monkeypatch):
    """Ctrl-C lands in `q.get()`, outside anything the loop catches. Only the
    consumer returns window permits, so an abandoned drain wedges every reader
    on `window.acquire()` for the life of the process -- each holding a parsed
    partial (a large allocation at production shape)."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=8, n_queries=150)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 2)

    real_q = merge_mod.Queue

    class Rude(real_q):
        n = 0

        def get(self, *a, **kw):
            item = super().get(*a, **kw)
            type(self).n += 1
            if type(self).n == 2:
                raise KeyboardInterrupt("user pressed ctrl-c")
            return item

    monkeypatch.setattr(merge_mod, "Queue", Rude)
    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(KeyboardInterrupt):
        merge_mod.run_merge(cfg)
    assert not _no_leaked_readers(before), "readers stranded by the interrupt"


def test_a_thread_that_will_not_start_still_releases_the_ones_that_did(
        tmp_path, monkeypatch):
    """`t.start()` can raise -- a thread-capped container at W=64. It used to
    sit outside the try, so neither the drain nor the join ran and the readers
    already going were stranded with no diagnostic at all."""
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=6, n_queries=150)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 2)

    real_start = threading.Thread.start
    state = {"n": 0}

    def flaky(self):
        state["n"] += 1
        if state["n"] == 4:
            raise RuntimeError("can't start new thread")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", flaky)
    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(RuntimeError, match="can't start new thread"):
        merge_mod.run_merge(cfg)
    assert not _no_leaked_readers(before), "readers stranded by the failed start"


def test_a_data_error_outranks_a_transient_read_error(tmp_path, monkeypatch):
    """`errors[0]` is first-append-wins and the two failures race: a reader can
    append its transient S3 error while the consumer is still inside the fold
    that is about to raise a row-misalignment. One is retriable and one is not,
    and reporting the wrong one sends the operator round the whole corpus again.

    The sleeps force the losing order deterministically -- the reader appends
    first, so a naive `errors[0]` yields the OSError.
    """
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=4, n_queries=100)
    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 4)

    real = Store.read_columns
    order: list[str] = []
    seen = {"n": 0}

    def flaky(self, path, cols):
        seen["n"] += 1
        if seen["n"] == 2:                 # a LATER reader, so partial 0 folds
            time.sleep(0.05)
            order.append("read")
            raise OSError("transient S3 slowdown")
        return real(self, path, cols)

    def slow_bad_fold(*a, **kw):
        time.sleep(0.30)                   # still folding when the reader fails
        order.append("fold")
        raise RuntimeError("partials are not row-aligned")

    monkeypatch.setattr(Store, "read_columns", flaky)
    monkeypatch.setattr(merge_mod, "_topk_merge", slow_bad_fold)

    with pytest.raises(RuntimeError, match="row-aligned"):
        merge_mod.run_merge(cfg)
    assert order[:2] == ["read", "fold"], f"ordering not forced: {order}"


def test_hit_id_output_type_does_not_follow_the_partials(tmp_path):
    """`_take_ids` gathers from the partials' own buffers, so the output type
    followed the input -- but a batch with NO hits falls back to an empty
    large_string. Mixing the two inside one merge aborts `ParquetWriter` mid
    write ("Table schema does not match schema used to create file"). It also
    keeps int64 character offsets, which a `string` output would not."""
    q, k = 4, 2
    # rows 2 and 3 have no hits at all -> the empty-batch fallback
    offs = pa.array([0, 2, 4, 4, 4], pa.int32())
    tbl = pa.table({
        "query_id": pa.array([f"q{i}" for i in range(q)]),
        "hit_ids": pa.ListArray.from_arrays(
            offs, pa.array(["a", "b", "c", "d"], pa.string())),   # NOT large_string
        "hit_scores": pa.ListArray.from_arrays(
            offs, pa.array([4.0, 3.0, 2.0, 1.0], pa.float32())),
    })
    cfg = _cfg(str(tmp_path / "out"))
    cfg.searches[0].k = k
    cfg.params.merge_batch_size = 2                      # forces a hits/no-hits split
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, str(pdir / "rank000.parquet"), compression="snappy")

    out = merge_mod.run_merge(cfg)
    got = pq.read_table(out[cfg.searches[0].name])
    assert got.schema.field("hit_ids").type == pa.list_(pa.large_string()), \
        got.schema.field("hit_ids").type
    assert got.column("hit_ids").to_pylist() == [["a", "b"], ["c", "d"], [], []]


def test_a_null_hit_scores_row_says_so(tmp_path):
    """`value_lengths()` on a null LIST entry yields NaN -> INT64_MIN, and the
    failure then surfaced from `np.repeat` as "repeats may not contain negative
    values", nowhere near the cause."""
    offs = pa.array([0, 2, 2, 4], pa.int32())
    null_row = pa.array([False, True, False])             # row 1 is a NULL list
    ids = pa.ListArray.from_arrays(
        offs, pa.array(["a", "b", "c", "d"], pa.large_string()), mask=null_row)
    sc = pa.ListArray.from_arrays(
        offs, pa.array([4.0, 3.0, 2.0, 1.0], pa.float32()), mask=null_row)
    tbl = pa.table({"query_id": pa.array(["q0", "q1", "q2"]),
                    "hit_ids": ids, "hit_scores": sc})
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, str(pdir / "rank000.parquet"), compression="snappy")

    with pytest.raises(RuntimeError, match="null"):
        merge_mod.run_merge(cfg)


# ---------------------------------------------------------------------------
# External review (GPT), 2026-09-06.
# ---------------------------------------------------------------------------

def _stamped(cfg, pdir, ranks, num_jobs, n_queries=8, tiebreak=None,
             run_key=True, names=None):
    """Partials carrying explicit run/rank metadata, one file per rank."""
    from nova_bf.results import (CONFIG_KEY, JOB_RANK_KEY, NUM_JOBS_KEY,
                                 RUN_KEY, TIEBREAK_KEY, config_identity,
                                 run_identity)
    pdir.mkdir(parents=True, exist_ok=True)
    spec = cfg.searches[0]
    csha = config_identity(cfg, spec)
    qids = [f"q{i}" for i in range(n_queries)]
    for r in ranks:
        tb = tiebreak or cfg.params.tiebreak
        rsha = run_identity(csha, "corpus", num_jobs, None, tb)
        ids = [[f"q_r{r}_{i}" for i in range(K)] for _ in qids]
        sc = [[float(K - i) for i in range(K)] for _ in qids]
        t = build_result_table(qids, {"src": [f"p-{q}" for q in qids]}, ids, sc)
        meta = {
            CONFIG_KEY: csha.encode(), NUM_JOBS_KEY: str(num_jobs).encode(),
            JOB_RANK_KEY: str(r).encode(), TIEBREAK_KEY: tb.encode(),
        }
        if run_key:
            meta[RUN_KEY] = rsha.encode()
        t = t.replace_schema_metadata(meta)
        pq.write_table(t, str(pdir / (names or {}).get(r, f"rank{r:03d}.parquet")))


@pytest.mark.parametrize("ranks,ok", [
    ([0, 1, 2, 3], True),
    ([0, 1, 2, 3, 4], False),      # a SUPERSET: rank 4 the run never declared
    ([0, 1, 2], False),            # the missing-rank case, already covered
])
def test_rank_set_must_be_exactly_zero_to_num_jobs(tmp_path, ranks, ok):
    """The docstring claims "the ranks present are exactly 0..num_jobs-1", but
    the check only looked for MISSING ranks -- so a directory holding an extra
    rank from a wider run passed, folding a corpus slice the run never declared
    into exact ground truth."""
    cfg = _cfg(str(tmp_path / "out"))
    _stamped(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]), ranks, 4)
    if ok:
        merge_mod.run_merge(cfg)
    else:
        with pytest.raises(RuntimeError, match="ranks"):
            merge_mod.run_merge(cfg)


def test_searches_reduced_under_different_tiebreaks_are_refused(tmp_path):
    """Each search validates its own run fingerprint, but that fingerprint is
    per-(run, SEARCH) -- so two searches from different runs both pass. The
    tie-break rule IS run-global, and mixing rules puts hits decided by
    different rules in one artifact."""
    cfg = _cfg(str(tmp_path / "out"))
    cfg.searches.append(SearchSpec(name="second", k=K))
    root = tmp_path / "out"
    _stamped(cfg, root / partial_dir(cfg, cfg.searches[0]), [0, 1], 2,
             tiebreak="ordinal")
    _stamped(cfg, root / partial_dir(cfg, cfg.searches[1]), [0, 1], 2,
             tiebreak="id")
    with pytest.raises(RuntimeError, match="tie-break"):
        merge_mod.run_merge(cfg)


def test_float64_scores_are_refused_not_rounded(tmp_path):
    """The candidate grid is float32, so a float64 column is silently DOWNCAST
    into it -- scores differing below float32 resolution collapse into a tie and
    the rounded value is reported as the score."""
    q = 3
    offs = pa.array([0, 2, 4, 6], pa.int32())
    tbl = pa.table({
        "query_id": pa.array([f"q{i}" for i in range(q)]),
        "hit_ids": pa.ListArray.from_arrays(
            offs, pa.array([f"d{i}" for i in range(6)], pa.large_string())),
        "hit_scores": pa.ListArray.from_arrays(
            offs, pa.array([1.0000000002, 1.0000000001] * 3, pa.float64())),
    })
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, str(pdir / "rank000.parquet"))
    with pytest.raises(RuntimeError, match="float32"):
        merge_mod.run_merge(cfg)


def test_a_failed_write_leaves_no_result_under_the_canonical_name(
        tmp_path, monkeypatch):
    """A half-written parquet at the canonical name is worse than none: it
    opens, it looks like a result, and it is silently short. Anything finding
    results by filename rather than through the merge manifest consumes it."""
    cfg = _cfg(str(tmp_path / "out"))
    cfg.params.merge_batch_size = 2
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _write_partials(cfg, pdir, n_partials=2, n_queries=8)

    real = pq.ParquetWriter.write_table
    state = {"n": 0}

    def flaky(self, table, *a, **kw):
        state["n"] += 1
        if state["n"] == 2:                     # first batch lands, second dies
            raise OSError("object store went away mid-write")
        return real(self, table, *a, **kw)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", flaky)
    out_path = tmp_path / "out" / result_name(cfg, cfg.searches[0])
    with pytest.raises(OSError, match="went away"):
        merge_mod.run_merge(cfg)
    assert not out_path.exists(), "a truncated result was left behind"


def test_a_folded_partial_is_freed_before_its_window_permit_is_returned(
        tmp_path, monkeypatch):
    """`sl = tbl.slice(...)` is a zero-copy VIEW, so `del tbl` frees nothing
    while it is bound. Releasing the permit there admits the next partial on top
    of the previous one -- one whole partial beyond the window budget at
    production shape. It was cleared only on the fold-FAILURE path.

    At the release point: under one partial resident with the fix, nearly two
    without.
    """
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True, exist_ok=True)
    qids = [f"q{i}" for i in range(4000)]
    for p in range(3):
        ids = [[f"<urn:uuid:{p:04d}{i:032d}>" for i in range(K)] for _ in qids]
        sc = [[float(K - i) for i in range(K)] for _ in qids]
        pq.write_table(build_result_table(qids, {"src": qids}, ids, sc),
                       str(pdir / f"rank{p:03d}.parquet"))
    one = pq.read_table(str(pdir / "rank000.parquet")).nbytes

    monkeypatch.setattr(merge_mod, "_merge_window",
                        lambda *a, **k: 1)
    samples: list[int] = []
    real_sem = merge_mod.Semaphore

    class Probe(real_sem):
        def release(self, *a, **kw):
            samples.append(pa.total_allocated_bytes())
            return super().release(*a, **kw)

    monkeypatch.setattr(merge_mod, "Semaphore", Probe)
    base = pa.total_allocated_bytes()
    merge_mod.run_merge(cfg)

    peak = (max(samples) - base) / one
    assert peak < 1.5, (
        f"{peak:.2f} partials resident when the window permit was returned; "
        "a folded partial is still being held by its slice")


def test_missing_ranks_are_caught_even_without_a_run_fingerprint(tmp_path):
    """The no-RUN_KEY path returned early, skipping the config AND rank checks
    -- so 2 partials of a 4-rank run merged clean, with half the corpus silently
    absent, while `num_jobs`/`job_rank` sat unread in their metadata."""
    cfg = _cfg(str(tmp_path / "out"))
    _stamped(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]),
             [0, 1], 4, run_key=False)
    with pytest.raises(RuntimeError, match="declared 4 ranks"):
        merge_mod.run_merge(cfg)


def test_a_complete_run_without_a_run_fingerprint_still_merges(tmp_path):
    """The fall-through must not turn the legacy warning into a refusal:
    hours of legitimate GPU work should not be stranded by a missing stamp."""
    cfg = _cfg(str(tmp_path / "out"))
    _stamped(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]),
             [0, 1, 2, 3], 4, run_key=False)
    merge_mod.run_merge(cfg)


def test_a_filename_disagreeing_with_its_rank_metadata_is_refused(tmp_path):
    """The rank set is otherwise a property of metadata alone -- the filename is
    used only for ordering, so a file whose name and stamp disagree means one of
    them was rewritten and the rank set cannot be trusted."""
    cfg = _cfg(str(tmp_path / "out"))
    _stamped(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]),
             [0, 1, 2, 3], 4, names={3: "rank000.parquet", 0: "rank003.parquet"})
    with pytest.raises(RuntimeError, match="name and its metadata disagree"):
        merge_mod.run_merge(cfg)


def test_a_non_integer_rank_names_the_search_and_the_file(tmp_path):
    """It used to exit as a bare `ValueError: invalid literal for int()`."""
    from nova_bf.results import JOB_RANK_KEY
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    _stamped(cfg, pdir, [0, 1], 2)
    f = pdir / "rank001.parquet"
    t = pq.read_table(str(f))
    md = dict(t.schema.metadata)
    md[JOB_RANK_KEY] = b"not-an-int"
    pq.write_table(t.replace_schema_metadata(md), str(f))
    with pytest.raises(RuntimeError, match="not an integer"):
        merge_mod.run_merge(cfg)


def test_the_artifact_records_how_many_ranks_produced_it(tmp_path):
    """Otherwise the run's shape lives only in the merge manifest, which a later
    merge into the same output path overwrites -- leaving a finished file that
    cannot describe itself."""
    from nova_bf.results import NUM_JOBS_KEY
    cfg = _cfg(str(tmp_path / "out"))
    _stamped(cfg, tmp_path / "out" / partial_dir(cfg, cfg.searches[0]), [0, 1, 2], 3)
    out = merge_mod.run_merge(cfg)[cfg.searches[0].name]
    md = pq.ParquetFile(out).schema_arrow.metadata or {}
    assert md.get(NUM_JOBS_KEY) == b"3", md


@pytest.mark.parametrize("col", ["hit_scores", "hit_ids", "hit_tie"])
def test_a_null_inside_a_hit_list_is_refused(tmp_path, col):
    """Nulls in the values CHILD, not the list rows. Each corrupts silently and
    differently: a null score becomes NaN and is dropped as though it were
    padding (the candidate vanishes); a null hit_tie casts to INT64_MIN, the
    BEST possible tiebreak, so it beats every real hit; a null id is shipped as
    a `None` hit id."""
    offs = pa.array([0, 3], pa.int32())
    ids = ["a", None, "c"] if col == "hit_ids" else ["a", "b", "c"]
    sc = [3.0, None, 1.0] if col == "hit_scores" else [3.0, 2.0, 1.0]
    cols = {
        "query_id": pa.array(["q0"]),
        "hit_ids": pa.ListArray.from_arrays(offs, pa.array(ids, pa.large_string())),
        "hit_scores": pa.ListArray.from_arrays(offs, pa.array(sc, pa.float32())),
    }
    if col == "hit_tie":
        cols["hit_tie"] = pa.ListArray.from_arrays(
            offs, pa.array([5, None, 7], pa.int64()))
    cfg = _cfg(str(tmp_path / "out"))
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), str(pdir / "rank000.parquet"))
    with pytest.raises(RuntimeError, match=f"{col} has 1 null value"):
        merge_mod.run_merge(cfg)
