"""Correctness tests for the brute-force merge phase.

Covers the streaming lockstep reduce over row-aligned partials:
  - the merged top-K equals the global top-K over the union of all partials,
  - payload is carried through and queries with fewer than k total candidates
    keep only their real hits (variable-length output, no -inf padding), and
  - partials written as MULTIPLE row groups (what a large partial becomes) merge
    correctly — the regression for the `to_pylist` "Nested data conversions"
    crash the old merge hit at 1M queries.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nova_bf.config import (
    BruteForceConfig,
    CorpusConfig,
    OutputConfig,
    QueriesConfig,
    SearchSpec,
)
from nova_bf.merge import run_merge
from nova_bf.results import build_result_table, partial_dir, result_name

Q, W, K = 5, 3, 4


def _make_cfg(tmp) -> BruteForceConfig:
    return BruteForceConfig(
        corpus=CorpusConfig(path=str(tmp / "corpus")),
        queries=QueriesConfig(path=str(tmp / "queries.parquet")),
        output=OutputConfig(path=str(tmp / "out")),
        searches=[SearchSpec(name="test", k=K)],
    )


@pytest.fixture
def scenario(tmp_path):
    """W row-aligned partials + the reference global top-K per query."""
    rng = np.random.default_rng(0)
    qids = [f"q{i}" for i in range(Q)]
    # per (query) accumulate every candidate across partials for the reference
    all_cands: dict[str, list[tuple[float, str]]] = {q: [] for q in qids}
    # per partial: hit_ids / hit_scores lists aligned to qids
    partials: list[tuple[list[list[str]], list[list[float]]]] = []
    score = 100.0
    for p in range(W):
        p_ids, p_scores = [], []
        for q in qids:
            # query "q0" gets only 1 candidate per partial → <K total (variable len);
            # others get a random 1..K so several queries exceed K total.
            n = 1 if q == "q0" else int(rng.integers(1, K + 1))
            ids = [f"{q}_p{p}_{i}" for i in range(n)]
            scores = [score := score - 1.0 for _ in range(n)]  # globally unique, no ties
            # a partial's own list is already sorted desc (as compute emits it)
            order = np.argsort(-np.array(scores))
            ids = [ids[j] for j in order]
            scores = [scores[j] for j in order]
            p_ids.append(ids)
            p_scores.append(scores)
            all_cands[q].extend(zip(scores, ids))
        partials.append((p_ids, p_scores))

    reference = {}
    for q in qids:
        top = sorted(all_cands[q], reverse=True)[:K]
        reference[q] = ([h for _, h in top], [s for s, _ in top])

    cfg = _make_cfg(tmp_path)
    pdir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    pdir.mkdir(parents=True)
    for p, (p_ids, p_scores) in enumerate(partials):
        payload = {"src": [f"payload-{q}" for q in qids]}  # identical across partials
        table = build_result_table(qids, payload, p_ids, p_scores)
        # row_group_size < Q forces MULTIPLE row groups → multi-chunk nested columns
        # on read: the exact shape that crashed the old to_pylist-based merge.
        pq.write_table(table, str(pdir / f"rank{p:03d}.parquet"), row_group_size=2)

    return cfg, qids, reference


def _read_result(cfg) -> dict[str, tuple[list[str], list[float], str]]:
    t = pq.read_table(f"{cfg.output.path}/{result_name(cfg, cfg.searches[0])}").to_pydict()
    return {
        q: (hi, hs, src)
        for q, hi, hs, src in zip(t["query_id"], t["hit_ids"], t["hit_scores"], t["src"])
    }


def test_merge_matches_global_topk(scenario):
    cfg, qids, reference = scenario
    run_merge(cfg)
    got = _read_result(cfg)
    assert sorted(got) == sorted(qids)
    for q in qids:
        hi, hs, src = got[q]
        ref_ids, ref_scores = reference[q]
        assert hi == ref_ids  # identical hit ids, identical (score-desc) order
        assert np.allclose(hs, ref_scores)
        assert src == f"payload-{q}"  # payload carried from partial 0


def test_short_query_keeps_only_real_hits(scenario):
    """q0 has 1 candidate per partial (W total < K) → no -inf padding leaks out."""
    cfg, _, reference = scenario
    run_merge(cfg)
    hi, hs, _ = _read_result(cfg)["q0"]
    assert len(hi) == W < K
    assert hi == reference["q0"][0]


def test_explicit_batch_size_is_invariant(scenario):
    """A tiny merge_batch_size (many batches) gives the same result as one batch."""
    cfg, qids, reference = scenario
    cfg.params.merge_batch_size = 2  # < Q → several lockstep batches
    run_merge(cfg)
    got = _read_result(cfg)
    for q in qids:
        assert got[q][0] == reference[q][0]
        assert np.allclose(got[q][1], reference[q][1])


def test_reduce_bounds_how_many_partials_are_resident(scenario, monkeypatch, tmp_path):
    """The reduce must hold at most `merge_window` partials at once.

    This is the property the whole partial-major rewrite exists for. The old
    shape opened ALL W partials and read the same query batch from each in
    lockstep; because parquet's smallest read unit is the row group, that cost
    W x row-group, not W x batch -- far past host memory for a sharded dense merge, which is
    what actually OOMed. Asserting the fold's ORDER would be wrong (the reduce
    is commutative on purpose); the invariant worth pinning is the ceiling on
    concurrent readers.
    """
    import nova_bf.merge as merge_mod

    cfg, qids, reference = scenario
    cfg.params.merge_ranged_reads = True          # -> Store(ranged_get=True)

    live = 0
    peak = 0
    real_read = merge_mod.Store.read_columns

    def counting_read(self, read_path, columns):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        try:
            return real_read(self, read_path, columns)
        finally:
            live -= 1

    monkeypatch.setattr(merge_mod.Store, "read_columns", counting_read)
    merge_mod.run_merge(cfg)

    # NOT `_merge_window(...)` -- that makes the bound self-referential, so a
    # mutant returning `n_partials` satisfied `peak <= 10**6` vacuously.
    want = cfg.params.merge_window
    assert peak <= want, (
        f"{peak} partials were resident at once; the window is {want}. "
        "An unbounded reduce is what OOMed at scale."
    )
    assert peak >= 1, "no partial was ever read — the test proves nothing"

    # ...and the answer is still the global top-K, folded partial-by-partial.
    got = _read_result(cfg)
    assert sorted(got) == sorted(qids)
    for q in qids:
        hi, hs, src = got[q]
        ref_ids, ref_scores = reference[q]
        assert hi == ref_ids
        assert np.allclose(hs, ref_scores)
        assert src == f"payload-{q}"
def test_mismatched_partial_counts_across_searches_raises(scenario, tmp_path):
    """Every search in one `compute` run is written by the same set of ranks,
    so a mismatched partial count between two searches means some rank died
    partway through writing its per-search outputs — this must raise loudly
    at merge time instead of silently merging the short search from fewer
    ranks than it actually had."""
    cfg, qids, reference = scenario

    second = SearchSpec(name="test2", k=K)
    cfg.searches = [*cfg.searches, second]
    src_dir = tmp_path / "out" / partial_dir(cfg, cfg.searches[0])
    dst_dir = tmp_path / "out" / partial_dir(cfg, second)
    dst_dir.mkdir(parents=True)
    # Copy only W-1 of the W partials — simulates a rank that wrote the first
    # search's partial but died before writing this second search's.
    for f in sorted(src_dir.iterdir())[:-1]:
        (dst_dir / f.name).write_bytes(f.read_bytes())

    with pytest.raises(RuntimeError, match="mismatched partial counts"):
        run_merge(cfg)


def test_merge_window_is_exactly_what_the_operator_set(monkeypatch):
    """A plain number, clamped only by how many partials there are to read.

    This replaced a derivation from parquet metadata. That estimate was
    measured 3.3x HIGH on real 32-hex ids -- collapsing the window to one
    reader and serialising a 10B merge -- and 0.11x LOW on a column whose
    lexicographic bounds are short but whose interior values are long. No
    correction fixed both directions, and a wrong guess that looks
    authoritative is worse than a number the operator chose.
    """
    import types
    import nova_bf.merge as m

    def _c(window):
        return types.SimpleNamespace(params=types.SimpleNamespace(merge_window=window))

    monkeypatch.delenv("NOVA_BF_MERGE_WINDOW", raising=False)
    assert m._merge_window(_c(5), 64) == 5, "no scaling, no budget, no surprise"
    assert m._merge_window(_c(64), 1000) == 64, (
        "no ceiling: a window too large for the box is the operator's call")
    assert m._merge_window(_c(1), 64) == 1
    assert m._merge_window(_c(64), 3) == 3, "clamped to the partials present"
    assert m._merge_window(_c(2), 1) == 1

    # The default applies when nothing is set, and gives read/fold overlap.
    assert m._merge_window(_c(None), 64) == m._MERGE_WINDOW_DEFAULT
    assert m._MERGE_WINDOW_DEFAULT >= 2, (
        "a default of 1 removes read/fold overlap entirely")

    # Env wins over config; an unusable env keeps the COMMITTED value rather
    # than dropping to the default an operator set it precisely to escape.
    monkeypatch.setenv("NOVA_BF_MERGE_WINDOW", "7")
    assert m._merge_window(_c(4), 64) == 7
    for bad in ("0", "-3", "2.5", "two", "  "):
        monkeypatch.setenv("NOVA_BF_MERGE_WINDOW", bad)
        assert m._merge_window(_c(4), 64) == 4, f"env={bad!r} discarded the config"
