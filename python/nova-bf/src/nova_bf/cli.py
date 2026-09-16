"""`nova bf <compute|merge>` — exec'd by the `nova` dispatcher as `nova-bf`."""

from __future__ import annotations

import logging

import click

from nova_bf.config import load_config


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("nova_bf").setLevel(logging.INFO)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Brute-force exact nearest-neighbor ground truth."""


@main.command()
@click.argument("config")
@click.option("--num-jobs", type=int, default=None, help="Total workers (enables distributed slicing).")
@click.option("--job-rank", type=int, default=None, help="This worker's rank; defaults to $SKYPILOT_JOB_RANK.")
@click.option("--io-workers", type=int, default=None, help="Override params.io_workers — concurrent corpus-file reader threads (for sweeping/tuning).")
@click.option("--io-thread-count", type=int, default=None, help="Override params.io_thread_count — pyarrow's global IO pool (true S3 fetch concurrency).")
@click.option("--cpu-thread-count", type=int, default=None, help="Override params.cpu_thread_count — the run's CPU width. Sizes pyarrow's global CPU pool (parquet decode + Arrow compute kernels), torch's intra-op pool, and the text-scan pool; the OTHER pool from --io-thread-count, which only fetches bytes. Lowering it to bound Arrow memory narrows the other two as well.")
@click.option("--max-files", type=int, default=None, help="Read only the first N corpus files of this slice. Benchmarking aid; output is PARTIAL.")
def compute(
    config: str,
    num_jobs: int | None,
    job_rank: int | None,
    io_workers: int | None,
    io_thread_count: int | None,
    cpu_thread_count: int | None,
    max_files: int | None,
) -> None:
    """Search the corpus and write per-query top-K (one worker's slice)."""
    _setup_logging()
    from nova_bf.compute import run_compute

    run_compute(
        load_config(config),
        num_jobs=num_jobs,
        job_rank=job_rank,
        io_workers=io_workers,
        io_thread_count=io_thread_count,
        cpu_thread_count=cpu_thread_count,
        max_files=max_files,
    )


@main.command()
@click.argument("config")
@click.option("--search", "searches", multiple=True,
              help="Reduce only these searches (repeatable). Every search is "
                   "still validated; one manifest is written per search.")
@click.option("--jobs", "-j", type=int, default=1, metavar="N",
              help="Reduce up to N searches CONCURRENTLY, as separate "
                   "processes.")
def merge(config: str, searches: tuple[str, ...], jobs: int) -> None:
    """Merge per-rank partial results into each search's top-K parquet."""
    _setup_logging()

    if jobs < 1:
        raise SystemExit(f"--jobs must be at least 1, got {jobs}")

    cfg = load_config(config)
    known = [s.name for s in cfg.searches]
    unknown = sorted(set(searches) - set(known))
    if unknown:
        # Validate before spawning workers so a typo cannot start partial work.
        raise SystemExit(
            f"--search named {unknown}, which this config does not define; "
            f"it has {sorted(known)}"
        )

    # Avoid concurrent writers when --search is repeated.
    want = list(dict.fromkeys(searches)) or known
    if jobs > 1 and len(want) > 1:
        raise SystemExit(_merge_fanout(config, want, jobs))

    from nova_bf.merge import run_merge

    run_merge(cfg, only=set(searches) or None)

def _merge_fanout(config: str, names: list[str], jobs: int) -> int:
    """Merge searches concurrently in separate child processes."""
    import signal
    import subprocess
    import sys
    import time

    if jobs < 1:
        raise ValueError("jobs must be at least 1")

    logger = logging.getLogger("nova_bf.merge")

    from nova_bf.merge import preflight_searches

    cfg = load_config(config)

    # Fail once in the parent before spawning children.
    try:
        preflight_searches(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.error("%s", exc)
        logger.error("refusing to fan out: this fails for every search")
        return 1

    running: dict[int, tuple[str, subprocess.Popen]] = {}
    pending = list(names)
    failed: list[str] = []

    t0 = time.perf_counter()
    logger.info("merging %d searches, %d at a time", len(names), jobs)

    # Convert SIGTERM into normal unwinding so `finally` can reap children.
    previous: dict[int, object] = {}

    def _on_term(signum, frame):  # noqa: ARG001
        raise SystemExit(1)

    try:
        previous[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _on_term)
    except ValueError:
        # Signal handlers can only be installed from the main thread.
        previous.clear()

    try:
        while pending or running:
            while pending and len(running) < jobs:
                name = pending.pop(0)
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "nova_bf.cli",
                        "merge",
                        config,
                        "--search",
                        name,
                    ]
                )
                running[proc.pid] = (name, proc)
                logger.info("  started %s (pid %d)", name, proc.pid)

            done = [
                pid
                for pid, (_, proc) in running.items()
                if proc.poll() is not None
            ]

            if not done:
                time.sleep(0.5)
                continue

            for pid in done:
                name, proc = running.pop(pid)
                if proc.returncode == 0:
                    logger.info("  %s finished", name)
                else:
                    failed.append(name)
                    logger.error(
                        "  %s FAILED (exit %d)", name, proc.returncode
                    )

    finally:
        # Prevent cleanup from being interrupted while children are reaped.
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                previous.setdefault(signum, signal.getsignal(signum))
                signal.signal(signum, signal.SIG_IGN)
            except ValueError:
                pass

        for name, proc in running.values():
            if proc.poll() is None:
                logger.error("  terminating %s (pid %d)", name, proc.pid)
                proc.terminate()

        deadline = time.monotonic() + 30

        for name, proc in running.values():
            try:
                proc.wait(timeout=max(0, deadline - time.monotonic()))
            except BaseException:  # noqa: BLE001
                logger.error("  killing %s (pid %d)", name, proc.pid)
                proc.kill()
                proc.wait()

        for signum, handler in previous.items():
            if handler is None:
                handler = signal.SIG_DFL
            try:
                signal.signal(signum, handler)
            except (ValueError, TypeError):
                pass

    logger.info(
        "merged %d search(es) in %.1fs",
        len(names),
        time.perf_counter() - t0,
    )

    if not failed:
        # Retire the legacy manifest only after successful child merges.
        from nova_bf.io import Store
        from nova_bf.merge import drop_legacy_manifest
        from nova_bf.results import result_name

        try:
            drop_legacy_manifest(
                Store(cfg.output.path),
                cfg,
                {
                    result_name(cfg, search)
                    for search in cfg.searches
                    if search.name in names
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "could not retire the legacy manifest (%r)", exc
            )

    if failed:
        ok = [name for name in names if name not in failed]
        logger.error(
            "%d of %d searches FAILED (%s): the output set is INCOMPLETE. "
            "Successful searches: %s. Any parquet for a failed search is from "
            "an EARLIER run -- check its timestamp. (The output stream opens "
            "only after every read, fold and provenance check has passed, so "
            "most failures never touch the previous result; one that dies "
            "mid-write removes what it was writing and leaves nothing.) "
            "Re-run failures with `nova-bf merge %s --search NAME`.",
            len(failed),
            len(names),
            ", ".join(sorted(failed)),
            ", ".join(sorted(ok)) or "none",
            config,
        )
        return 1

    return 0


if __name__ == "__main__":
    main()
