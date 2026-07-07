from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import config
from common_io import scan_h5_files
from progress import append_error, eta_from_seconds, fmt_seconds, load_progress, log, mark_step, resource_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all revision analyses.")
    parser.add_argument("--resume", action="store_true", help="Skip completed steps recorded in result/logs/progress.json.")
    parser.add_argument("--force", action="store_true", help="Run all steps even if progress says completed.")
    parser.add_argument("--skip-scvi", action="store_true", help="Skip optional scVI step.")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs, dependencies and planned steps without running analyses.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a small end-to-end test.")
    parser.add_argument("--max-samples", type=int, default=2, help="Smoke-test sample limit.")
    parser.add_argument("--max-cells", type=int, default=3000, help="Smoke-test per-sample cell limit.")
    parser.add_argument("--profile", choices=["core"], default="core")
    return parser.parse_args()


def _dependency_status() -> list[dict]:
    required = ["scanpy", "anndata", "numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn", "h5py"]
    optional = ["harmonypy", "torch", "scvi", "pydeseq2", "psutil", "tabulate"]
    rows = []
    for name in required + optional:
        rows.append({"package": name, "required": name in required, "available": importlib.util.find_spec(name) is not None})
    return rows


def dry_run() -> int:
    log("DRY RUN: checking inputs and environment", also_file=False)
    meta = scan_h5_files()
    print(f"Input h5 files: {len(meta)}")
    if not meta.empty:
        print(f"Raw cells: {int(meta['cells'].sum()):,}")
        print(meta.groupby("study").agg(samples=("sample_id", "count"), cells=("cells", "sum")).to_string())
    print("\nDependencies:")
    ok = True
    for row in _dependency_status():
        status = "OK" if row["available"] else ("MISSING" if row["required"] else "optional-missing")
        print(f"  {row['package']:<12} {status}")
        if row["required"] and not row["available"]:
            ok = False
    print("\nRun order:")
    for i, (_, script, label, optional) in enumerate(config.STEPS, start=1):
        opt = " optional" if optional else ""
        print(f"  {i:02d}. {script} - {label}{opt}")
    print(f"\nResources: {resource_snapshot()}")
    return 0 if ok and not meta.empty else 1


def _completed(step_id: str) -> bool:
    state = load_progress()
    return state.get("steps", {}).get(step_id, {}).get("status") == "completed"


def _estimate_remaining(completed_durations: list[float], remaining_steps: int) -> float | None:
    if not completed_durations:
        return None
    return (sum(completed_durations) / len(completed_durations)) * remaining_steps


def run_step(index: int, total: int, step_id: str, script: str, label: str, optional: bool, args: argparse.Namespace, durations: list[float]) -> bool:
    if args.skip_scvi and step_id == "11_scvi_optional":
        log(f"[{index}/{total}] skipping optional scVI (--skip-scvi)")
        mark_step(step_id, "completed", skipped=True, reason="--skip-scvi")
        return True
    if args.resume and not args.force and _completed(step_id):
        log(f"[{index}/{total}] already completed, skipping: {label}")
        return True
    remaining_eta = _estimate_remaining(durations, total - index + 1)
    log("=" * 80)
    log(f"[{index}/{total}] START {label} ({script})")
    log(f"Overall progress: {(index - 1) / total:.1%}; estimated finish: {eta_from_seconds(remaining_eta)}; {resource_snapshot()}")
    mark_step(step_id, "running", label=label, script=script, started_at=time.time(), optional=optional)
    cmd = [sys.executable, str(config.CODE_DIR / script)]
    if args.smoke_test:
        cmd += ["--smoke-test", "--max-samples", str(args.max_samples), "--max-cells", str(args.max_cells)]
    if args.force:
        cmd.append("--force")
    if args.skip_scvi:
        cmd.append("--skip-scvi")
    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(config.CODE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        rc = proc.wait()
    except Exception as exc:
        rc = 1
        append_error(step_id, str(exc))
    duration = time.time() - start
    if rc == 0:
        durations.append(duration)
        mark_step(step_id, "completed", label=label, duration_seconds=duration, completed_at=time.time(), optional=optional)
        log(f"[{index}/{total}] DONE {label} in {fmt_seconds(duration)}")
        return True
    append_error(step_id, f"exit code {rc}")
    status = "failed_optional" if optional else "failed"
    mark_step(step_id, status, label=label, duration_seconds=duration, exit_code=rc, optional=optional)
    log(f"[{index}/{total}] {'OPTIONAL FAILED' if optional else 'FAILED'} {label} after {fmt_seconds(duration)}")
    return optional


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return dry_run()
    config.ensure_result_dirs()
    log("Starting revision analysis pipeline")
    log(f"Mode: profile={args.profile}, resume={args.resume}, force={args.force}, smoke_test={args.smoke_test}, skip_scvi={args.skip_scvi}")
    log(f"Resources: {resource_snapshot()}")
    steps = config.STEPS
    durations: list[float] = []
    started = time.time()
    for i, (step_id, script, label, optional) in enumerate(steps, start=1):
        ok = run_step(i, len(steps), step_id, script, label, optional, args, durations)
        if not ok:
            log(f"Pipeline stopped at required step: {label}")
            return 1
    log("=" * 80)
    log(f"Pipeline completed in {fmt_seconds(time.time() - started)}")
    log(f"Result directory: {config.RESULT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

