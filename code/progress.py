from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import config


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def fmt_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def eta_from_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    return (datetime.now() + timedelta(seconds=float(seconds))).strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, also_file: bool = True) -> None:
    text = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(text, flush=True)
    if also_file:
        config.ensure_result_dirs()
        with (config.LOG_DIR / "run.log").open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def resource_snapshot() -> str:
    parts = []
    try:
        import psutil

        vm = psutil.virtual_memory()
        parts.append(f"RAM {vm.used / 1024**3:.1f}/{vm.total / 1024**3:.1f}GB ({vm.percent:.0f}%)")
    except Exception:
        parts.append("RAM unknown")
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=3).strip()
        if out:
            name, used, total = [x.strip() for x in out.splitlines()[0].split(",")[:3]]
            parts.append(f"GPU {name}; memory {float(used)/1024:.1f}/{float(total)/1024:.1f}GB")
        else:
            parts.append("GPU unavailable")
    except Exception:
        parts.append("GPU unavailable")
    return " | ".join(parts)


def load_progress() -> dict:
    path = config.LOG_DIR / "progress.json"
    if not path.exists():
        return {"started_at": now_iso(), "steps": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"started_at": now_iso(), "steps": {}}


def save_progress(state: dict) -> None:
    config.ensure_result_dirs()
    path = config.LOG_DIR / "progress.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def mark_step(step_id: str, status: str, **fields) -> None:
    state = load_progress()
    state.setdefault("steps", {})
    record = state["steps"].setdefault(step_id, {})
    if status == "completed":
        record.pop("exit_code", None)
        record.pop("error", None)
    record.update({"status": status, "updated_at": now_iso(), **fields})
    save_progress(state)


def append_error(step_id: str, message: str) -> None:
    config.ensure_result_dirs()
    with (config.LOG_DIR / "errors.log").open("a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {step_id}: {message}\n")


class StepTimer:
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()

    def update(self, message: str) -> None:
        log(f"{self.name}: {message} | elapsed {fmt_seconds(time.time() - self.start)} | {resource_snapshot()}")

    def done(self) -> None:
        log(f"{self.name}: completed in {fmt_seconds(time.time() - self.start)}")
