#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_run_scheduling_pipeline.py

One-command full scheduling pipeline.

Backend full JSON
-> preprocess
-> demand ML inference
-> PV ML inference
-> SAC/RL inference with transfer post-processing
-> final backend response JSON

All outputs are saved under:
  scheduling/output/YYYYMMDD_HHMMSS_<request_id>/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

SCHEDULING_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = SCHEDULING_DIR / "src"
DEFAULT_REQUEST_DIR = SCHEDULING_DIR / "runtime_requests"
DEFAULT_OUTPUT_ROOT = SCHEDULING_DIR / "output"


def parse_run_stamp(payload: dict) -> str:
    ts = payload.get("request_timestamp") or payload.get("timestamp")
    if ts:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("Asia/Seoul").tz_localize(None)
    else:
        t = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    return t.strftime("%Y%m%d_%H%M%S")


def safe_name(value: str) -> str:
    out = str(value).strip().replace("/", "_").replace("\\", "_").replace(":", "")
    return "".join(ch if ch.isalnum() or ch in ["-", "_", "."] else "_" for ch in out)


def run_cmd(cmd: list[str], stage_name: str, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage_name}.log"
    print(f"\n[RUN] {stage_name}")
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(f"[FAIL] {stage_name}. See log: {log_path}")
        raise SystemExit(proc.returncode)
    print(f"[DONE] {stage_name}. Log: {log_path}")


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, help="Backend full JSON path")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--no-strict", action="store_true", help="Pass --no-strict to preprocessing")
    args = parser.parse_args()

    input_json = Path(args.input_json).expanduser().resolve()
    if not input_json.exists():
        raise FileNotFoundError(f"input JSON not found: {input_json}")

    payload = load_payload(input_json)
    request_id = safe_name(payload.get("request_id", input_json.stem))
    run_stamp = parse_run_stamp(payload)
    run_dir = Path(args.output_root).expanduser().resolve() / f"{run_stamp}_{request_id}"

    pre_dir = run_dir / "00_preprocessed"
    demand_dir = run_dir / "01_demand_forecast"
    pv_dir = run_dir / "02_pv_forecast"
    rl_dir = run_dir / "03_rl_schedule"
    final_dir = run_dir / "04_final_response"
    log_dir = run_dir / "logs"

    for d in [pre_dir, demand_dir, pv_dir, rl_dir, final_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    copied_input = run_dir / "backend_request_original.json"
    shutil.copy2(input_json, copied_input)

    py = sys.executable

    preprocess_cmd = [
        py,
        str(SRC_DIR / "01_preprocess_backend_request.py"),
        "--input-json", str(copied_input),
        "--output-dir", str(pre_dir),
    ]
    if args.no_strict:
        preprocess_cmd.append("--no-strict")

    run_cmd(preprocess_cmd, "00_preprocess", log_dir)

    run_cmd([
        py,
        str(SRC_DIR / "02_predict_demand_scheduling.py"),
        "--input", str(pre_dir / "demand_backend_input.json"),
        "--output-dir", str(demand_dir),
    ], "01_demand_forecast", log_dir)

    run_cmd([
        py,
        str(SRC_DIR / "03_predict_pv_scheduling.py"),
        "--input", str(pre_dir / "pv_backend_input.json"),
        "--output-dir", str(pv_dir),
    ], "02_pv_forecast", log_dir)

    final_response = final_dir / "ai_schedule_response.json"
    final_schedule_csv = rl_dir / "sac_pipeline_schedule.csv"
    final_actions_csv = rl_dir / "sac_pipeline_actions.csv"
    final_metrics = rl_dir / "sac_pipeline_metrics.json"
    runtime_input_csv = rl_dir / "rl_runtime_input_for_sac.csv"

    run_cmd([
        py,
        str(SRC_DIR / "04_sac_inference_scheduling.py"),
        "--demand-csv", str(demand_dir / "ev_demand_prediction_v2_from_backend_json.csv"),
        "--pv-csv", str(pv_dir / "pv_prediction_v3_from_backend_json.csv"),
        "--runtime-json", str(pre_dir / "rl_runtime_request.json"),
        "--model-path", str(SCHEDULING_DIR / "models/rl/sac_ev_scheduler_v3_1m.zip"),
        "--runtime-input-csv", str(runtime_input_csv),
        "--output-json", str(final_response),
        "--output-csv", str(final_schedule_csv),
        "--output-action-csv", str(final_actions_csv),
        "--output-metrics-json", str(final_metrics),
    ], "03_rl_schedule", log_dir)

    summary = {
        "request_id": payload.get("request_id"),
        "run_dir": str(run_dir),
        "backend_request_original": str(copied_input),
        "preprocess_report": str(pre_dir / "preprocess_report.json"),
        "demand_csv": str(demand_dir / "ev_demand_prediction_v2_from_backend_json.csv"),
        "pv_csv": str(pv_dir / "pv_prediction_v3_from_backend_json.csv"),
        "rl_runtime_input_csv": str(runtime_input_csv),
        "final_response_json": str(final_response),
        "metrics_json": str(final_metrics),
    }
    with (run_dir / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ FULL SCHEDULING PIPELINE COMPLETED")
    print(f"RUN DIR      : {run_dir}")
    print(f"FINAL JSON   : {final_response}")
    print(f"SUMMARY      : {run_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
