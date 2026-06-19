#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAC inference pipeline for EV multi-station day-ahead scheduling.

This script consumes:
1. Demand ML output CSV
   - ev-demand-forecast-v2/outputs/ev_demand_prediction_v2_from_backend_json.csv

2. PV ML output CSV
   - pv-dayahead-v2/outputs/pv_prediction_v3_from_backend_json.csv

3. Current station/runtime state JSON
   - station SoC
   - ESS constraints
   - grid limit
   - transfer constraints
   - ToU price

Then:
- builds SAC runtime input table
- runs trained SAC v3-1M model
- applies rule-based transfer post-processing
- writes backend-style schedule JSON

Final model:
- models/sac_ev_scheduler_v3_1m.zip

Response includes:
- forecast_results
  - pv_day_ahead_forecast
  - demand_day_ahead_forecast
  - demand_cluster_forecast
- station_day_ahead_schedule
  - ESS schedule
  - grid usage
  - PV/load forecast values
  - expected SoC
  - transfer schedule
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from stable_baselines3 import SAC

from evcs_env import MultiStationEVSchedulerEnv


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEAMPLAY_DIR = PROJECT_DIR.parent

# 기존 개별 모델 출력 경로는 참고용으로만 남김.
# OLD_DEFAULT_DEMAND_CSV = TEAMPLAY_DIR / "ev-demand-forecast-v2/outputs/ev_demand_prediction_v2_from_backend_json.csv"
# OLD_DEFAULT_PV_CSV = TEAMPLAY_DIR / "pv-dayahead-v2/outputs/pv_prediction_v3_from_backend_json.csv"
DEFAULT_DEMAND_CSV = PROJECT_DIR / "output/manual_test/demand/ev_demand_prediction_v2_from_backend_json.csv"
DEFAULT_PV_CSV = PROJECT_DIR / "output/manual_test/pv/pv_prediction_v3_from_backend_json.csv"

DEFAULT_MODEL_PATH = PROJECT_DIR / "models/rl/sac_ev_scheduler_v3_1m.zip"
DEFAULT_RUNTIME_JSON = PROJECT_DIR / "output/manual_test/preprocessed/rl_runtime_request.json"

DEFAULT_RUNTIME_INPUT_CSV = PROJECT_DIR / "output/manual_test/rl/rl_runtime_input_for_sac.csv"

# 기존 저장 경로는 개별 RL 테스트 때 참고용으로만 남김.
# OLD_DEFAULT_OUTPUT_JSON = PROJECT_DIR / "outputs/sac_pipeline_schedule_response.json"
# OLD_DEFAULT_OUTPUT_CSV = PROJECT_DIR / "outputs/sac_pipeline_schedule.csv"
# OLD_DEFAULT_OUTPUT_ACTION_CSV = PROJECT_DIR / "outputs/sac_pipeline_actions.csv"
# OLD_DEFAULT_OUTPUT_METRICS_JSON = PROJECT_DIR / "outputs/sac_pipeline_metrics.json"
DEFAULT_OUTPUT_JSON = PROJECT_DIR / "output/manual_test/rl/sac_pipeline_schedule_response.json"
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "output/manual_test/rl/sac_pipeline_schedule.csv"
DEFAULT_OUTPUT_ACTION_CSV = PROJECT_DIR / "output/manual_test/rl/sac_pipeline_actions.csv"
DEFAULT_OUTPUT_METRICS_JSON = PROJECT_DIR / "output/manual_test/rl/sac_pipeline_metrics.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DEFAULT_TOU_PRICE = {
    0: 83.1,
    1: 83.1,
    2: 83.1,
    3: 83.1,
    4: 83.1,
    5: 83.1,
    6: 83.1,
    7: 83.1,
    8: 140.0,
    9: 140.0,
    10: 140.0,
    11: 270.8,
    12: 140.0,
    13: 270.8,
    14: 270.8,
    15: 270.8,
    16: 270.8,
    17: 270.8,
    18: 140.0,
    19: 140.0,
    20: 140.0,
    21: 140.0,
    22: 83.1,
    23: 83.1,
}


DEFAULT_ENV_PARAMS = {
    "soc_min": 0.1,
    "soc_max": 0.9,
    "initial_soc": 0.5,

    "ess_capacity_kwh": 100.0,
    "ess_max_charge_kw": 50.0,
    "ess_max_discharge_kw": 50.0,
    "ess_charge_efficiency": 0.95,
    "ess_discharge_efficiency": 0.95,

    "grid_limit_kw": 200.0,

    "transfer_enabled": True,
    "transfer_capacity_kw": 30.0,
    "transfer_loss_rate": 0.03,
}


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def load_json_if_exists(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_tou_price(runtime: dict[str, Any]) -> dict[int, float]:
    """
    Accepts either:
    - runtime["tou_price_krw_per_kwh"] as dict {"0": 83.1, ...}
    - runtime["tou_price_krw_per_kwh"] as list [83.1, ...]
    - runtime["cluster_state"]["tou_price_krw_per_kwh"]

    If missing, uses DEFAULT_TOU_PRICE.
    """
    raw = runtime.get("tou_price_krw_per_kwh")
    if raw is None:
        raw = get_nested(runtime, ["cluster_state", "tou_price_krw_per_kwh"])

    if raw is None:
        return DEFAULT_TOU_PRICE.copy()

    if isinstance(raw, list):
        if len(raw) != 24:
            raise ValueError("tou_price_krw_per_kwh list must have 24 values.")
        return {i: float(raw[i]) for i in range(24)}

    if isinstance(raw, dict):
        out = {int(k): float(v) for k, v in raw.items()}
        missing = [h for h in range(24) if h not in out]
        if missing:
            raise ValueError(f"TOU price missing hours: {missing}")
        return out

    raise ValueError("Invalid tou_price_krw_per_kwh format.")


def build_station_runtime_map(
    runtime: dict[str, Any],
    station_names: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Builds runtime parameters per station.

    Matching priority:
    1. station_name
    2. station_id
    3. default values
    """
    stations_runtime = runtime.get("stations", [])
    if not isinstance(stations_runtime, list):
        stations_runtime = []

    by_name: dict[str, dict[str, Any]] = {}
    by_id: dict[int, dict[str, Any]] = {}

    for s in stations_runtime:
        if not isinstance(s, dict):
            continue

        station_name = s.get("station_name")
        station_id = s.get("station_id")

        if station_name is not None:
            by_name[str(station_name)] = s

        if station_id is not None:
            by_id[int(station_id)] = s

    out: dict[str, dict[str, Any]] = {}

    global_soc_min = safe_float(
        get_nested(runtime, ["constraints", "soc_min"], DEFAULT_ENV_PARAMS["soc_min"]),
        DEFAULT_ENV_PARAMS["soc_min"],
    )
    global_soc_max = safe_float(
        get_nested(runtime, ["constraints", "soc_max"], DEFAULT_ENV_PARAMS["soc_max"]),
        DEFAULT_ENV_PARAMS["soc_max"],
    )

    global_grid_limit = safe_float(
        get_nested(
            runtime,
            ["cluster_state", "grid_limit_kw"],
            runtime.get("grid_limit_kw", DEFAULT_ENV_PARAMS["grid_limit_kw"]),
        ),
        DEFAULT_ENV_PARAMS["grid_limit_kw"],
    )

    global_transfer_enabled = bool(
        get_nested(
            runtime,
            ["transfer", "enabled"],
            runtime.get("transfer_enabled", DEFAULT_ENV_PARAMS["transfer_enabled"]),
        )
    )

    global_transfer_capacity = safe_float(
        get_nested(
            runtime,
            ["transfer", "capacity_kw"],
            runtime.get("transfer_capacity_kw", DEFAULT_ENV_PARAMS["transfer_capacity_kw"]),
        ),
        DEFAULT_ENV_PARAMS["transfer_capacity_kw"],
    )

    global_transfer_loss_rate = safe_float(
        get_nested(
            runtime,
            ["transfer", "loss_rate"],
            runtime.get("transfer_loss_rate", DEFAULT_ENV_PARAMS["transfer_loss_rate"]),
        ),
        DEFAULT_ENV_PARAMS["transfer_loss_rate"],
    )

    for idx, name in enumerate(station_names):
        s = by_name.get(name, by_id.get(idx, {}))
        current_state = s.get("current_state", {}) if isinstance(s.get("current_state", {}), dict) else {}

        soc = (
            s.get("soc")
            if "soc" in s
            else s.get("current_soc")
            if "current_soc" in s
            else current_state.get("soc", DEFAULT_ENV_PARAMS["initial_soc"])
        )

        out[name] = {
            "station_id": int(s.get("station_id", idx)),
            "station_name": name,

            "soc_init": safe_float(soc, DEFAULT_ENV_PARAMS["initial_soc"]),
            "soc_min": safe_float(s.get("soc_min", global_soc_min), global_soc_min),
            "soc_max": safe_float(s.get("soc_max", global_soc_max), global_soc_max),

            "ess_capacity_kwh": safe_float(
                s.get("ess_capacity_kwh", DEFAULT_ENV_PARAMS["ess_capacity_kwh"]),
                DEFAULT_ENV_PARAMS["ess_capacity_kwh"],
            ),
            "ess_max_charge_kw": safe_float(
                s.get("ess_max_charge_kw", DEFAULT_ENV_PARAMS["ess_max_charge_kw"]),
                DEFAULT_ENV_PARAMS["ess_max_charge_kw"],
            ),
            "ess_max_discharge_kw": safe_float(
                s.get("ess_max_discharge_kw", DEFAULT_ENV_PARAMS["ess_max_discharge_kw"]),
                DEFAULT_ENV_PARAMS["ess_max_discharge_kw"],
            ),
            "ess_charge_efficiency": safe_float(
                s.get("ess_charge_efficiency", DEFAULT_ENV_PARAMS["ess_charge_efficiency"]),
                DEFAULT_ENV_PARAMS["ess_charge_efficiency"],
            ),
            "ess_discharge_efficiency": safe_float(
                s.get("ess_discharge_efficiency", DEFAULT_ENV_PARAMS["ess_discharge_efficiency"]),
                DEFAULT_ENV_PARAMS["ess_discharge_efficiency"],
            ),

            # Current env treats grid_limit_kw as cluster-level limit.
            "grid_limit_kw": safe_float(s.get("grid_limit_kw", global_grid_limit), global_grid_limit),

            "transfer_enabled": bool(s.get("transfer_enabled", global_transfer_enabled)),
            "transfer_capacity_kw": safe_float(
                s.get("transfer_capacity_kw", global_transfer_capacity),
                global_transfer_capacity,
            ),
            "transfer_loss_rate": safe_float(
                s.get("transfer_loss_rate", global_transfer_loss_rate),
                global_transfer_loss_rate,
            ),
        }

    return out


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def build_sac_runtime_input_table(
    demand_csv: str | Path,
    pv_csv: str | Path,
    runtime_json: str | Path | None,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
    Builds a 5-station x 24-hour input table required by MultiStationEVSchedulerEnv.
    """
    demand_path = Path(demand_csv)
    pv_path = Path(pv_csv)
    output_path = Path(output_csv)

    if not demand_path.exists():
        raise FileNotFoundError(f"Demand CSV not found: {demand_path}")

    if not pv_path.exists():
        raise FileNotFoundError(f"PV CSV not found: {pv_path}")

    runtime = load_json_if_exists(runtime_json)

    demand_df = pd.read_csv(demand_path)
    pv_df = pd.read_csv(pv_path)

    require_columns(
        demand_df,
        ["station_name", "slot", "slot_label", "slot_start", "slot_end", "predicted_demand_kwh"],
        "Demand forecast CSV",
    )

    require_columns(
        pv_df,
        ["slot", "predicted_pv_kwh"],
        "PV forecast CSV",
    )

    station_names = list(dict.fromkeys(demand_df["station_name"].tolist()))
    if len(station_names) != 5:
        raise ValueError(f"Expected 5 station names from demand CSV, got {len(station_names)}: {station_names}")

    if demand_df["slot"].nunique() != 24:
        raise ValueError(f"Demand CSV must contain 24 slots, got {demand_df['slot'].nunique()}")

    if pv_df["slot"].nunique() != 24:
        raise ValueError(f"PV CSV must contain 24 slots, got {pv_df['slot'].nunique()}")

    tou_price_map = normalize_tou_price(runtime)
    station_runtime = build_station_runtime_map(runtime, station_names)

    demand_df = demand_df.copy()
    demand_df["slot"] = demand_df["slot"].astype(int)
    demand_df["hour"] = demand_df["slot"].astype(int)

    demand_df["station_id"] = demand_df["station_name"].map(
        {name: station_runtime[name]["station_id"] for name in station_names}
    )
    demand_df["demand_kwh"] = demand_df["predicted_demand_kwh"].astype(float)

    pv_small = pv_df[["slot", "predicted_pv_kwh"]].copy()
    pv_small["slot"] = pv_small["slot"].astype(int)
    pv_small = pv_small.rename(columns={"predicted_pv_kwh": "pv_kwh"})

    if "predicted_norm_kwh_per_kw" in pv_df.columns:
        pv_small["pv_norm_kwh_per_kw"] = pv_df["predicted_norm_kwh_per_kw"].astype(float)
    else:
        pv_small["pv_norm_kwh_per_kw"] = 0.0

    out = demand_df.merge(pv_small, on="slot", how="left", validate="many_to_one")

    if out["pv_kwh"].isna().any():
        raise ValueError("PV merge failed. Some pv_kwh values are NaN.")

    out["tou_price"] = out["hour"].map(tou_price_map)
    if out["tou_price"].isna().any():
        missing_hours = sorted(out.loc[out["tou_price"].isna(), "hour"].unique().tolist())
        raise ValueError(f"TOU price missing for hours: {missing_hours}")

    for col in [
        "soc_init",
        "soc_min",
        "soc_max",
        "ess_capacity_kwh",
        "ess_max_charge_kw",
        "ess_max_discharge_kw",
        "ess_charge_efficiency",
        "ess_discharge_efficiency",
        "grid_limit_kw",
        "transfer_enabled",
        "transfer_capacity_kw",
        "transfer_loss_rate",
    ]:
        out[col] = out["station_name"].map({name: station_runtime[name][col] for name in station_names})

    out["cluster_predicted_demand_kwh"] = out.groupby("slot")["demand_kwh"].transform("sum")
    out["cluster_predicted_pv_kwh"] = out.groupby("slot")["pv_kwh"].transform("sum")

    columns = [
        "slot",
        "hour",
        "slot_label",
        "slot_start",
        "slot_end",
        "station_id",
        "station_name",
        "pv_kwh",
        "pv_norm_kwh_per_kw",
        "demand_kwh",
        "cluster_predicted_pv_kwh",
        "cluster_predicted_demand_kwh",
        "tou_price",
        "soc_init",
        "soc_min",
        "soc_max",
        "ess_capacity_kwh",
        "ess_max_charge_kw",
        "ess_max_discharge_kw",
        "ess_charge_efficiency",
        "ess_discharge_efficiency",
        "grid_limit_kw",
        "transfer_enabled",
        "transfer_capacity_kw",
        "transfer_loss_rate",
    ]

    out = out[columns].sort_values(["slot", "station_id"]).reset_index(drop=True)

    if len(out) != 120:
        raise ValueError(f"SAC runtime input must have 120 rows, got {len(out)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    return out


def run_sac_inference(
    model_path: str | Path,
    sac_input_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SAC model not found: {model_path}")

    model = SAC.load(model_path, device=DEVICE)

    env = MultiStationEVSchedulerEnv(
        input_csv=sac_input_csv,
        use_scenarios=False,
        random_initial_soc=False,
        seed=42,
    )

    obs, info = env.reset(seed=42)

    done = False
    total_reward = 0.0
    action_records: list[dict[str, Any]] = []

    while not done:
        current_slot = env.current_slot

        action, _ = model.predict(obs, deterministic=True)

        ess_ratio = action[: env.station_count]
        pv_priority_raw = action[env.station_count:]
        pv_priority = (pv_priority_raw + 1.0) / 2.0

        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        for station_idx in range(env.station_count):
            action_records.append(
                {
                    "slot": current_slot,
                    "hour": current_slot,
                    "station_id": station_idx,
                    "ess_ratio": float(ess_ratio[station_idx]),
                    "pv_priority_raw": float(pv_priority_raw[station_idx]),
                    "pv_priority": float(pv_priority[station_idx]),
                }
            )

    schedule_df = env.get_episode_dataframe()
    action_df = pd.DataFrame(action_records)

    return schedule_df, action_df, total_reward


def attach_transfer_columns(schedule_df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    out = schedule_df.copy()

    cfg_cols = [
        "slot",
        "station_id",
        "grid_limit_kw",
        "transfer_enabled",
        "transfer_capacity_kw",
        "transfer_loss_rate",
    ]

    cfg = input_df[cfg_cols].copy()

    if "grid_limit_kw" in out.columns:
        out = out.drop(columns=["grid_limit_kw"])

    out = out.merge(
        cfg,
        on=["slot", "station_id"],
        how="left",
        validate="one_to_one",
    )

    out["transfer_out_kwh"] = 0.0
    out["transfer_in_kwh"] = 0.0
    out["transfer_loss_kwh"] = 0.0
    out["transfer"] = [[] for _ in range(len(out))]

    return out


def apply_rule_based_transfer(schedule_df: pd.DataFrame) -> pd.DataFrame:
    out = schedule_df.copy()

    for slot, idxs in out.groupby("slot").groups.items():
        idxs = list(idxs)
        hour_df = out.loc[idxs].copy()

        transfer_enabled = bool(hour_df["transfer_enabled"].iloc[0])
        if not transfer_enabled:
            continue

        transfer_capacity = safe_float(hour_df["transfer_capacity_kw"].iloc[0])
        loss_rate = safe_float(hour_df["transfer_loss_rate"].iloc[0])

        donor_indices = [
            idx for idx in idxs
            if safe_float(out.at[idx, "pv_lost_kwh"]) > 1e-9
        ]

        recipient_indices = [
            idx for idx in idxs
            if safe_float(out.at[idx, "grid_to_ev_kwh"]) > 1e-9
        ]

        used_links: set[tuple[int, int]] = set()

        for d_idx in donor_indices:
            donor_station = int(out.at[d_idx, "station_id"])

            for r_idx in recipient_indices:
                recipient_station = int(out.at[r_idx, "station_id"])

                if donor_station == recipient_station:
                    continue

                if (donor_station, recipient_station) in used_links:
                    continue

                donor_surplus = safe_float(out.at[d_idx, "pv_lost_kwh"])
                recipient_grid_need = safe_float(out.at[r_idx, "grid_to_ev_kwh"])

                if donor_surplus <= 1e-9 or recipient_grid_need <= 1e-9:
                    continue

                max_send_for_recipient = recipient_grid_need / max(1e-9, (1.0 - loss_rate))

                send_kwh = min(
                    donor_surplus,
                    transfer_capacity,
                    max_send_for_recipient,
                )

                receive_kwh = send_kwh * (1.0 - loss_rate)
                loss_kwh = send_kwh - receive_kwh

                if send_kwh <= 1e-9:
                    continue

                out.at[d_idx, "pv_lost_kwh"] = donor_surplus - send_kwh
                out.at[d_idx, "transfer_out_kwh"] = safe_float(out.at[d_idx, "transfer_out_kwh"]) + send_kwh
                out.at[d_idx, "transfer_loss_kwh"] = safe_float(out.at[d_idx, "transfer_loss_kwh"]) + loss_kwh

                transfer_list = list(out.at[d_idx, "transfer"])
                transfer_list.append(
                    {
                        "target_station_id": recipient_station,

                        "transfer_power_kw": round(send_kwh, 6),
                        "received_power_kw": round(receive_kwh, 6),
                        "loss_power_kw": round(loss_kwh, 6),

                        "transfer_energy_kwh": round(send_kwh, 6),
                        "received_energy_kwh": round(receive_kwh, 6),
                        "loss_energy_kwh": round(loss_kwh, 6),
                    }
                )
                out.at[d_idx, "transfer"] = transfer_list

                out.at[r_idx, "grid_to_ev_kwh"] = max(0.0, recipient_grid_need - receive_kwh)
                out.at[r_idx, "grid_usage_kwh"] = max(
                    0.0,
                    safe_float(out.at[r_idx, "grid_usage_kwh"]) - receive_kwh,
                )
                out.at[r_idx, "transfer_in_kwh"] = safe_float(out.at[r_idx, "transfer_in_kwh"]) + receive_kwh
                out.at[r_idx, "self_supply_kwh"] = safe_float(out.at[r_idx, "self_supply_kwh"]) + receive_kwh

                used_links.add((donor_station, recipient_station))

    return out.sort_values(["slot", "station_id"]).reset_index(drop=True)


def calculate_metrics(
    input_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    total_reward: float,
) -> dict[str, Any]:
    grid_only_cost = float((input_df["demand_kwh"] * input_df["tou_price"]).sum())
    sac_cost = float((schedule_df["grid_usage_kwh"] * schedule_df["tou_price"]).sum())

    hourly_grid = schedule_df.groupby("slot")["grid_usage_kwh"].sum()
    hourly_limit = input_df.groupby("slot")["grid_limit_kw"].first()

    cost_reduction = grid_only_cost - sac_cost
    cost_reduction_pct = (cost_reduction / grid_only_cost * 100.0) if grid_only_cost > 0 else 0.0

    return {
        "grid_only_cost_krw": round(grid_only_cost, 6),
        "sac_cost_with_transfer_krw": round(sac_cost, 6),
        "cost_reduction_krw": round(cost_reduction, 6),
        "cost_reduction_pct": round(cost_reduction_pct, 6),

        "total_demand_kwh": round(float(schedule_df["demand_kwh"].sum()), 6),
        "total_pv_kwh": round(float(schedule_df["pv_kwh"].sum()), 6),
        "total_grid_usage_kwh": round(float(schedule_df["grid_usage_kwh"].sum()), 6),
        "total_self_supply_kwh": round(float(schedule_df["self_supply_kwh"].sum()), 6),
        "total_pv_lost_kwh": round(float(schedule_df["pv_lost_kwh"].sum()), 6),

        "total_transfer_out_kwh": round(float(schedule_df["transfer_out_kwh"].sum()), 6),
        "total_transfer_in_kwh": round(float(schedule_df["transfer_in_kwh"].sum()), 6),
        "total_transfer_loss_kwh": round(float(schedule_df["transfer_loss_kwh"].sum()), 6),

        "max_cluster_grid_usage_kwh_per_hour": round(float(hourly_grid.max()), 6),
        "peak_violation_slots": int((hourly_grid > hourly_limit).sum()),

        "soc_min": round(float(schedule_df["soc_after"].min()), 6),
        "soc_max": round(float(schedule_df["soc_after"].max()), 6),

        "total_reward_before_transfer": round(float(total_reward), 6),
        "device": DEVICE,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transfer_postprocess_enabled": True,
    }


def build_forecast_results_block(input_df: pd.DataFrame) -> dict[str, Any]:
    """
    Build forecast result blocks for backend response.

    Includes:
    - PV day-ahead forecast: 24 rows
    - Demand day-ahead forecast by station: 5 stations x 24 hours = 120 rows
    - Cluster demand forecast: 24 rows

    Note:
    - pv_kwh is per-station PV forecast.
    - cluster_predicted_pv_kwh is sum of PV forecast over all 5 stations.
    - demand_kwh is per-station EV demand forecast.
    - cluster_predicted_demand_kwh is sum of demand over all 5 stations.
    """
    df = input_df.copy()

    pv_cols = [
        "slot",
        "hour",
        "slot_label",
        "slot_start",
        "slot_end",
        "pv_kwh",
        "cluster_predicted_pv_kwh",
    ]

    pv_df = (
        df[pv_cols]
        .drop_duplicates("slot")
        .sort_values("slot")
        .reset_index(drop=True)
    )

    pv_day_ahead_forecast = []
    for _, r in pv_df.iterrows():
        pv_day_ahead_forecast.append(
            {
                "slot": int(r["slot"]),
                "hour": int(r["hour"]),
                "slot_label": str(r["slot_label"]),
                "slot_start": str(r["slot_start"]),
                "slot_end": str(r["slot_end"]),
                "predicted_pv_kwh_per_station": round(float(r["pv_kwh"]), 6),
                "predicted_cluster_pv_kwh": round(float(r["cluster_predicted_pv_kwh"]), 6),
            }
        )

    demand_cols = [
        "station_id",
        "station_name",
        "slot",
        "hour",
        "slot_label",
        "slot_start",
        "slot_end",
        "demand_kwh",
    ]

    demand_df = (
        df[demand_cols]
        .sort_values(["station_id", "slot"])
        .reset_index(drop=True)
    )

    demand_day_ahead_forecast = []
    for _, r in demand_df.iterrows():
        demand_day_ahead_forecast.append(
            {
                "station_id": int(r["station_id"]),
                "station_name": str(r["station_name"]),
                "slot": int(r["slot"]),
                "hour": int(r["hour"]),
                "slot_label": str(r["slot_label"]),
                "slot_start": str(r["slot_start"]),
                "slot_end": str(r["slot_end"]),
                "predicted_demand_kwh": round(float(r["demand_kwh"]), 6),
            }
        )

    cluster_cols = [
        "slot",
        "hour",
        "slot_label",
        "slot_start",
        "slot_end",
        "cluster_predicted_demand_kwh",
    ]

    cluster_df = (
        df[cluster_cols]
        .drop_duplicates("slot")
        .sort_values("slot")
        .reset_index(drop=True)
    )

    demand_cluster_forecast = []
    for _, r in cluster_df.iterrows():
        demand_cluster_forecast.append(
            {
                "slot": int(r["slot"]),
                "hour": int(r["hour"]),
                "slot_label": str(r["slot_label"]),
                "slot_start": str(r["slot_start"]),
                "slot_end": str(r["slot_end"]),
                "predicted_cluster_demand_kwh": round(float(r["cluster_predicted_demand_kwh"]), 6),
            }
        )

    return {
        "pv_day_ahead_forecast": pv_day_ahead_forecast,
        "demand_day_ahead_forecast": demand_day_ahead_forecast,
        "demand_cluster_forecast": demand_cluster_forecast,
    }


def build_backend_response_json(
    schedule_df: pd.DataFrame,
    input_df: pd.DataFrame,
    metrics: dict[str, Any],
    runtime: dict[str, Any],
    model_path: str | Path,
) -> dict[str, Any]:
    first = input_df.iloc[0]

    request_id = runtime.get("request_id", "sac-runtime-schedule-0001")
    schedule_target_date = runtime.get("schedule_target_date", str(first["slot_start"])[:10])

    slot_info = (
        input_df[["slot", "slot_label", "slot_start", "slot_end"]]
        .drop_duplicates("slot")
        .set_index("slot")
        .to_dict(orient="index")
    )

    station_day_ahead_schedule = []

    for station_id, sdf in schedule_df.groupby("station_id", sort=True):
        sdf = sdf.sort_values("slot").reset_index(drop=True)

        hourly_plan = []

        for _, r in sdf.iterrows():
            hour = int(r["hour"])
            slot = int(r["slot"])
            ess_signed = float(r["ess_power_kwh_signed"])

            if ess_signed > 1e-9:
                ess_mode = "charge"
                ess_power = abs(ess_signed)
            elif ess_signed < -1e-9:
                ess_mode = "discharge"
                ess_power = abs(ess_signed)
            else:
                ess_mode = "idle"
                ess_power = 0.0

            s_info = slot_info.get(slot, {})

            hourly_plan.append(
                {
                    "hour": hour,
                    "slot_label": str(s_info.get("slot_label", f"{hour:02d}:00~{hour + 1:02d}:00")),
                    "slot_start": str(s_info.get("slot_start", "")),
                    "slot_end": str(s_info.get("slot_end", "")),

                    "ess_mode": ess_mode,
                    "ess_power_kw": round(ess_power, 6),
                    "ess_power_signed_kw": round(ess_signed, 6),
                    "ess_energy_kwh": round(ess_power, 6),

                    "grid_usage_kw": round(float(r["grid_usage_kwh"]), 6),
                    "grid_usage_kwh": round(float(r["grid_usage_kwh"]), 6),

                    "pv_generation_pred_kwh": round(float(r["pv_kwh"]), 6),
                    "load_pred_kwh": round(float(r["demand_kwh"]), 6),

                    "pv_priority": round(float(r["pv_priority"]), 6),
                    "expected_soc": round(float(r["soc_after"]), 6),

                    "transfer": r["transfer"],
                }
            )

        station_day_ahead_schedule.append(
            {
                "station_id": int(station_id),
                "station_name": str(sdf["station_name"].iloc[0]),
                "hourly_plan": hourly_plan,
            }
        )

    return {
        "request_id": request_id,
        "timestamp": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "schedule_target_date": schedule_target_date,
        "schedule_horizon_hours": 24,
        "schedule_mode": "day-ahead",

        "model": {
            "algorithm": "SAC",
            "version": "ev-rl-scheduler-v3-1m",
            "model_path": str(model_path),
            "device": DEVICE,
            "transfer_action_enabled": False,
            "transfer_postprocess_enabled": True,
            "description": "SAC v3-1M controls ESS charge/discharge and PV priority. Rule-based post-processing adds hourly inter-station transfer.",
        },

        "status": {
            "is_success": True,
            "error_code": 0,
            "message": "day_ahead_schedule_created",
        },

        "forecast_results": build_forecast_results_block(input_df),

        "metrics": metrics,
        "station_day_ahead_schedule": station_day_ahead_schedule,
    }


def run_pipeline(
    demand_csv: str | Path = DEFAULT_DEMAND_CSV,
    pv_csv: str | Path = DEFAULT_PV_CSV,
    runtime_json: str | Path | None = DEFAULT_RUNTIME_JSON,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    runtime_input_csv: str | Path = DEFAULT_RUNTIME_INPUT_CSV,
    output_json: str | Path = DEFAULT_OUTPUT_JSON,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    output_action_csv: str | Path = DEFAULT_OUTPUT_ACTION_CSV,
    output_metrics_json: str | Path = DEFAULT_OUTPUT_METRICS_JSON,
) -> dict[str, Any]:
    runtime = load_json_if_exists(runtime_json)

    input_df = build_sac_runtime_input_table(
        demand_csv=demand_csv,
        pv_csv=pv_csv,
        runtime_json=runtime_json,
        output_csv=runtime_input_csv,
    )

    schedule_df, action_df, total_reward = run_sac_inference(
        model_path=model_path,
        sac_input_csv=runtime_input_csv,
    )

    schedule_df = attach_transfer_columns(schedule_df, input_df)
    schedule_df = apply_rule_based_transfer(schedule_df)

    metrics = calculate_metrics(
        input_df=input_df,
        schedule_df=schedule_df,
        total_reward=total_reward,
    )

    response = build_backend_response_json(
        schedule_df=schedule_df,
        input_df=input_df,
        metrics=metrics,
        runtime=runtime,
        model_path=model_path,
    )

    output_json = Path(output_json)
    output_csv = Path(output_csv)
    output_action_csv = Path(output_action_csv)
    output_metrics_json = Path(output_metrics_json)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_action_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_json.parent.mkdir(parents=True, exist_ok=True)

    schedule_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    action_df.to_csv(output_action_csv, index=False, encoding="utf-8-sig")

    with output_metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    return response


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--demand-csv", default=str(DEFAULT_DEMAND_CSV))
    parser.add_argument("--pv-csv", default=str(DEFAULT_PV_CSV))
    parser.add_argument("--runtime-json", default=str(DEFAULT_RUNTIME_JSON))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))

    parser.add_argument("--runtime-input-csv", default=str(DEFAULT_RUNTIME_INPUT_CSV))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-action-csv", default=str(DEFAULT_OUTPUT_ACTION_CSV))
    parser.add_argument("--output-metrics-json", default=str(DEFAULT_OUTPUT_METRICS_JSON))

    args = parser.parse_args()

    response = run_pipeline(
        demand_csv=args.demand_csv,
        pv_csv=args.pv_csv,
        runtime_json=args.runtime_json,
        model_path=args.model_path,
        runtime_input_csv=args.runtime_input_csv,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_action_csv=args.output_action_csv,
        output_metrics_json=args.output_metrics_json,
    )

    print("✅ SAC inference pipeline completed")
    print(f"MODEL       : {args.model_path}")
    print(f"MODEL VER   : ev-rl-scheduler-v3-1m")
    print(f"DEVICE      : {DEVICE}")
    print(f"OUTPUT JSON : {args.output_json}")
    print(f"METRICS     : {args.output_metrics_json}")
    print()
    print(json.dumps(response["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
