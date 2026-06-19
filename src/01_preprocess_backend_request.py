#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_preprocess_backend_request.py

Backend full JSON -> three runtime JSON files for the scheduling pipeline.

Outputs under --output-dir:
- demand_backend_input.json : input for demand ML inference
- pv_backend_input.json     : input for PV ML inference
- rl_runtime_request.json   : input for SAC/RL inference
- preprocess_report.json    : validation counts and warnings

Expected backend full JSON keys:
- request_id, request_timestamp, schedule_target_date, schedule_horizon_hours
- stations
- demand_past_demand_hourly
- demand_past_weather_hourly
- demand_forecast_short_term_hourly
- pv_past_generation_hourly
- pv_past_weather_hourly
- pv_forecast_short_term_hourly
- tou_price_hourly or tou_price_krw_per_kwh
- cluster_state, constraints, transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def pick(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip()
        if v in ["", "없음", "강수없음", "적설없음", "-", "null", "None"]:
            return 0.0
        v = v.replace("mm", "").replace("cm", "").replace("m/s", "").strip()
        try:
            return float(v)
        except ValueError:
            return default
    try:
        return float(value)
    except Exception:
        return default


def normalize_timestamp(row: dict[str, Any], field_candidates: list[str]) -> str:
    value = pick(row, field_candidates)
    if value is None:
        raise ValueError(f"timestamp field missing. candidates={field_candidates}, row={row}")
    return str(value)


def normalize_weather_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize ASOS/AWS observed weather row into model field names."""
    return {
        "tm": normalize_timestamp(row, ["timestamp", "tm", "time", "일시"]),
        "ta": to_float(pick(row, ["ta", "temperature_c", "temperature", "TMP", "기온"])),
        "rn": to_float(pick(row, ["rn", "precipitation_mm", "rain", "강수량"])),
        "ws": to_float(pick(row, ["ws", "wind_speed_ms", "wind_speed", "WSD", "풍속"])),
        "wd": to_float(pick(row, ["wd", "wind_direction_deg", "wind_direction", "VEC", "풍향"])),
        "hm": to_float(pick(row, ["hm", "humidity_pct", "humidity", "REH", "습도"])),
        "pa": to_float(pick(row, ["pa", "pressure_hpa", "pressure", "현지기압"])),
        "ps": to_float(pick(row, ["ps", "sea_level_pressure_hpa", "sea_level_pressure", "해면기압"])),
        "ss": to_float(pick(row, ["ss", "sunshine_hours", "sunshine", "일조"], 0.0)),
        "icsr": to_float(pick(row, ["icsr", "solar_radiation_mj_m2", "solar_radiation", "radiation", "si", "일사"], 0.0)),
        "dsnw": to_float(pick(row, ["dsnw", "snow_cm", "snow", "적설"], 0.0)),
        "hr3Fhsc": to_float(pick(row, ["hr3Fhsc", "new_snow_3h_cm", "new_snow_3h", "3시간신적설"], 0.0)),
        "dc10Tca": to_float(pick(row, ["dc10Tca", "cloud_amount", "cloud", "cld", "SKY", "전운량"], 0.0)),
    }


def normalize_forecast_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize KMA short-term forecast row into model field names."""
    tmp = to_float(pick(row, ["TMP", "tmp", "temperature_c", "temperature"]))
    return {
        "tmef": normalize_timestamp(row, ["tmef", "timestamp", "time", "forecast_time"]),
        "TMP": tmp,
        "POP": to_float(pick(row, ["POP", "pop", "precipitation_probability"], 0.0)),
        "PTY": to_float(pick(row, ["PTY", "pty", "precipitation_type"], 0.0)),
        "PCP": to_float(pick(row, ["PCP", "pcp", "precipitation_mm"], 0.0)),
        "SNO": to_float(pick(row, ["SNO", "sno", "snow_cm"], 0.0)),
        "REH": to_float(pick(row, ["REH", "reh", "humidity_pct"], 0.0)),
        "SKY": to_float(pick(row, ["SKY", "sky", "cloud_code"], 1.0)),
        "WSD": to_float(pick(row, ["WSD", "wsd", "wind_speed_ms"], 0.0)),
        "VEC": to_float(pick(row, ["VEC", "vec", "wind_direction_deg"], 0.0)),
        "UUU": to_float(pick(row, ["UUU", "uuu"], 0.0)),
        "VVV": to_float(pick(row, ["VVV", "vvv"], 0.0)),
        "TMN": to_float(pick(row, ["TMN", "tmn", "min_temperature_c"], tmp)),
        "TMX": to_float(pick(row, ["TMX", "tmx", "max_temperature_c"], tmp)),
    }


def station_name_map(payload: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for s in payload.get("stations", []):
        if "station_id" in s and "station_name" in s:
            out[int(s["station_id"])] = str(s["station_name"])
    return out


def normalize_demand_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sid_to_name = station_name_map(payload)
    rows = []
    for row in payload.get("demand_past_demand_hourly", []):
        sid = pick(row, ["station_id", "stationId"])
        station_name = pick(row, ["station_name", "stationName"])
        if station_name is None and sid is not None:
            station_name = sid_to_name.get(int(sid))
        if station_name is None:
            raise ValueError(f"demand row station_name missing: {row}")
        rows.append({
            "tm": normalize_timestamp(row, ["timestamp", "tm", "slot_start", "time"]),
            "station_name": str(station_name),
            "demand_kwh": to_float(pick(row, ["demand_kwh", "demand", "load_kwh", "load"])),
        })
    return rows


def normalize_pv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("pv_past_generation_hourly", []):
        rows.append({
            "tm": normalize_timestamp(row, ["timestamp", "tm", "slot_start", "time"]),
            "gen_kwh": to_float(pick(row, ["gen_kwh", "pv_generation_kwh", "generation_kwh", "pv_kwh"])),
        })
    return rows


def normalize_tou(payload: dict[str, Any]) -> dict[str, float]:
    if isinstance(payload.get("tou_price_krw_per_kwh"), dict):
        return {str(int(k)): float(v) for k, v in payload["tou_price_krw_per_kwh"].items()}
    if isinstance(payload.get("tou_price_krw_per_kwh"), list):
        vals = payload["tou_price_krw_per_kwh"]
        return {str(i): float(vals[i]) for i in range(24)}
    if isinstance(payload.get("tou_price_hourly"), list):
        out = {}
        for item in payload["tou_price_hourly"]:
            hour = int(pick(item, ["slot", "hour"]))
            price = float(pick(item, ["price_krw_per_kwh", "tou_price", "price"]))
            out[str(hour)] = price
        return out
    raise ValueError("TOU price is required: tou_price_krw_per_kwh or tou_price_hourly")


def normalize_runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(payload)
    runtime["tou_price_krw_per_kwh"] = normalize_tou(payload)
    runtime.pop("demand_past_demand_hourly", None)
    runtime.pop("demand_past_weather_hourly", None)
    runtime.pop("demand_forecast_short_term_hourly", None)
    runtime.pop("pv_past_generation_hourly", None)
    runtime.pop("pv_past_weather_hourly", None)
    runtime.pop("pv_forecast_short_term_hourly", None)
    return runtime


def validate_counts(payload: dict[str, Any], strict: bool) -> dict[str, Any]:
    counts = {
        "stations": len(payload.get("stations", [])),
        "demand_past_demand_hourly": len(payload.get("demand_past_demand_hourly", [])),
        "demand_past_weather_hourly": len(payload.get("demand_past_weather_hourly", [])),
        "demand_forecast_short_term_hourly": len(payload.get("demand_forecast_short_term_hourly", [])),
        "pv_past_generation_hourly": len(payload.get("pv_past_generation_hourly", [])),
        "pv_past_weather_hourly": len(payload.get("pv_past_weather_hourly", [])),
        "pv_forecast_short_term_hourly": len(payload.get("pv_forecast_short_term_hourly", [])),
    }
    expected_min = {
        "stations": 5,
        "demand_past_demand_hourly": 950,
        "demand_past_weather_hourly": 168,
        "demand_forecast_short_term_hourly": 48,
        "pv_past_generation_hourly": 190,
        "pv_past_weather_hourly": 168,
        "pv_forecast_short_term_hourly": 48,
    }
    warnings = []
    errors = []
    for key, minimum in expected_min.items():
        if counts[key] < minimum:
            errors.append(f"{key}: expected at least {minimum}, got {counts[key]}")
    if counts["stations"] != 5:
        errors.append(f"station count must be exactly 5, got {counts['stations']}")
    for s in payload.get("stations", []):
        chargers = s.get("chargers")
        if chargers is not None and len(chargers) != 5:
            errors.append(f"station_id={s.get('station_id')} charger count must be 5, got {len(chargers)}")
    if strict and errors:
        raise ValueError("backend request validation failed:\n" + "\n".join(errors))
    return {"counts": counts, "warnings": warnings, "errors": errors, "strict": strict}


def preprocess(input_json: str | Path, output_dir: str | Path, strict: bool = True) -> dict[str, Path]:
    input_path = Path(input_json)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    report = validate_counts(payload, strict=strict)

    demand_payload = {
        "request_id": payload.get("request_id"),
        "request_timestamp": payload.get("request_timestamp"),
        "schedule_target_date": payload.get("schedule_target_date"),
        "schedule_horizon_hours": payload.get("schedule_horizon_hours", 24),
        "past_demand_hourly": normalize_demand_rows(payload),
        "past_weather_hourly": [normalize_weather_row(x) for x in payload.get("demand_past_weather_hourly", [])],
        "forecast_short_term_hourly": [normalize_forecast_row(x) for x in payload.get("demand_forecast_short_term_hourly", [])],
    }

    pv_payload = {
        "request_id": payload.get("request_id"),
        "request_timestamp": payload.get("request_timestamp"),
        "schedule_target_date": payload.get("schedule_target_date"),
        "schedule_horizon_hours": payload.get("schedule_horizon_hours", 24),
        "past_pv_hourly": normalize_pv_rows(payload),
        "past_weather_hourly": [normalize_weather_row(x) for x in payload.get("pv_past_weather_hourly", [])],
        "forecast_short_term_hourly": [normalize_forecast_row(x) for x in payload.get("pv_forecast_short_term_hourly", [])],
    }

    runtime_payload = normalize_runtime_payload(payload)

    demand_path = output_path / "demand_backend_input.json"
    pv_path = output_path / "pv_backend_input.json"
    runtime_path = output_path / "rl_runtime_request.json"
    report_path = output_path / "preprocess_report.json"

    for path, data in [
        (demand_path, demand_payload),
        (pv_path, pv_payload),
        (runtime_path, runtime_payload),
        (report_path, report),
    ]:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "demand_input": demand_path,
        "pv_input": pv_path,
        "runtime_input": runtime_path,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-strict", action="store_true", help="Do not fail on insufficient counts; write report only.")
    args = parser.parse_args()

    paths = preprocess(
        input_json=args.input_json,
        output_dir=args.output_dir,
        strict=not args.no_strict,
    )

    print("[DONE] preprocessing")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
