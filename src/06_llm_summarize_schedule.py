#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_llm_summarize_schedule.py

Reads ai_schedule_response.json and creates:
1. llm_input_compact.json
2. llm_summary.json
3. ai_schedule_response_with_llm.json

Important:
- Do NOT expose cost_reduction_krw / cost_reduction_pct in LLM summary.
- Day-ahead result only contains predicted/estimated values.
- Actual operation performance must be evaluated after next-day real data is collected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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


FORBIDDEN_SUMMARY_TERMS = [
    # 실제 운영 후에만 판단 가능한 성과/비교 표현 금지
    "비용 절감",
    "절감률",
    "절감액",
    "비용을 줄",
    "cost_reduction",
    "grid-only",
    "grid only",
    "grid_only",
    "그리드온리",
    "그리드 온리",

    # 근거 없는 성과 단정 표현 금지
    "경제성과 안정성",
    "우수한 성과",
    "실제 성과가 우수",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def fmt_krw(value: Any) -> str:
    v = round(safe_float(value, 0.0))
    return f"{v:,}원"


def fmt_kwh(value: Any) -> str:
    return f"{safe_float(value, 0.0):,.2f}kWh"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_tou_price_map_from_runtime_input(runtime_input_csv: Path | None) -> dict[tuple[int, int], float]:
    """
    Returns {(station_id, hour): tou_price}.
    If runtime_input_csv is missing, returns empty dict.
    """
    if runtime_input_csv is None:
        return {}

    if not runtime_input_csv.exists():
        return {}

    df = pd.read_csv(runtime_input_csv)

    required = {"station_id", "hour", "tou_price"}
    if not required.issubset(set(df.columns)):
        return {}

    out: dict[tuple[int, int], float] = {}
    for _, r in df.iterrows():
        station_id = int(r["station_id"])
        hour = int(r["hour"])
        out[(station_id, hour)] = float(r["tou_price"])

    return out


def get_hourly_tou_price(station_id: int, hour: int, tou_map: dict[tuple[int, int], float]) -> float:
    if (station_id, hour) in tou_map:
        return float(tou_map[(station_id, hour)])
    return float(DEFAULT_TOU_PRICE.get(hour, 0.0))


def calculate_station_expected_bill(
    station_id: int,
    hourly_plan: list[dict[str, Any]],
    tou_map: dict[tuple[int, int], float],
) -> float:
    total = 0.0

    for h in hourly_plan:
        if "expected_bill_krw" in h:
            total += safe_float(h.get("expected_bill_krw"), 0.0)
            continue

        hour = safe_int(h.get("hour"), 0)
        grid_usage_kwh = safe_float(h.get("grid_usage_kwh"), 0.0)
        tou_price = get_hourly_tou_price(station_id, hour, tou_map)
        total += grid_usage_kwh * tou_price

    return total


def build_station_demand_summary(
    data: dict[str, Any],
    tou_map: dict[tuple[int, int], float],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    stations = data.get("station_day_ahead_schedule", [])
    if not isinstance(stations, list):
        return summaries

    for station in stations:
        station_id = safe_int(station.get("station_id"), 0)
        station_name = str(station.get("station_name", f"station_{station_id}"))
        hourly_plan = station.get("hourly_plan", [])

        if not isinstance(hourly_plan, list):
            hourly_plan = []

        total_predicted_demand = sum(safe_float(h.get("load_pred_kwh"), 0.0) for h in hourly_plan)
        total_expected_grid_usage = sum(safe_float(h.get("grid_usage_kwh"), 0.0) for h in hourly_plan)
        total_predicted_pv = sum(safe_float(h.get("pv_generation_pred_kwh"), 0.0) for h in hourly_plan)
        total_transfer_out = 0.0
        total_transfer_in = 0.0

        for h in hourly_plan:
            transfer_list = h.get("transfer", [])
            if isinstance(transfer_list, list):
                total_transfer_out += sum(safe_float(t.get("transfer_energy_kwh"), 0.0) for t in transfer_list)

            # recipient 쪽 transfer_in은 hourly_plan에 직접 없을 수 있으므로 전체 요약에서는 생략 가능

        peak_row = max(
            hourly_plan,
            key=lambda h: safe_float(h.get("load_pred_kwh"), 0.0),
            default={},
        )

        expected_bill = calculate_station_expected_bill(
            station_id=station_id,
            hourly_plan=hourly_plan,
            tou_map=tou_map,
        )

        summaries.append(
            {
                "station_id": station_id,
                "station_name": station_name,
                "total_predicted_demand_kwh": round(total_predicted_demand, 2),
                "peak_predicted_demand_kwh": round(safe_float(peak_row.get("load_pred_kwh"), 0.0), 2),
                "peak_demand_slot": str(peak_row.get("slot_label", "")),
                "total_predicted_pv_kwh": round(total_predicted_pv, 2),
                "expected_grid_usage_kwh": round(total_expected_grid_usage, 2),
                "expected_schedule_bill_krw": round(expected_bill),
                "transfer_out_kwh": round(total_transfer_out, 2),
                "transfer_in_kwh": round(total_transfer_in, 2),
            }
        )

    summaries.sort(key=lambda x: x["total_predicted_demand_kwh"], reverse=True)
    return summaries


def build_hourly_cluster_summary(data: dict[str, Any]) -> list[dict[str, Any]]:
    forecast = data.get("forecast_results", {})
    demand_cluster = forecast.get("demand_cluster_forecast", [])
    pv_forecast = forecast.get("pv_day_ahead_forecast", [])

    pv_by_slot = {}
    if isinstance(pv_forecast, list):
        for r in pv_forecast:
            slot = safe_int(r.get("slot"), -1)
            pv_by_slot[slot] = safe_float(r.get("predicted_cluster_pv_kwh"), 0.0)

    out = []
    if isinstance(demand_cluster, list):
        for r in demand_cluster:
            slot = safe_int(r.get("slot"), -1)
            out.append(
                {
                    "slot": slot,
                    "hour": safe_int(r.get("hour"), slot),
                    "slot_label": str(r.get("slot_label", "")),
                    "predicted_cluster_demand_kwh": round(
                        safe_float(r.get("predicted_cluster_demand_kwh"), 0.0),
                        2,
                    ),
                    "predicted_cluster_pv_kwh": round(pv_by_slot.get(slot, 0.0), 2),
                }
            )

    return out


def build_compact_llm_input(
    data: dict[str, Any],
    runtime_input_csv: Path | None,
) -> dict[str, Any]:
    metrics = data.get("metrics", {})
    model_info = data.get("model", {})
    status = data.get("status", {})

    tou_map = load_tou_price_map_from_runtime_input(runtime_input_csv)
    station_summary = build_station_demand_summary(data, tou_map)

    expected_schedule_bill = metrics.get("sac_cost_with_transfer_krw")
    if expected_schedule_bill is None:
        expected_schedule_bill = sum(s["expected_schedule_bill_krw"] for s in station_summary)

    compact = {
        "request_id": data.get("request_id"),
        "schedule_target_date": data.get("schedule_target_date"),
        "schedule_horizon_hours": data.get("schedule_horizon_hours"),
        "schedule_mode": data.get("schedule_mode"),

        "status": {
            "is_success": status.get("is_success"),
            "error_code": status.get("error_code"),
            "message": status.get("message"),
        },

        "model": {
            "algorithm": model_info.get("algorithm"),
            "version": model_info.get("version"),
            "device": model_info.get("device"),
            "transfer_postprocess_enabled": model_info.get("transfer_postprocess_enabled"),
        },

        "estimated_result": {
            "total_predicted_demand_kwh": metrics.get("total_demand_kwh"),
            "total_predicted_pv_kwh": metrics.get("total_pv_kwh"),
            "total_expected_grid_usage_kwh": metrics.get("total_grid_usage_kwh"),
            "expected_schedule_bill_krw": expected_schedule_bill,
            "total_self_supply_kwh": metrics.get("total_self_supply_kwh"),
            "total_pv_lost_kwh": metrics.get("total_pv_lost_kwh"),
            "total_transfer_out_kwh": metrics.get("total_transfer_out_kwh"),
            "total_transfer_in_kwh": metrics.get("total_transfer_in_kwh"),
            "total_transfer_loss_kwh": metrics.get("total_transfer_loss_kwh"),
            "max_cluster_grid_usage_kwh_per_hour": metrics.get("max_cluster_grid_usage_kwh_per_hour"),
            "peak_violation_slots": metrics.get("peak_violation_slots"),
            "soc_min": metrics.get("soc_min"),
            "soc_max": metrics.get("soc_max"),
        },

        "station_demand_summary": station_summary,
        "hourly_cluster_summary": build_hourly_cluster_summary(data),

        "important_notice": (
            "이 결과는 day-ahead 예측 수요와 예측 PV를 기준으로 생성된 예상 스케줄이다. "
            "expected_schedule_bill_krw는 스케줄 적용 기준 예상 전기세이다. "
            "실제 운영 성과 비교는 다음날 실제 수요, 실제 PV, 실제 계통 사용량이 수집된 뒤 별도 평가 단계에서 산출해야 한다. "
            "LLM 요약에서는 cost_reduction_krw, cost_reduction_pct, grid_only_cost_krw를 사용하지 않는다."
        ),
    }

    return compact


def build_rule_based_summary_text(compact: dict[str, Any]) -> str:
    estimated = compact.get("estimated_result", {})
    stations = compact.get("station_demand_summary", [])

    target_date = compact.get("schedule_target_date", "")
    total_demand = estimated.get("total_predicted_demand_kwh")
    total_pv = estimated.get("total_predicted_pv_kwh")
    total_grid = estimated.get("total_expected_grid_usage_kwh")
    bill = estimated.get("expected_schedule_bill_krw")
    peak_violations = estimated.get("peak_violation_slots")
    soc_min = estimated.get("soc_min")
    soc_max = estimated.get("soc_max")

    station_lines = []
    for s in stations:
        station_lines.append(
            f"- {s['station_name']}: 총 예상 수요 {fmt_kwh(s['total_predicted_demand_kwh'])}, "
            f"최대 시간대 수요 {fmt_kwh(s['peak_predicted_demand_kwh'])}"
            f"({s['peak_demand_slot']}), "
            f"예상 전기세 {fmt_krw(s['expected_schedule_bill_krw'])}"
        )

    station_text = "\n".join(station_lines)

    return f"""1. 한 줄 요약
{target_date} EV 충전소 day-ahead 스케줄링이 완료되었고, 총 예상 수요는 {fmt_kwh(total_demand)}, 스케줄 적용 기준 예상 전기세는 {fmt_krw(bill)}입니다.

2. 전체 예상 수요와 예상 전기세
전체 예상 수요는 {fmt_kwh(total_demand)}이며, 예측 PV 발전량은 {fmt_kwh(total_pv)}입니다. 스케줄 적용 후 예상 계통 사용량은 {fmt_kwh(total_grid)}이고, 예상 전기세는 {fmt_krw(bill)}입니다. 이 전기세는 실제 청구 금액이 아니라 예측 수요와 예측 PV 기반의 추정값입니다.

3. 충전소별 예상 수요
{station_text}

4. 운영 해석
낮 시간대에는 PV 발전량을 활용하고, ESS 충전·방전 및 충전소 간 전력 전송 후처리를 통해 계통 사용량을 낮추는 방향으로 스케줄이 구성되었습니다. 피크 위반 예상 슬롯은 {peak_violations}개입니다.

5. 주의할 점
SoC 예상 범위는 {soc_min}~{soc_max}입니다. 이 결과는 다음날 운영 전 예측 기반 계획이므로, 실제 운영 성과 비교는 실제 수요와 실제 계통 사용량이 수집된 뒤 별도 평가 단계에서 계산해야 합니다.

6. 백엔드 화면 표시용 짧은 문구
{target_date} 스케줄링 완료. 총 예상 수요 {fmt_kwh(total_demand)}, 예상 전기세 {fmt_krw(bill)}, 피크 위반 예상 {peak_violations}건입니다. 이 값은 예측 기반 추정값입니다."""


def contains_forbidden_terms(text: str) -> bool:
    lower = text.lower()
    for term in FORBIDDEN_SUMMARY_TERMS:
        if term.lower() in lower:
            return True
    return False


def generate_llm_summary(
    compact: dict[str, Any],
    model_path: Path,
    max_new_tokens: int,
) -> str:
    system_prompt = """
너는 전기차 충전소 AI 스케줄링 결과를 한국어로 요약하는 운영 분석가다.

반드시 지켜야 할 규칙:
- 입력 JSON의 숫자를 임의로 바꾸지 않는다.
- 비용 절감률, 비용 절감액, 비용 절감 효과라는 표현을 절대 쓰지 않는다.
- grid-only와 비교하지 않는다.
- expected_schedule_bill_krw는 예측 수요와 예측 PV 기반의 예상 전기세라고 설명한다.
- 실제 운영 성과 비교는 다음날 실제 데이터 수집 후 별도 평가 단계에서 산출해야 한다고만 설명한다.
- 충전소별 예상 수요와 예상 전기세를 반드시 포함한다.
"""

    user_prompt = f"""
다음은 EV 충전소 day-ahead 스케줄링 결과를 LLM 입력용으로 압축한 JSON이다.

아래 형식으로 한국어 요약을 작성해라.

1. 한 줄 요약
2. 전체 예상 수요와 예상 전기세
3. 충전소별 예상 수요
4. 운영 해석
5. 주의할 점
6. 백엔드 화면 표시용 짧은 문구

금지:
- 비용 절감률 언급 금지
- 비용 절감액 언급 금지
- grid-only 비교 금지
- 실제 성과처럼 표현 금지

JSON:
{json.dumps(compact, ensure_ascii=False, indent=2)}
"""

    print("[LLM] tokenizer loading...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    print("[LLM] model loading...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("[LLM] generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, help="Path to ai_schedule_response.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for LLM summary")
    parser.add_argument(
        "--model-path",
        default="/home/dgx_spark/Desktop/teamplay/scheduling/models/llm/qwen3-4b-instruct",
        help="Local LLM model path",
    )
    parser.add_argument(
        "--runtime-input-csv",
        default=None,
        help="Optional rl_runtime_input_for_sac.csv path. Used to calculate station expected bill accurately.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--skip-llm", action="store_true", help="Only create compact JSON and rule-based summary")
    args = parser.parse_args()

    input_json = Path(args.input_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    runtime_input_csv = Path(args.runtime_input_csv).expanduser().resolve() if args.runtime_input_csv else None

    if not input_json.exists():
        raise FileNotFoundError(f"input JSON not found: {input_json}")

    if not args.skip_llm and not model_path.exists():
        raise FileNotFoundError(f"LLM model path not found: {model_path}")

    data = load_json(input_json)
    compact = build_compact_llm_input(data, runtime_input_csv)

    compact_path = output_dir / "llm_input_compact.json"
    summary_path = output_dir / "llm_summary.json"
    merged_path = output_dir / "ai_schedule_response_with_llm.json"

    save_json(compact_path, compact)

    rule_based_text = build_rule_based_summary_text(compact)

    if args.skip_llm:
        llm_text = rule_based_text
        summary_mode = "rule_based_only"
    else:
        llm_text = generate_llm_summary(
            compact=compact,
            model_path=model_path,
            max_new_tokens=args.max_new_tokens,
        )
        summary_mode = "llm"

        # Safety fallback: if the LLM still mentions reduction/comparison terms, use deterministic summary.
        if contains_forbidden_terms(llm_text):
            print("[WARN] LLM output contained forbidden or unsupported terms. Falling back to rule-based summary.")
            llm_text = rule_based_text
            summary_mode = "rule_based_fallback"

    summary_payload = {
        "summary_status": "success",
        "summary_mode": summary_mode,
        "llm_model_path": str(model_path),
        "input_json": str(input_json),
        "compact_input_json": str(compact_path),
        "summary_text": llm_text,
        "backend_display_text": rule_based_text.split("6. 백엔드 화면 표시용 짧은 문구", 1)[-1].strip(),
        "important_notice": compact["important_notice"],
    }

    save_json(summary_path, summary_payload)

    merged = data.copy()
    merged["llm_summary"] = {
        "summary_status": summary_payload["summary_status"],
        "summary_mode": summary_payload["summary_mode"],
        "summary_text": summary_payload["summary_text"],
        "backend_display_text": summary_payload["backend_display_text"],
        "compact_input_json": str(compact_path),
        "llm_summary_json": str(summary_path),
        "important_notice": compact["important_notice"],
    }

    # Public-facing metrics without cost reduction fields.
    # 기존 metrics는 보존하되, 백엔드 화면에는 public_estimated_metrics 사용 권장.
    merged["public_estimated_metrics"] = compact["estimated_result"]
    merged["station_demand_summary"] = compact["station_demand_summary"]

    save_json(merged_path, merged)

    print("✅ LLM schedule summary completed")
    print(f"COMPACT JSON : {compact_path}")
    print(f"SUMMARY JSON : {summary_path}")
    print(f"MERGED JSON  : {merged_path}")
    print()
    print("[SUMMARY]")
    print(llm_text)


if __name__ == "__main__":
    main()
