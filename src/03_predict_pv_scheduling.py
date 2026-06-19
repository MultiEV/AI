from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb


# PATCH_PV_LAG_NAN_FALLBACK_START
def _patch_fill_pv_lag_nan(frame, feature_cols=None):
    """
    PV 추론 runtime safety patch.

    pv_lag_24h_or_fallback_norm 등 PV lag/fallback feature에 NaN이 남으면
    가능한 대체 feature -> hour/slot 평균 -> 전체 평균 -> 0 순서로 채운다.

    마지막 0.0 fallback은 보수적으로 '해당 lag PV 발전량 없음'으로 처리하는 값이다.
    """
    try:
        import numpy as _np
        import pandas as _pd
    except Exception:
        return frame

    if frame is None or not isinstance(frame, _pd.DataFrame):
        return frame

    if frame.empty:
        return frame

    frame = frame.copy()
    frame.replace([_np.inf, -_np.inf], _np.nan, inplace=True)

    if feature_cols is None:
        candidate_cols = list(frame.columns)
    else:
        candidate_cols = [c for c in feature_cols if c in frame.columns]

    target_cols = []
    for c in candidate_cols:
        lc = str(c).lower()
        if (
            str(c) == "pv_lag_24h_or_fallback_norm"
            or "pv_lag" in lc
            or "pv_roll" in lc
            or "pv_rolling" in lc
            or ("fallback" in lc and "pv" in lc)
        ):
            target_cols.append(c)

    if "pv_lag_24h_or_fallback_norm" in frame.columns:
        if "pv_lag_24h_or_fallback_norm" not in target_cols:
            target_cols.append("pv_lag_24h_or_fallback_norm")

    if not target_cols:
        return frame

    hour_col = None
    for cand in ["hour", "slot"]:
        if cand in frame.columns:
            hour_col = cand
            break

    alt_candidates = [
        "pv_roll_24h_mean_norm",
        "pv_rolling_24h_mean_norm",
        "pv_roll_7d_same_hour_mean_norm",
        "pv_rolling_7d_same_hour_mean_norm",
        "pv_lag_168h_norm",
        "pv_lag_72h_norm",
        "pv_lag_48h_norm",
        "pv_lag_24h_norm",
        "pv_lag_1h_norm",
        "pv_norm_kwh_per_kw",
        "predicted_norm_kwh_per_kw",
    ]

    for c in target_cols:
        frame[c] = _pd.to_numeric(frame[c], errors="coerce")

        if not frame[c].isna().any():
            continue

        # 1) 다른 PV lag/rolling feature가 있으면 먼저 사용
        for alt in alt_candidates:
            if alt in frame.columns and alt != c:
                frame[c] = frame[c].fillna(_pd.to_numeric(frame[alt], errors="coerce"))

        # 2) 같은 hour/slot 평균
        if hour_col is not None and frame[c].isna().any():
            frame[c] = frame[c].fillna(
                frame.groupby(hour_col)[c].transform("mean")
            )

        # 3) 전체 평균
        if frame[c].isna().any():
            frame[c] = frame[c].fillna(frame[c].mean())

        # 4) 그래도 없으면 보수적으로 0
        frame[c] = frame[c].fillna(0.0)

    return frame
# PATCH_PV_LAG_NAN_FALLBACK_END


SCHEDULING_DIR = Path(__file__).resolve().parents[1]
BASE = SCHEDULING_DIR
# OLD_BASE = Path(".")  # 기존 개별 모델 테스트용 기준 경로

MODEL_PATH = BASE / "models/pv/lgbm_pv_dayahead_v3_single_site.txt"
FEATURE_PATH = BASE / "models/pv/pv_dayahead_v3_single_site_feature_columns.json"
CONFIG_PATH = BASE / "models/pv/pv_dayahead_v3_single_site_config.json"
HARD_ZERO_SLOT_PATH = BASE / "models/pv/pv_v3_hard_zero_slots.json"

# 기존 저장 경로는 개별 모델 테스트 때 참고용으로만 남김.
# OLD_OUT_JSON = BASE / "outputs/pv_prediction_v3_from_backend_json.json"
# OLD_OUT_CSV = BASE / "outputs/pv_prediction_v3_from_backend_json.csv"
OUT_JSON = BASE / "output/manual_test/pv/pv_prediction_v3_from_backend_json.json"
OUT_CSV = BASE / "output/manual_test/pv/pv_prediction_v3_from_backend_json.csv"

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# 백엔드 JSON에는 발전소 정보를 안 넣는다.
# AI 서버 내부 고정값.
MODEL_CAPACITY_KW = 50.0


def parse_dt(value):
    t = pd.Timestamp(value)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Seoul").tz_localize(None)
    return t


def to_float(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip()
        if v in ["", "없음", "강수없음", "적설없음", "-"]:
            return 0.0
        v = v.replace("mm", "").replace("cm", "").strip()
        try:
            return float(v)
        except ValueError:
            return default
    try:
        return float(value)
    except Exception:
        return default


def slot_label(slot: int) -> str:
    end = "24:00" if slot == 23 else f"{slot + 1:02d}:00"
    return f"{slot:02d}:00~{end}"


def normalize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 실제 API나 백엔드 필드명이 조금 다를 때 대응
    rename_map = {}
    if "si" in df.columns and "icsr" not in df.columns:
        rename_map["si"] = "icsr"
    if "cld" in df.columns and "dc10Tca" not in df.columns:
        rename_map["cld"] = "dc10Tca"
    df = df.rename(columns=rename_map)

    required = ["tm", "ta", "rn", "ws", "wd", "hm", "pa", "ps", "ss", "icsr", "dc10Tca"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"past_weather_hourly missing fields: {missing}")

    df["tm"] = df["tm"].apply(parse_dt)

    for col in ["ta", "rn", "ws", "wd", "hm", "pa", "ps", "ss", "icsr", "dc10Tca"]:
        df[col] = df[col].apply(to_float)

    for col in ["rn", "ss", "icsr"]:
        df[col] = df[col].fillna(0.0)

    df = df.sort_values("tm").reset_index(drop=True)
    return df


def build_obs_features(weather_df: pd.DataFrame) -> dict:
    """
    학습 코드와 동일하게 최근 168개 시간자료를 사용.
    입력은 D-7 00:00 ~ D 22:00까지 올 수 있으므로 tail(168)을 사용한다.
    """
    w = weather_df.sort_values("tm").copy()
    if len(w) < 168:
        raise ValueError(f"past_weather_hourly must have at least 168 rows, got {len(w)}")

    last24 = w.tail(24)
    last72 = w.tail(72)
    last168 = w.tail(168)

    return {
        "obs_ta_mean_7d": last168["ta"].mean(),
        "obs_ta_max_7d": last168["ta"].max(),
        "obs_ta_min_7d": last168["ta"].min(),
        "obs_hm_mean_7d": last168["hm"].mean(),
        "obs_ws_mean_7d": last168["ws"].mean(),
        "obs_rn_sum_7d": last168["rn"].sum(),

        "obs_si_sum_1d": last24["icsr"].sum(),
        "obs_si_sum_3d": last72["icsr"].sum(),
        "obs_si_sum_7d": last168["icsr"].sum(),
        "obs_si_mean_7d": last168["icsr"].mean(),
        "obs_si_max_7d": last168["icsr"].max(),

        "obs_ss_sum_1d": last24["ss"].sum(),
        "obs_ss_sum_3d": last72["ss"].sum(),
        "obs_ss_sum_7d": last168["ss"].sum(),

        "obs_cld_mean_7d": last168["dc10Tca"].mean(),
        "obs_cld_max_7d": last168["dc10Tca"].max(),
    }


def normalize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["tmef", "TMP", "POP", "PTY", "PCP", "REH", "SKY", "WSD", "VEC", "UUU", "VVV", "TMN", "TMX"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"forecast_short_term_3h missing fields: {missing}")

    df["tmef"] = df["tmef"].apply(parse_dt)

    for col in ["TMP", "POP", "PTY", "PCP", "REH", "SKY", "WSD", "VEC", "UUU", "VVV", "TMN", "TMX"]:
        df[col] = df[col].apply(to_float)

    df = df.sort_values("tmef").drop_duplicates("tmef", keep="last").reset_index(drop=True)
    return df


def forecast_to_hourly(forecast_df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    3시간 단위 예보를 1시간 단위 24개 row로 변환.
    연속형 변수는 보간, 코드/강수 계열은 forward-fill.
    D+2의 22, 23시는 21시 값을 forward-fill한다.
    """
    target_date = pd.Timestamp(target_date).normalize()
    target_index = pd.date_range(target_date, periods=24, freq="h")

    f = forecast_df.copy()
    f = f.set_index("tmef").sort_index()

    union_index = f.index.union(target_index).sort_values()
    out = f.reindex(union_index)

    continuous_cols = ["TMP", "REH", "WSD", "VEC", "UUU", "VVV", "TMN", "TMX"]
    step_cols = ["POP", "PTY", "PCP", "SKY"]

    out[continuous_cols] = out[continuous_cols].interpolate(method="time").ffill().bfill()
    out[step_cols] = out[step_cols].ffill().bfill()

    hourly = out.loc[target_index].copy()
    hourly = hourly.reset_index().rename(columns={"index": "tmef"})
    hourly["slot"] = hourly["tmef"].dt.hour

    if len(hourly) != 24:
        raise ValueError(f"forecast hourly conversion failed: {len(hourly)} rows")

    return hourly


def build_d2_features(d2_hourly: pd.DataFrame) -> dict:
    return {
        "d2_TMP_mean": d2_hourly["TMP"].mean(),
        "d2_TMP_max": d2_hourly["TMP"].max(),
        "d2_TMP_min": d2_hourly["TMP"].min(),
        "d2_POP_mean": d2_hourly["POP"].mean(),
        "d2_PCP_sum": d2_hourly["PCP"].sum(),
        "d2_REH_mean": d2_hourly["REH"].mean(),
        "d2_WSD_mean": d2_hourly["WSD"].mean(),
        "d2_SKY_mean": d2_hourly["SKY"].mean(),
        "d2_rain_hours": int(((d2_hourly["PCP"] > 0) | (d2_hourly["PTY"] > 0)).sum()),
        "d2_cloudy_hours": int((d2_hourly["SKY"] >= 3).sum()),
    }


def normalize_pv_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["tm", "gen_kwh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"past_pv_hourly missing fields: {missing}")

    df["tm"] = df["tm"].apply(parse_dt)
    df["gen_kwh"] = df["gen_kwh"].apply(to_float)
    df["target_norm_kwh_per_kw"] = df["gen_kwh"] / MODEL_CAPACITY_KW
    df["date"] = df["tm"].dt.normalize()
    df["slot"] = df["tm"].dt.hour

    df = df.sort_values("tm").reset_index(drop=True)
    return df


def make_pv_lookup(pv_df: pd.DataFrame):
    lookup = {}
    for row in pv_df.itertuples(index=False):
        date_str = pd.Timestamp(getattr(row, "date")).strftime("%Y-%m-%d")
        slot = int(getattr(row, "slot"))
        norm = float(getattr(row, "target_norm_kwh_per_kw"))
        lookup[(date_str, slot)] = norm

    def get_norm(date_ts, slot):
        key = (pd.Timestamp(date_ts).strftime("%Y-%m-%d"), int(slot))
        return lookup.get(key, np.nan)

    return get_norm


def build_feature_rows(payload: dict, feature_cols: list) -> pd.DataFrame:
    request_timestamp = parse_dt(payload["request_timestamp"])
    run_date = request_timestamp.normalize()

    if "schedule_target_date" in payload:
        target_date = pd.Timestamp(payload["schedule_target_date"]).normalize()
    else:
        target_date = run_date + pd.Timedelta(days=1)

    d2_date = target_date + pd.Timedelta(days=1)

    weather_df = normalize_weather_columns(pd.DataFrame(payload["past_weather_hourly"]))
    forecast_rows = payload.get("forecast_short_term_hourly")
    if forecast_rows is None:
        forecast_rows = payload.get("forecast_short_term_3h")
    if forecast_rows is None:
        raise ValueError("forecast_short_term_hourly or forecast_short_term_3h is required")
    forecast_df = normalize_forecast_columns(pd.DataFrame(forecast_rows))
    pv_df = normalize_pv_columns(pd.DataFrame(payload["past_pv_hourly"]))

    obs_feat = build_obs_features(weather_df)

    d1_hourly = forecast_to_hourly(forecast_df, target_date)
    d2_hourly = forecast_to_hourly(forecast_df, d2_date)

    d2_feat = build_d2_features(d2_hourly)

    d1_tmn = d1_hourly["TMN"].min()
    d1_tmx = d1_hourly["TMX"].max()

    get_pv_norm = make_pv_lookup(pv_df)

    # 최근 24시간: D-1 22~24 + D 00~22
    recent24_vals = []
    for h in [22, 23]:
        recent24_vals.append(get_pv_norm(run_date - pd.Timedelta(days=1), h))
    for h in range(0, 22):
        recent24_vals.append(get_pv_norm(run_date, h))
    recent24_vals = np.array(recent24_vals, dtype=float)

    # 사용 가능한 전체 과거 발전량 평균
    available_vals = pv_df["target_norm_kwh_per_kw"].to_numpy(dtype=float)

    rows = []

    for slot in range(24):
        f = d1_hourly[d1_hourly["slot"] == slot]
        if len(f) == 0:
            raise ValueError(f"D+1 forecast missing for slot {slot}")
        f = f.iloc[0]

        if slot <= 21:
            lag24 = get_pv_norm(run_date, slot)
        else:
            lag24 = get_pv_norm(run_date - pd.Timedelta(days=1), slot)

        row = {
            "target_year": target_date.year,
            "target_month": target_date.month,
            "target_day": target_date.day,
            "slot": slot,

            "d1_TMP": float(f["TMP"]),
            "d1_POP": float(f["POP"]),
            "d1_PTY": float(f["PTY"]),
            "d1_PCP": float(f["PCP"]),
            "d1_REH": float(f["REH"]),
            "d1_SKY": float(f["SKY"]),
            "d1_WSD": float(f["WSD"]),
            "d1_VEC": float(f["VEC"]),
            "d1_UUU": float(f["UUU"]),
            "d1_VVV": float(f["VVV"]),
            "d1_TMN": float(d1_tmn),
            "d1_TMX": float(d1_tmx),

            "pv_lag_24h_or_fallback_norm": lag24,
            "pv_lag_48h_norm": get_pv_norm(target_date - pd.Timedelta(days=2), slot),
            "pv_lag_72h_norm": get_pv_norm(target_date - pd.Timedelta(days=3), slot),
            "pv_lag_168h_norm": get_pv_norm(target_date - pd.Timedelta(days=7), slot),
            "pv_roll_past_24h_mean_norm": float(np.nanmean(recent24_vals)),
            "pv_roll_available_mean_norm": float(np.nanmean(available_vals)),
        }

        row.update(obs_feat)
        row.update(d2_feat)
        rows.append(row)

    pred_df = pd.DataFrame(rows)

    missing_cols = [c for c in feature_cols if c not in pred_df.columns]
    if missing_cols:
        raise ValueError(f"missing feature columns: {missing_cols}")

    missing_values = pred_df[feature_cols].isna().sum()
    missing_values = missing_values[missing_values > 0]

    if len(missing_values) > 0:
        print(f"[WARN] feature has NaN values before PV fallback:\n{missing_values}")
    pred_df = _patch_fill_pv_lag_nan(pred_df, feature_cols)

    missing_values = pred_df[feature_cols].isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        raise ValueError(f"feature still has NaN values after PV fallback:\n{missing_values}")

    return pred_df, target_date, run_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="output/manual_test/preprocessed/pv_backend_input.json",
        help="Preprocessed PV JSON input path"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run output directory. If omitted, writes to scheduling/output/manual_test/pv"
    )
    args = parser.parse_args()

    global OUT_JSON, OUT_CSV
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        OUT_JSON = out_dir / "pv_prediction_v3_from_backend_json.json"
        OUT_CSV = out_dir / "pv_prediction_v3_from_backend_json.csv"
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

    model = lgb.Booster(model_file=str(MODEL_PATH))

    pred_df, target_date, run_date = build_feature_rows(payload, feature_cols)

    pred_norm = model.predict(pred_df[feature_cols])
    pred_norm = np.clip(pred_norm, 0.0, 1.0)

    # Hard-zero physical constraint:
    # 실제 학습 데이터에서 한 번도 발전량이 발생하지 않은 시간대는 0으로 강제 보정한다.
    hard_zero_slots = []
    if HARD_ZERO_SLOT_PATH.exists():
        with open(HARD_ZERO_SLOT_PATH, "r", encoding="utf-8") as f:
            hard_zero_config = json.load(f)
        hard_zero_slots = hard_zero_config.get("hard_zero_slots", [])

    if hard_zero_slots:
        hard_zero_mask = pred_df["slot"].isin(hard_zero_slots).to_numpy()
        pred_norm[hard_zero_mask] = 0.0

    pred_kwh = pred_norm * MODEL_CAPACITY_KW

    rows = []
    for i, row in pred_df.iterrows():
        slot = int(row["slot"])
        slot_start = target_date + pd.Timedelta(hours=slot)
        slot_end = slot_start + pd.Timedelta(hours=1)

        rows.append({
            "slot": slot,
            "slot_label": slot_label(slot),
            "slot_start": slot_start.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "slot_end": slot_end.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "predicted_norm_kwh_per_kw": float(pred_norm[i]),
            "predicted_pv_kwh": float(pred_kwh[i])
        })

    output = {
        "request_id": payload.get("request_id"),
        "model_version": "v3_single_site",
        "input_mode": "backend_json_without_pv_site",
        "run_timestamp": payload.get("request_timestamp"),
        "run_date": run_date.strftime("%Y-%m-%d"),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "horizon_hours": 24,
        "slot_definition": "slot 0 = 00:00~01:00, slot 23 = 23:00~24:00",
        "ai_server_fixed_config": {
            "model_capacity_kw": MODEL_CAPACITY_KW,
            "site_info_required_from_backend": False,
            "training_site": config.get("site_name_for_training", "fixed single-site model")
        },
        "data_policy_applied": {
            "lag24_fallback_for_slot_22_23": True,
            "fallback_rule": "slot 22 and 23 use previous-day same-slot PV because D 22:00~23:00 and D 23:00~24:00 are unavailable at 22:10"
        },
        "predictions": rows
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print("[DONE] JSON:", OUT_JSON)
    print("[DONE] CSV :", OUT_CSV)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
