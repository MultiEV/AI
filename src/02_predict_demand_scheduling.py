from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import holidays

# PATCH_DEMAND_LAG_NAN_FALLBACK_START
def _patch_fill_demand_lag_nan_inplace(frame):
    """
    demand_lag_24h_or_fallback 등 수요 lag/rolling feature에 NaN이 남으면
    row rolling feature -> station+hour 평균 -> station 평균 -> hour 평균 -> 전체 평균 -> 0 순서로 채운다.
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

    frame.replace([_np.inf, -_np.inf], _np.nan, inplace=True)

    cols = []
    for c in frame.columns:
        lc = str(c).lower()
        if (
            str(c) == "demand_lag_24h_or_fallback"
            or "demand_lag" in lc
            or "fallback" in lc
            or lc.startswith("lag_")
            or lc.startswith("roll_")
            or "rolling" in lc
            or "roll_" in lc
        ):
            cols.append(c)

    if not cols:
        return frame

    station_col = None
    for cand in ["station_name", "station_id"]:
        if cand in frame.columns:
            station_col = cand
            break

    hour_col = None
    for cand in ["hour", "slot"]:
        if cand in frame.columns:
            hour_col = cand
            break

    for c in cols:
        frame[c] = _pd.to_numeric(frame[c], errors="coerce")

        if not frame[c].isna().any():
            continue

        if str(c) == "demand_lag_24h_or_fallback":
            for alt in [
                "roll_24h_mean",
                "rolling_24h_mean",
                "roll_6h_mean",
                "rolling_6h_mean",
                "roll_3h_mean",
                "rolling_3h_mean",
                "lag_168h",
                "lag_24h",
                "lag_1h",
            ]:
                if alt in frame.columns:
                    frame[c] = frame[c].fillna(_pd.to_numeric(frame[alt], errors="coerce"))

        if station_col is not None and hour_col is not None and frame[c].isna().any():
            frame[c] = frame[c].fillna(
                frame.groupby([station_col, hour_col])[c].transform("mean")
            )

        if station_col is not None and frame[c].isna().any():
            frame[c] = frame[c].fillna(
                frame.groupby(station_col)[c].transform("mean")
            )

        if hour_col is not None and frame[c].isna().any():
            frame[c] = frame[c].fillna(
                frame.groupby(hour_col)[c].transform("mean")
            )

        if frame[c].isna().any():
            frame[c] = frame[c].fillna(frame[c].mean())

        frame[c] = frame[c].fillna(0.0)

    return frame


def _patch_apply_demand_lag_nan_fallback():
    """
    함수 내부 로컬 변수 중 feature DataFrame 후보를 찾아 NaN fallback을 적용한다.
    """
    try:
        _local_vars = locals()
    except Exception:
        return

    for _name in [
        "X",
        "x",
        "feature_df",
        "features_df",
        "df_features",
        "predict_df",
        "pred_df",
        "df_pred",
        "input_df",
        "out_df",
        "df",
    ]:
        try:
            if _name in _local_vars:
                _patch_fill_demand_lag_nan_inplace(_local_vars[_name])
        except Exception as _e:
            print(f"[WARN] demand fallback skipped for {_name}: {_e}")
# PATCH_DEMAND_LAG_NAN_FALLBACK_END


SCHEDULING_DIR = Path(__file__).resolve().parents[1]
BASE = SCHEDULING_DIR
# OLD_BASE = Path(".")  # 기존 개별 모델 테스트용 기준 경로

MODEL_PATH = BASE / "models/demand/lgbm_ev_demand_v2.txt"
FEATURE_PATH = BASE / "models/demand/ev_demand_v2_feature_columns.json"
META_PATH = BASE / "models/demand/ev_demand_v2_station_metadata.json"

# 기존 저장 경로는 개별 모델 테스트 때 참고용으로만 남김.
# OLD_OUT_JSON = BASE / "outputs/ev_demand_prediction_v2_from_backend_json.json"
# OLD_OUT_CSV = BASE / "outputs/ev_demand_prediction_v2_from_backend_json.csv"
# OLD_OUT_CLUSTER_CSV = BASE / "outputs/ev_demand_prediction_v2_cluster_summary.csv"
OUT_JSON = BASE / "output/manual_test/demand/ev_demand_prediction_v2_from_backend_json.json"
OUT_CSV = BASE / "output/manual_test/demand/ev_demand_prediction_v2_from_backend_json.csv"
OUT_CLUSTER_CSV = BASE / "output/manual_test/demand/ev_demand_prediction_v2_cluster_summary.csv"

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


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

    rename_map = {}
    if "일시" in df.columns and "tm" not in df.columns:
        rename_map["일시"] = "tm"
    if "기온(°C)" in df.columns and "ta" not in df.columns:
        rename_map["기온(°C)"] = "ta"
    if "강수량(mm)" in df.columns and "rn" not in df.columns:
        rename_map["강수량(mm)"] = "rn"
    if "풍속(m/s)" in df.columns and "ws" not in df.columns:
        rename_map["풍속(m/s)"] = "ws"
    if "풍향(16방위)" in df.columns and "wd" not in df.columns:
        rename_map["풍향(16방위)"] = "wd"
    if "습도(%)" in df.columns and "hm" not in df.columns:
        rename_map["습도(%)"] = "hm"
    if "현지기압(hPa)" in df.columns and "pa" not in df.columns:
        rename_map["현지기압(hPa)"] = "pa"
    if "해면기압(hPa)" in df.columns and "ps" not in df.columns:
        rename_map["해면기압(hPa)"] = "ps"
    if "일조(hr)" in df.columns and "ss" not in df.columns:
        rename_map["일조(hr)"] = "ss"
    if "일사(MJ/m2)" in df.columns and "icsr" not in df.columns:
        rename_map["일사(MJ/m2)"] = "icsr"
    if "적설(cm)" in df.columns and "dsnw" not in df.columns:
        rename_map["적설(cm)"] = "dsnw"
    if "3시간신적설(cm)" in df.columns and "hr3Fhsc" not in df.columns:
        rename_map["3시간신적설(cm)"] = "hr3Fhsc"
    if "전운량(10분위)" in df.columns and "dc10Tca" not in df.columns:
        rename_map["전운량(10분위)"] = "dc10Tca"

    df = df.rename(columns=rename_map)

    required = [
        "tm", "ta", "rn", "ws", "wd", "hm", "pa", "ps",
        "ss", "icsr", "dsnw", "hr3Fhsc", "dc10Tca"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"past_weather_hourly missing fields: {missing}")

    df["tm"] = df["tm"].apply(parse_dt)

    for col in ["ta", "rn", "ws", "wd", "hm", "pa", "ps", "ss", "icsr", "dsnw", "hr3Fhsc", "dc10Tca"]:
        df[col] = df[col].apply(to_float)

    for col in ["rn", "ss", "icsr", "dsnw", "hr3Fhsc"]:
        df[col] = df[col].fillna(0.0)

    for col in ["ta", "ws", "wd", "hm", "pa", "ps", "dc10Tca"]:
        df[col] = df[col].interpolate(limit_direction="both")

    df = df.sort_values("tm").reset_index(drop=True)
    return df


def build_obs_weather_features(weather_df: pd.DataFrame) -> dict:
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
        "obs_snow_sum_7d": last168["dsnw"].sum(),
        "obs_new_snow_sum_7d": last168["hr3Fhsc"].sum(),
        "obs_cld_mean_7d": last168["dc10Tca"].mean(),
        "obs_cld_max_7d": last168["dc10Tca"].max(),

        "obs_ta_mean_1d": last24["ta"].mean(),
        "obs_rn_sum_1d": last24["rn"].sum(),
        "obs_snow_sum_1d": last24["dsnw"].sum(),
        "obs_new_snow_sum_1d": last24["hr3Fhsc"].sum(),

        "obs_ta_mean_3d": last72["ta"].mean(),
        "obs_rn_sum_3d": last72["rn"].sum(),
        "obs_snow_sum_3d": last72["dsnw"].sum(),
        "obs_new_snow_sum_3d": last72["hr3Fhsc"].sum(),
    }


def normalize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = [
        "tmef", "TMP", "POP", "PTY", "PCP", "SNO",
        "REH", "SKY", "WSD", "VEC", "UUU", "VVV", "TMN", "TMX"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"forecast_short_term_3h missing fields: {missing}")

    df["tmef"] = df["tmef"].apply(parse_dt)

    for col in ["TMP", "POP", "PTY", "PCP", "SNO", "REH", "SKY", "WSD", "VEC", "UUU", "VVV", "TMN", "TMX"]:
        df[col] = df[col].apply(to_float)

    df = df.sort_values("tmef").drop_duplicates("tmef", keep="last").reset_index(drop=True)
    return df


def forecast_to_hourly(forecast_df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    target_date = pd.Timestamp(target_date).normalize()
    target_index = pd.date_range(target_date, periods=24, freq="h")

    f = forecast_df.copy()
    f = f.set_index("tmef").sort_index()

    union_index = f.index.union(target_index).sort_values()
    out = f.reindex(union_index)

    continuous_cols = ["TMP", "REH", "WSD", "VEC", "UUU", "VVV", "TMN", "TMX"]
    step_cols = ["POP", "PTY", "PCP", "SNO", "SKY"]

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
        "d2_SNO_sum": d2_hourly["SNO"].sum(),
        "d2_REH_mean": d2_hourly["REH"].mean(),
        "d2_WSD_mean": d2_hourly["WSD"].mean(),
        "d2_SKY_mean": d2_hourly["SKY"].mean(),
        "d2_rain_hours": int(((d2_hourly["PCP"] > 0) | (d2_hourly["PTY"].isin([1, 2, 4]))).sum()),
        "d2_snow_hours": int(((d2_hourly["SNO"] > 0) | (d2_hourly["PTY"].isin([2, 3]))).sum()),
        "d2_cloudy_hours": int((d2_hourly["SKY"] >= 3).sum()),
    }


def normalize_demand_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["tm", "station_name", "demand_kwh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"past_demand_hourly missing fields: {missing}")

    df["tm"] = df["tm"].apply(parse_dt)
    df["demand_kwh"] = df["demand_kwh"].apply(to_float)
    df["date"] = df["tm"].dt.normalize()
    df["slot"] = df["tm"].dt.hour

    df = df.sort_values(["station_name", "tm"]).reset_index(drop=True)
    return df


def make_demand_lookup(demand_df: pd.DataFrame):
    lookup = {}

    for row in demand_df.itertuples(index=False):
        station = getattr(row, "station_name")
        date_str = pd.Timestamp(getattr(row, "date")).strftime("%Y-%m-%d")
        slot = int(getattr(row, "slot"))
        demand = float(getattr(row, "demand_kwh"))
        lookup[(station, date_str, slot)] = demand

    def get(station, date_ts, slot):
        key = (station, pd.Timestamp(date_ts).strftime("%Y-%m-%d"), int(slot))
        return lookup.get(key, np.nan)

    return get


def holiday_features(target_date: pd.Timestamp) -> dict:
    d = pd.Timestamp(target_date).normalize()
    years = [d.year - 1, d.year, d.year + 1]
    kr_holidays = holidays.KR(years=years)

    def is_holiday(date):
        return date.date() in kr_holidays

    def holiday_name(date):
        return str(kr_holidays.get(date.date(), ""))

    def is_lunar_like(date):
        name = holiday_name(date)
        keywords = ["Seollal", "Korean New Year", "Chuseok", "Lunar", "설날", "추석"]
        return any(k in name for k in keywords)

    holiday_dates = [pd.Timestamp(x) for x in kr_holidays.keys()]
    if holiday_dates:
        nearest = min(abs((h.normalize() - d).days) for h in holiday_dates)
    else:
        nearest = 999

    return {
        "is_weekend": int(d.dayofweek >= 5),
        "is_holiday": int(is_holiday(d)),
        "is_weekend_or_holiday": int(d.dayofweek >= 5 or is_holiday(d)),
        "is_lunar_holiday": int(is_lunar_like(d)),
        "is_lunar_prev_day": int(is_lunar_like(d + pd.Timedelta(days=1))),
        "is_lunar_next_day": int(is_lunar_like(d - pd.Timedelta(days=1))),
        "is_lunar_prev_week": int(any(is_lunar_like(d + pd.Timedelta(days=i)) for i in range(1, 8))),
        "is_lunar_next_week": int(any(is_lunar_like(d - pd.Timedelta(days=i)) for i in range(1, 8))),
        "is_before_holiday": int(is_holiday(d + pd.Timedelta(days=1))),
        "is_after_holiday": int(is_holiday(d - pd.Timedelta(days=1))),
        "days_to_nearest_holiday": float(nearest),
        "is_bridge_day": int(
            d.dayofweek < 5 and
            not is_holiday(d) and
            (is_holiday(d - pd.Timedelta(days=1)) or is_holiday(d + pd.Timedelta(days=1)))
        ),
        "is_vacation_season": int(d.month in [1, 7, 8, 12]),
    }


def build_feature_rows(payload: dict, feature_cols: list, meta: dict):
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
    demand_df = normalize_demand_columns(pd.DataFrame(payload["past_demand_hourly"]))

    obs_feat = build_obs_weather_features(weather_df)

    d1_hourly = forecast_to_hourly(forecast_df, target_date)
    d2_hourly = forecast_to_hourly(forecast_df, d2_date)

    d2_feat = build_d2_features(d2_hourly)
    d1_tmn = d1_hourly["TMN"].min()
    d1_tmx = d1_hourly["TMX"].max()

    get_demand = make_demand_lookup(demand_df)

    station_names = meta["station_names"]
    station_columns = meta["station_columns"]

    capacity_map = {
        s["station_name"]: float(s["charger_capacity_kw"])
        for s in meta["stations"]
    }

    holiday_feat = holiday_features(target_date)

    rows = []

    for station in station_names:
        recent24_vals = []
        for h in [22, 23]:
            recent24_vals.append(get_demand(station, run_date - pd.Timedelta(days=1), h))
        for h in range(0, 22):
            recent24_vals.append(get_demand(station, run_date, h))
        recent24_vals = np.array(recent24_vals, dtype=float)

        available_vals = demand_df[demand_df["station_name"] == station]["demand_kwh"].to_numpy(dtype=float)

        for slot in range(24):
            f = d1_hourly[d1_hourly["slot"] == slot]
            if len(f) == 0:
                raise ValueError(f"D+1 forecast missing for slot {slot}")
            f = f.iloc[0]

            if slot <= 21:
                lag24 = get_demand(station, run_date, slot)
            else:
                lag24 = get_demand(station, run_date - pd.Timedelta(days=1), slot)

            row = {
                "station_name": station,
                "target_year": target_date.year,
                "target_month": target_date.month,
                "target_day": target_date.day,
                "target_dayofweek": target_date.dayofweek,
                "slot": slot,
                "charger_capacity_kw": capacity_map[station],

                "d1_TMP": float(f["TMP"]),
                "d1_POP": float(f["POP"]),
                "d1_PTY": float(f["PTY"]),
                "d1_PCP": float(f["PCP"]),
                "d1_SNO": float(f["SNO"]),
                "d1_REH": float(f["REH"]),
                "d1_SKY": float(f["SKY"]),
                "d1_WSD": float(f["WSD"]),
                "d1_VEC": float(f["VEC"]),
                "d1_UUU": float(f["UUU"]),
                "d1_VVV": float(f["VVV"]),
                "d1_TMN": float(d1_tmn),
                "d1_TMX": float(d1_tmx),

                "demand_lag_24h_or_fallback": lag24,
                "demand_lag_48h": get_demand(station, target_date - pd.Timedelta(days=2), slot),
                "demand_lag_72h": get_demand(station, target_date - pd.Timedelta(days=3), slot),
                "demand_lag_168h": get_demand(station, target_date - pd.Timedelta(days=7), slot),
                "demand_lag_192h": get_demand(station, target_date - pd.Timedelta(days=8), slot),
                "demand_roll_past_24h_mean": float(np.nanmean(recent24_vals)),
                "demand_roll_available_mean": float(np.nanmean(available_vals)),
                "demand_roll_available_max": float(np.nanmax(available_vals)),
            }

            for s, col in zip(station_names, station_columns):
                row[col] = 1 if station == s else 0

            row.update(holiday_feat)
            row.update(obs_feat)
            row.update(d2_feat)
            rows.append(row)

    pred_df = pd.DataFrame(rows)

    # PATCH_DEMAND_LAG_NAN_FALLBACK_APPLY
    try:
        for _patch_name in [
            "X", "x", "feature_df", "features_df", "df_features",
            "predict_df", "pred_df", "df_pred", "input_df", "out_df", "df"
        ]:
            if _patch_name in locals():
                _patch_fill_demand_lag_nan_inplace(locals()[_patch_name])
    except Exception as _patch_e:
        print(f"[WARN] demand lag NaN fallback patch skipped: {_patch_e}")
    missing_cols = [c for c in feature_cols if c not in pred_df.columns]
    if missing_cols:
        raise ValueError(f"missing feature columns: {missing_cols}")

    missing_values = pred_df[feature_cols].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if len(missing_values) > 0:
        raise ValueError(f"feature has NaN values:\n{missing_values}")

    return pred_df, target_date, run_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="output/manual_test/preprocessed/demand_backend_input.json",
        help="Preprocessed demand JSON input path"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run output directory. If omitted, writes to scheduling/output/manual_test/demand"
    )
    args = parser.parse_args()

    global OUT_JSON, OUT_CSV, OUT_CLUSTER_CSV
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        OUT_JSON = out_dir / "ev_demand_prediction_v2_from_backend_json.json"
        OUT_CSV = out_dir / "ev_demand_prediction_v2_from_backend_json.csv"
        OUT_CLUSTER_CSV = out_dir / "ev_demand_prediction_v2_cluster_summary.csv"
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = lgb.Booster(model_file=str(MODEL_PATH))

    pred_df, target_date, run_date = build_feature_rows(payload, feature_cols, meta)

    # PATCH_DEMAND_LAG_NAN_FALLBACK_APPLY
    try:
        for _patch_name in [
            "X", "x", "feature_df", "features_df", "df_features",
            "predict_df", "pred_df", "df_pred", "input_df", "out_df", "df"
        ]:
            if _patch_name in locals():
                _patch_fill_demand_lag_nan_inplace(locals()[_patch_name])
    except Exception as _patch_e:
        print(f"[WARN] demand lag NaN fallback patch skipped: {_patch_e}")
    pred = model.predict(pred_df[feature_cols])
    pred = np.clip(pred, 0.0, None)

    rows = []
    for i, row in pred_df.iterrows():
        slot = int(row["slot"])
        slot_start = target_date + pd.Timedelta(hours=slot)
        slot_end = slot_start + pd.Timedelta(hours=1)

        rows.append({
            "station_name": row["station_name"],
            "slot": slot,
            "slot_label": slot_label(slot),
            "slot_start": slot_start.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "slot_end": slot_end.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "predicted_demand_kwh": float(pred[i])
        })

    result_df = pd.DataFrame(rows)
    cluster_df = (
        result_df.groupby(["slot", "slot_label", "slot_start", "slot_end"], as_index=False)
        ["predicted_demand_kwh"]
        .sum()
        .rename(columns={"predicted_demand_kwh": "cluster_predicted_demand_kwh"})
    )

    output = {
        "request_id": payload.get("request_id"),
        "model_version": "ev_demand_v2",
        "input_mode": "backend_json",
        "run_timestamp": payload.get("request_timestamp"),
        "run_date": run_date.strftime("%Y-%m-%d"),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "horizon_hours": 24,
        "station_count": len(meta["station_names"]),
        "target_charger_count_per_station": meta.get("target_charger_count", 5),
        "target_meaning": "hourly charging energy demand in kWh, scaled to 5 chargers per station",
        "slot_definition": "slot 0 = 00:00~01:00, slot 23 = 23:00~24:00",
        "predictions": rows,
        "cluster_summary": cluster_df.to_dict(orient="records"),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    result_df.to_csv(OUT_CSV, index=False)
    cluster_df.to_csv(OUT_CLUSTER_CSV, index=False)

    print("[DONE] JSON:", OUT_JSON)
    print("[DONE] CSV :", OUT_CSV)
    print("[DONE] CLUSTER CSV:", OUT_CLUSTER_CSV)
    print("\n[STATION PREDICTIONS HEAD]")
    print(result_df.head(30))
    print("\n[CLUSTER SUMMARY]")
    print(cluster_df)


if __name__ == "__main__":
    main()
