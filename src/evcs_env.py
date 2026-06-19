#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gymnasium environment for EV multi-station day-ahead scheduling.

Supports two modes:

1) Single-day mode
   - input_csv = data/processed/rl_day_ahead_input.csv
   - use_scenarios = False
   - Used for deterministic inference/evaluation on one day-ahead input.

2) Scenario mode
   - input_csv = data/processed/rl_scenarios_dataset.csv
   - use_scenarios = True
   - reset() randomly selects one scenario_id.
   - Used for SAC training generalization.

Version v2 environment:
- 5 stations
- 24 hourly steps
- Continuous action:
    action[0:5]  : ESS charge/discharge ratio per station, [-1, 1]
                   negative = discharge, positive = charge
    action[5:10] : PV priority raw action, [-1, 1]
                   mapped to [0, 1]
                   0.0 = PV -> EV first
                   1.0 = PV -> ESS first

- Transfer is NOT an RL action in this environment.
  Transfer is applied later as rule-based post-processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_DIR / "data/processed/rl_day_ahead_input.csv"
DEFAULT_SCENARIO_CSV = PROJECT_DIR / "data/processed/rl_scenarios_dataset.csv"


class MultiStationEVSchedulerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        input_csv: str | Path = DEFAULT_INPUT_CSV,
        use_scenarios: bool | None = None,
        fixed_scenario_id: int | None = None,
        random_initial_soc: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.input_csv = Path(input_csv)
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {self.input_csv}")

        self.df_all = pd.read_csv(self.input_csv).sort_values(
            [c for c in ["scenario_id", "slot", "station_id"] if c in pd.read_csv(self.input_csv, nrows=1).columns]
        ).reset_index(drop=True)

        self.required_columns = [
            "slot",
            "hour",
            "station_id",
            "station_name",
            "pv_kwh",
            "demand_kwh",
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
        self._validate_required_columns(self.df_all)

        if use_scenarios is None:
            self.use_scenarios = "scenario_id" in self.df_all.columns
        else:
            self.use_scenarios = bool(use_scenarios)

        if self.use_scenarios and "scenario_id" not in self.df_all.columns:
            raise ValueError("use_scenarios=True but input CSV has no scenario_id column.")

        self.fixed_scenario_id = fixed_scenario_id
        self.random_initial_soc = random_initial_soc
        self.rng = np.random.default_rng(seed)

        if self.use_scenarios:
            self.scenario_ids = sorted(self.df_all["scenario_id"].unique().astype(int).tolist())
            if not self.scenario_ids:
                raise ValueError("No scenario_id values found.")

            if self.fixed_scenario_id is not None and self.fixed_scenario_id not in self.scenario_ids:
                raise ValueError(
                    f"fixed_scenario_id={self.fixed_scenario_id} not found in scenario_ids."
                )

            first_scenario = self.fixed_scenario_id if self.fixed_scenario_id is not None else self.scenario_ids[0]
            self.df = self._scenario_df(first_scenario)
            self.current_scenario_id = int(first_scenario)
            self.current_scenario_label = self._get_scenario_label(self.df)
        else:
            self.scenario_ids = []
            self.df = self.df_all.sort_values(["slot", "station_id"]).reset_index(drop=True)
            self.current_scenario_id = None
            self.current_scenario_label = None

        self.station_ids = sorted(self.df["station_id"].unique().astype(int).tolist())
        self.station_count = len(self.station_ids)
        self.horizon_hours = int(self.df["slot"].nunique())

        if self.station_count != 5:
            raise ValueError(f"Expected 5 stations, got {self.station_count}")

        if self.horizon_hours != 24:
            raise ValueError(f"Expected 24 slots, got {self.horizon_hours}")

        self._validate_episode_df(self.df)

        # Normalization constants from full dataset
        self.max_tou_price = max(1.0, float(self.df_all["tou_price"].max()))
        self.max_pv_kwh = max(1.0, float(self.df_all["pv_kwh"].max()))
        self.max_demand_kwh = max(1.0, float(self.df_all["demand_kwh"].max()))
        self.max_grid_limit = max(1.0, float(self.df_all["grid_limit_kw"].max()))

        # Episode constants, refreshed on reset when scenario changes
        self.soc_min = 0.1
        self.soc_max = 0.9
        self.initial_soc = 0.5
        self.ess_capacity_kwh = 100.0
        self.ess_max_charge_kw = 50.0
        self.ess_max_discharge_kw = 50.0
        self.eta_c = 0.95
        self.eta_d = 0.95
        self.grid_limit_kw = 200.0

        self._refresh_episode_constants()

        # action: 5 ESS ratios + 5 PV priority raw values
        self.action_dim = self.station_count * 2
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # observation: 3 common + station_count * 3
        # [hour_norm, tou_price_norm, grid_limit_norm,
        #  station0_soc, station0_pv_norm, station0_demand_norm, ...]
        self.obs_dim = 3 + self.station_count * 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.current_slot = 0
        self.soc_energy = np.zeros(self.station_count, dtype=np.float64)
        self.episode_records: list[dict[str, Any]] = []

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _scenario_df(self, scenario_id: int) -> pd.DataFrame:
        sdf = self.df_all[self.df_all["scenario_id"].astype(int) == int(scenario_id)].copy()
        if sdf.empty:
            raise ValueError(f"Scenario {scenario_id} not found.")
        return sdf.sort_values(["slot", "station_id"]).reset_index(drop=True)

    def _get_scenario_label(self, df: pd.DataFrame) -> str | None:
        if "scenario_label" not in df.columns:
            return None
        labels = df["scenario_label"].dropna().unique().tolist()
        return str(labels[0]) if labels else None

    def _validate_episode_df(self, df: pd.DataFrame) -> None:
        station_count = int(df["station_id"].nunique())
        slot_count = int(df["slot"].nunique())

        if station_count != 5:
            raise ValueError(f"Episode must have 5 stations, got {station_count}")

        if slot_count != 24:
            raise ValueError(f"Episode must have 24 slots, got {slot_count}")

        rows = len(df)
        expected_rows = station_count * slot_count
        if rows != expected_rows:
            raise ValueError(f"Episode rows must be {expected_rows}, got {rows}")

        rows_per_slot = df.groupby("slot").size()
        if not (rows_per_slot == station_count).all():
            bad = rows_per_slot[rows_per_slot != station_count]
            raise ValueError(f"Each slot must have {station_count} station rows. Bad slots: {bad}")

    def _refresh_episode_constants(self) -> None:
        first = self.df.iloc[0]

        self.soc_min = float(first["soc_min"])
        self.soc_max = float(first["soc_max"])
        self.initial_soc = float(first["soc_init"])

        self.ess_capacity_kwh = float(first["ess_capacity_kwh"])
        self.ess_max_charge_kw = float(first["ess_max_charge_kw"])
        self.ess_max_discharge_kw = float(first["ess_max_discharge_kw"])
        self.eta_c = float(first["ess_charge_efficiency"])
        self.eta_d = float(first["ess_discharge_efficiency"])
        self.grid_limit_kw = float(first["grid_limit_kw"])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_slot = 0
        self.episode_records = []

        # Select scenario for this episode
        if self.use_scenarios:
            if options and "scenario_id" in options:
                scenario_id = int(options["scenario_id"])
            elif self.fixed_scenario_id is not None:
                scenario_id = int(self.fixed_scenario_id)
            else:
                scenario_id = int(self.rng.choice(self.scenario_ids))

            self.df = self._scenario_df(scenario_id)
            self._validate_episode_df(self.df)

            self.current_scenario_id = scenario_id
            self.current_scenario_label = self._get_scenario_label(self.df)
            self._refresh_episode_constants()

            # In scenario mode, soc_init is already scenario-specific.
            init_soc_series = (
                self.df.sort_values(["station_id", "slot"])
                .groupby("station_id")["soc_init"]
                .first()
                .reindex(self.station_ids)
            )
            init_soc = init_soc_series.to_numpy(dtype=np.float64)

        else:
            self.df = self.df_all.sort_values(["slot", "station_id"]).reset_index(drop=True)
            self._validate_episode_df(self.df)
            self.current_scenario_id = None
            self.current_scenario_label = None
            self._refresh_episode_constants()

            if self.random_initial_soc:
                init_soc = self.rng.uniform(self.soc_min, self.soc_max, size=self.station_count)
            else:
                init_soc = np.full(self.station_count, self.initial_soc, dtype=np.float64)

        init_soc = np.clip(init_soc, self.soc_min, self.soc_max)
        self.soc_energy = init_soc * self.ess_capacity_kwh

        obs = self._get_observation()

        info = {
            "initial_soc": init_soc.tolist(),
            "use_scenarios": self.use_scenarios,
            "scenario_id": self.current_scenario_id,
            "scenario_label": self.current_scenario_label,
        }

        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        if action.shape != (self.action_dim,):
            raise ValueError(f"Action shape must be {(self.action_dim,)}, got {action.shape}")

        slot_df = self._slot_df(self.current_slot)

        ess_ratio = action[: self.station_count]
        pv_priority_raw = action[self.station_count :]
        pv_priority = (pv_priority_raw + 1.0) / 2.0
        pv_priority = np.clip(pv_priority, 0.0, 1.0)

        records = []
        cluster_grid_usage = 0.0
        total_cost = 0.0
        total_self_supply = 0.0
        total_pv_lost = 0.0
        total_soc_violation = 0.0

        slot_grid_limit = float(slot_df["grid_limit_kw"].iloc[0])

        for local_idx, (_, r) in enumerate(slot_df.iterrows()):
            station_id = int(r["station_id"])
            hour = int(r["hour"])
            tou_price = float(r["tou_price"])
            demand = max(0.0, float(r["demand_kwh"]))
            pv = max(0.0, float(r["pv_kwh"]))

            soc_before = self.soc_energy[local_idx] / self.ess_capacity_kwh

            raw_ratio = float(ess_ratio[local_idx])
            priority = float(pv_priority[local_idx])

            # Convert ESS action ratio to physical command.
            # positive = charge input kWh in this 1-hour slot
            # negative = discharge output kWh in this 1-hour slot
            desired_charge = max(0.0, raw_ratio) * self.ess_max_charge_kw
            desired_discharge = max(0.0, -raw_ratio) * self.ess_max_discharge_kw

            pv_to_ev = 0.0
            pv_to_ess = 0.0
            grid_to_ev = 0.0
            grid_to_ess = 0.0
            ess_to_ev = 0.0
            pv_lost = 0.0

            remaining_demand = demand
            remaining_pv = pv

            # Case A: PV priority to EV
            if priority < 0.5:
                pv_to_ev = min(remaining_pv, remaining_demand)
                remaining_pv -= pv_to_ev
                remaining_demand -= pv_to_ev

                # ESS discharge to EV if action asks discharge
                if desired_discharge > 0 and remaining_demand > 0:
                    available_discharge_output = max(
                        0.0,
                        (self.soc_energy[local_idx] - self.soc_min * self.ess_capacity_kwh) * self.eta_d,
                    )
                    ess_to_ev = min(remaining_demand, desired_discharge, available_discharge_output)
                    self.soc_energy[local_idx] -= ess_to_ev / self.eta_d
                    remaining_demand -= ess_to_ev

                # ESS charge if action asks charge, using remaining PV first
                if desired_charge > 0:
                    charge_room_input = max(
                        0.0,
                        (self.soc_max * self.ess_capacity_kwh - self.soc_energy[local_idx]) / self.eta_c,
                    )
                    pv_to_ess = min(remaining_pv, desired_charge, charge_room_input)
                    self.soc_energy[local_idx] += pv_to_ess * self.eta_c
                    remaining_pv -= pv_to_ess

                    remaining_charge = max(0.0, desired_charge - pv_to_ess)
                    charge_room_input = max(
                        0.0,
                        (self.soc_max * self.ess_capacity_kwh - self.soc_energy[local_idx]) / self.eta_c,
                    )
                    grid_to_ess = min(remaining_charge, charge_room_input)
                    self.soc_energy[local_idx] += grid_to_ess * self.eta_c

                grid_to_ev = max(0.0, remaining_demand)
                pv_lost = max(0.0, remaining_pv)

            # Case B: PV priority to ESS
            else:
                # ESS charge first, using PV first
                if desired_charge > 0:
                    charge_room_input = max(
                        0.0,
                        (self.soc_max * self.ess_capacity_kwh - self.soc_energy[local_idx]) / self.eta_c,
                    )
                    pv_to_ess = min(remaining_pv, desired_charge, charge_room_input)
                    self.soc_energy[local_idx] += pv_to_ess * self.eta_c
                    remaining_pv -= pv_to_ess

                    remaining_charge = max(0.0, desired_charge - pv_to_ess)
                    charge_room_input = max(
                        0.0,
                        (self.soc_max * self.ess_capacity_kwh - self.soc_energy[local_idx]) / self.eta_c,
                    )
                    grid_to_ess = min(remaining_charge, charge_room_input)
                    self.soc_energy[local_idx] += grid_to_ess * self.eta_c

                # Remaining PV to EV
                pv_to_ev = min(remaining_pv, remaining_demand)
                remaining_pv -= pv_to_ev
                remaining_demand -= pv_to_ev

                # ESS discharge to EV if action asks discharge
                if desired_discharge > 0 and remaining_demand > 0:
                    available_discharge_output = max(
                        0.0,
                        (self.soc_energy[local_idx] - self.soc_min * self.ess_capacity_kwh) * self.eta_d,
                    )
                    ess_to_ev = min(remaining_demand, desired_discharge, available_discharge_output)
                    self.soc_energy[local_idx] -= ess_to_ev / self.eta_d
                    remaining_demand -= ess_to_ev

                grid_to_ev = max(0.0, remaining_demand)
                pv_lost = max(0.0, remaining_pv)

            # Safety clamp for numerical drift
            soc_after_raw = self.soc_energy[local_idx] / self.ess_capacity_kwh
            soc_violation = max(0.0, self.soc_min - soc_after_raw) + max(0.0, soc_after_raw - self.soc_max)
            total_soc_violation += soc_violation

            soc_after = float(np.clip(soc_after_raw, self.soc_min, self.soc_max))
            self.soc_energy[local_idx] = soc_after * self.ess_capacity_kwh

            grid_usage = grid_to_ev + grid_to_ess
            self_supply = pv_to_ev + pv_to_ess + ess_to_ev

            cluster_grid_usage += grid_usage
            total_cost += grid_usage * tou_price
            total_self_supply += self_supply
            total_pv_lost += pv_lost

            ess_power_signed = (pv_to_ess + grid_to_ess) - ess_to_ev
            if ess_power_signed > 1e-9:
                ess_mode = "charge"
            elif ess_power_signed < -1e-9:
                ess_mode = "discharge"
            else:
                ess_mode = "idle"

            records.append(
                {
                    "scenario_id": self.current_scenario_id,
                    "scenario_label": self.current_scenario_label,
                    "slot": self.current_slot,
                    "hour": hour,
                    "station_id": station_id,
                    "station_name": r["station_name"],
                    "tou_price": tou_price,
                    "demand_kwh": demand,
                    "pv_kwh": pv,
                    "soc_before": soc_before,
                    "soc_after": soc_after,
                    "ess_ratio": raw_ratio,
                    "pv_priority": priority,
                    "ess_mode": ess_mode,
                    "ess_power_kwh_signed": ess_power_signed,
                    "grid_usage_kwh": grid_usage,
                    "pv_to_ev_kwh": pv_to_ev,
                    "pv_to_ess_kwh": pv_to_ess,
                    "ess_to_ev_kwh": ess_to_ev,
                    "grid_to_ev_kwh": grid_to_ev,
                    "grid_to_ess_kwh": grid_to_ess,
                    "self_supply_kwh": self_supply,
                    "pv_lost_kwh": pv_lost,
                    "grid_limit_kw": slot_grid_limit,
                }
            )

        peak_violation = max(0.0, cluster_grid_usage - slot_grid_limit)

        # Reward normalization.
        cost_term = total_cost / 10000.0
        self_supply_term = total_self_supply / 100.0
        pv_loss_term = total_pv_lost / 100.0
        peak_term = peak_violation / 100.0
        soc_term = total_soc_violation * 100.0

        reward = (
            -1.0 * cost_term
            + 0.2 * self_supply_term
            - 0.5 * pv_loss_term
            - 5.0 * peak_term
            - 10.0 * soc_term
        )

        self.episode_records.extend(records)

        self.current_slot += 1
        terminated = self.current_slot >= self.horizon_hours
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)

        info = {
            "slot": self.current_slot - 1,
            "scenario_id": self.current_scenario_id,
            "scenario_label": self.current_scenario_label,
            "cluster_grid_usage_kwh": cluster_grid_usage,
            "grid_limit_kw": slot_grid_limit,
            "total_cost_krw": total_cost,
            "total_self_supply_kwh": total_self_supply,
            "total_pv_lost_kwh": total_pv_lost,
            "peak_violation_kwh": peak_violation,
            "soc_violation": total_soc_violation,
            "records": records,
        }

        return obs, float(reward), terminated, truncated, info

    def _slot_df(self, slot: int) -> pd.DataFrame:
        sdf = self.df[self.df["slot"] == slot].sort_values("station_id").reset_index(drop=True)
        if len(sdf) != self.station_count:
            raise ValueError(f"Slot {slot} must have {self.station_count} station rows, got {len(sdf)}")
        return sdf

    def _get_observation(self) -> np.ndarray:
        slot = min(self.current_slot, self.horizon_hours - 1)
        sdf = self._slot_df(slot)

        hour = float(sdf["hour"].iloc[0])
        tou_price = float(sdf["tou_price"].iloc[0])
        grid_limit = float(sdf["grid_limit_kw"].iloc[0])

        obs = [
            hour / 23.0,
            tou_price / self.max_tou_price,
            grid_limit / self.max_grid_limit,
        ]

        for local_idx, (_, r) in enumerate(sdf.iterrows()):
            soc = self.soc_energy[local_idx] / self.ess_capacity_kwh
            pv_norm = float(r["pv_kwh"]) / self.max_pv_kwh
            demand_norm = float(r["demand_kwh"]) / self.max_demand_kwh

            obs.extend(
                [
                    float(np.clip(soc, 0.0, 1.0)),
                    float(np.clip(pv_norm, 0.0, 1.0)),
                    float(np.clip(demand_norm, 0.0, 1.0)),
                ]
            )

        return np.asarray(obs, dtype=np.float32)

    def get_episode_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.episode_records)

    def render(self):
        if not self.episode_records:
            print("No records yet.")
            return

        last = self.episode_records[-self.station_count :]
        for r in last:
            print(
                f"slot={r['slot']} station={r['station_id']} "
                f"soc={r['soc_after']:.3f} grid={r['grid_usage_kwh']:.3f} "
                f"mode={r['ess_mode']}"
            )


def run_random_episode(env: MultiStationEVSchedulerEnv, reset_seed: int | None = None) -> pd.DataFrame:
    obs, info = env.reset(seed=reset_seed)
    print("reset info:", info)
    print("obs shape:", obs.shape)
    print("action shape:", env.action_space.shape)

    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    ep = env.get_episode_dataframe()
    print("episode rows:", len(ep))
    print("total reward:", total_reward)
    print("soc min:", ep["soc_after"].min())
    print("soc max:", ep["soc_after"].max())
    print("total grid usage:", ep["grid_usage_kwh"].sum())
    print("total cost:", (ep["grid_usage_kwh"] * ep["tou_price"]).sum())
    print(ep.head())
    print()
    return ep


if __name__ == "__main__":
    print("===== Single-day mode test =====")
    env_single = MultiStationEVSchedulerEnv(
        input_csv=DEFAULT_INPUT_CSV,
        use_scenarios=False,
        random_initial_soc=False,
        seed=42,
    )
    run_random_episode(env_single, reset_seed=42)

    if DEFAULT_SCENARIO_CSV.exists():
        print("===== Scenario mode test =====")
        env_scenario = MultiStationEVSchedulerEnv(
            input_csv=DEFAULT_SCENARIO_CSV,
            use_scenarios=True,
            random_initial_soc=False,
            seed=42,
        )
        run_random_episode(env_scenario, reset_seed=42)
