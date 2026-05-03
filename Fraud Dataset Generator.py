from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    # Population
    n_customers:              int   = 10000
    txns_per_customer_lambda: float = 10.0

    # Transaction amounts (log-normal, personal offsets)
    amount_mu_global:         float = 2.95
    amount_sigma:             float = 1.2
    amount_mu_personal_std:   float = 0.4

    # Arrival times (von Mises peak)
    time_peak_hour:           float = 13.0
    time_kappa:               float = 1.5
    time_personal_std_hr:     float = 3.0

    # Geography
    countries:                List[str]   = field(default_factory=lambda: ["A", "B", "C"])
    country_probs:            List[float] = field(default_factory=lambda: [0.85, 0.08, 0.07])
    home_country_loyalty:     float = 0.88
    new_country_window_h:     float = 336.0   # rolling window for foreign novelty (2 weeks)

    # Remote channel
    remote_marginal_p:        float = 0.36

    # Fraud model — intercept auto-calibrated at runtime for ~1% fraud rate
    fraud_intercept:          float = -9.3272

    w_amount:                 float = 1.75
    w_night:                  float = 2.25
    w_foreign:                float = 2.75
    w_velocity_1h:            float = 1.75
    w_velocity_24h:           float = 1.25
    w_gap_short:              float = 1.50
    w_remote:                 float = 2.50
    w_remote_x_foreign:       float = 1.00
    w_remote_x_night:         float = 1.00
    w_velocity_x_gap:         float = 0.75

    # Timing
    period_duration_hours:    float = 168.0   # 1 week per period
    n_warmup_periods:         int   = 20


# ---------------------------------------------------------------------------
# Customer history
# ---------------------------------------------------------------------------

class CustomerHistory:

    def __init__(self, home_country: str):
        self.home_country = home_country
        self.timestamps:  List[float] = []
        self.amounts:     List[float] = []
        self.countries:   List[str]   = []
        self._ts_arr: Optional[np.ndarray] = None
        self._dirty: bool = False

    def features(self, ts: float, amount: float, country: str,
                 window_h: float = 336.0) -> dict:
        if not self.timestamps:
            return dict(velocity_1h=0.0, velocity_24h=0.0,
                        amount_zscore=0.0, is_foreign=int(country != self.home_country),
                        hours_since_last=48.0)

        if self._dirty or self._ts_arr is None:
            self._ts_arr = np.asarray(self.timestamps)
            self._dirty = False
        arr  = self._ts_arr
        gap  = ts - self.timestamps[-1]
        v1h  = len(arr) - int(arr.searchsorted(ts - 1.0,  side="left"))
        v24h = len(arr) - int(arr.searchsorted(ts - 24.0, side="left"))

        log_amts = np.log(np.clip(self.amounts, 1e-6, None))
        z        = (np.log(max(amount, 1e-6)) - log_amts.mean()) / max(log_amts.std(), 0.25)

        # is_foreign: rolling window novelty; treat a transaction as foreign if the country
        # has not been seen recently.
        recent = set(self.countries[int(arr.searchsorted(ts - window_h, side="left")):])

        return dict(
            velocity_1h      = float(v1h),
            velocity_24h     = float(v24h),
            amount_zscore    = round(float(z), 4),
            is_foreign       = int(country not in recent),
            hours_since_last = round(max(gap, 0.0), 3),
        )

    def record(self, ts: float, amount: float, country: str):
        self.timestamps.append(ts)
        self.amounts.append(amount)
        self.countries.append(country)
        self._dirty = True


# ---------------------------------------------------------------------------
# Fraud scoring
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0))))


def calibrate_intercept(cfg: GeneratorConfig, target_rate: float = 0.01,
                        seed: int = 42) -> GeneratorConfig:
    """Binary search for fraud_intercept so mean P(fraud) ≈ target_rate."""
    rng = np.random.default_rng(seed)
    base_probe = replace(cfg, n_customers=2000)
    profiles = _build_profiles(base_probe, rng)
    hists = {p["customer_id"]: CustomerHistory(p["home_country"]) for p in profiles}
    _run_warmup(profiles, hists, base_probe, rng)

    logits_no_intercept = []
    for prof in profiles:
        hist = hists[prof["customer_id"]]
        n = max(1, int(rng.poisson(base_probe.txns_per_customer_lambda)))
        ts_arr = _sample_arrival_times(n, prof["time_peak_hr"], base_probe.time_kappa,
                                       base_probe.period_duration_hours, 0.0, rng)
        amounts = np.exp(rng.normal(prof["amount_mu"], base_probe.amount_sigma, n))
        ctrs = _sample_countries(base_probe, prof["home_idx"], n, rng)
        for i in range(n):
            ts, amount, ctry = float(ts_arr[i]), float(amounts[i]), ctrs[i]
            is_night = int(ts % 24.0 >= 22.0 or ts % 24.0 < 6.0)
            txn_f = hist.features(ts, amount, ctry, base_probe.new_country_window_h)
            is_foreign = txn_f["is_foreign"]
            is_remote = int(rng.random() < base_probe.remote_marginal_p)
            gap = float(np.log1p(1.0 / max(txn_f["hours_since_last"], 0.01)))
            logit = (
                cfg.w_amount           * txn_f["amount_zscore"]
                + cfg.w_night          * is_night
                + cfg.w_foreign        * is_foreign
                + cfg.w_remote         * is_remote
                + cfg.w_velocity_1h    * float(np.log1p(txn_f["velocity_1h"]))
                + cfg.w_velocity_24h   * float(np.log1p(txn_f["velocity_24h"]))
                + cfg.w_gap_short      * gap
                + cfg.w_remote_x_foreign * (is_remote * is_foreign)
                + cfg.w_remote_x_night   * (is_remote * is_night)
                + cfg.w_velocity_x_gap   * (float(np.log1p(txn_f["velocity_1h"])) * gap)
            )
            logits_no_intercept.append(logit)
            hist.record(ts, amount, ctry)

    scores = np.array(logits_no_intercept)
    lo, hi = -30.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        rate = float(np.mean(1.0 / (1.0 + np.exp(-np.clip(scores + mid, -15.0, 15.0)))))
        if abs(rate - target_rate) < 1e-6:
            break
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    intercept = float(0.5 * (lo + hi))
    print(f"  Calibrated intercept: {intercept:.4f}  (target fraud rate: {target_rate*100:.1f}%)")
    return replace(cfg, fraud_intercept=intercept)


def compute_fraud_score(txn_features: dict, is_night: int, is_foreign: int,
                        is_remote: int, cfg: GeneratorConfig) -> float:
    """Returns P(fraud) via logistic regression on transaction features."""
    gap = float(np.log1p(1.0 / max(txn_features["hours_since_last"], 0.01)))
    logit = (
        cfg.fraud_intercept
        + cfg.w_amount             * txn_features["amount_zscore"]
        + cfg.w_night              * is_night
        + cfg.w_foreign            * is_foreign
        + cfg.w_remote             * is_remote
        + cfg.w_velocity_1h        * float(np.log1p(txn_features["velocity_1h"]))
        + cfg.w_velocity_24h       * float(np.log1p(txn_features["velocity_24h"]))
        + cfg.w_gap_short          * gap
        + cfg.w_remote_x_foreign   * (is_remote * is_foreign)
        + cfg.w_remote_x_night     * (is_remote * is_night)
        + cfg.w_velocity_x_gap     * (float(np.log1p(txn_features["velocity_1h"])) * gap)
    )
    return _sigmoid(logit)


# ---------------------------------------------------------------------------
# Arrival times
# ---------------------------------------------------------------------------

def _sample_arrival_times(n_txn: int, peak_hour: float, kappa: float,
                           duration_h: float, period_start: float,
                           rng: np.random.Generator) -> np.ndarray:
    loc_rad = (peak_hour / 24.0) * 2.0 * np.pi

    rad_samples = stats.vonmises.rvs(
        kappa,
        loc=loc_rad,
        size=n_txn,
        random_state=rng
    )

    hour_samples = (rad_samples / (2.0 * np.pi)) * 24.0
    hour_samples = hour_samples % 24.0

    n_days = int(duration_h // 24)
    day_offsets = rng.integers(0, n_days, size=n_txn) * 24.0

    timestamps = period_start + day_offsets + hour_samples

    return np.sort(timestamps)


# ---------------------------------------------------------------------------
# Profiles, countries, warmup
# ---------------------------------------------------------------------------

def _build_profiles(cfg: GeneratorConfig, rng: np.random.Generator) -> list:
    p = np.asarray(cfg.country_probs, float); p /= p.sum()
    return [{
        "customer_id":  cid,
        "home_country": (home := cfg.countries[rng.choice(len(cfg.countries), p=p)]),
        "home_idx":     cfg.countries.index(home),
        "amount_mu":    float(rng.normal(cfg.amount_mu_global, cfg.amount_mu_personal_std)),
        "time_peak_hr": float(np.clip(rng.normal(cfg.time_peak_hour, cfg.time_personal_std_hr), 0, 23.99)),
    } for cid in range(cfg.n_customers)]


def _sample_countries(cfg: GeneratorConfig, home_idx: int, n: int,
                      rng: np.random.Generator) -> List[str]:
    p = np.full(len(cfg.countries), (1 - cfg.home_country_loyalty) / (len(cfg.countries) - 1))
    p[home_idx] = cfg.home_country_loyalty
    idxs = rng.choice(len(cfg.countries), size=n, p=p)
    return [cfg.countries[i] for i in idxs]


def _run_warmup(profiles: list, histories: Dict[int, CustomerHistory],
                cfg: GeneratorConfig, rng: np.random.Generator):
    for i in range(cfg.n_warmup_periods):
        start = -(cfg.n_warmup_periods - i) * cfg.period_duration_hours
        for prof in profiles:
            n    = max(1, int(rng.poisson(cfg.txns_per_customer_lambda)))
            ts   = _sample_arrival_times(n, prof["time_peak_hr"], cfg.time_kappa,
                                         cfg.period_duration_hours, start, rng)
            amts = rng.lognormal(prof["amount_mu"], cfg.amount_sigma, size=n)
            ctrs = _sample_countries(cfg, prof["home_idx"], n, rng)
            h    = histories[prof["customer_id"]]
            for i in range(n):
                h.record(float(ts[i]), float(amts[i]), ctrs[i])


# ---------------------------------------------------------------------------
# Period generator
# ---------------------------------------------------------------------------

def _generate_one_period(cfg: GeneratorConfig, histories: Dict[int, CustomerHistory],
                         profiles: list, period_start: float,
                         txn_id_start: int, period_idx: int,
                         rng: np.random.Generator) -> pd.DataFrame:
    rows:  List[dict] = []
    txn_id = txn_id_start

    for prof in profiles:
        cid, home = prof["customer_id"], prof["home_country"]
        hist      = histories[cid]
        n         = max(1, int(rng.poisson(cfg.txns_per_customer_lambda)))
        ts_arr    = _sample_arrival_times(n, prof["time_peak_hr"], cfg.time_kappa,
                                          cfg.period_duration_hours, period_start, rng)
        amounts   = rng.lognormal(prof["amount_mu"], cfg.amount_sigma, size=n)
        ctrs      = _sample_countries(cfg, prof["home_idx"], n, rng)

        for i in range(n):
            ts, amount, ctry = float(ts_arr[i]), float(amounts[i]), ctrs[i]
            time_hour  = ts % 24.0
            is_night   = int(time_hour >= 22.0 or time_hour < 6.0)
            txn_features = hist.features(ts, amount, ctry, cfg.new_country_window_h)
            is_foreign = txn_features["is_foreign"]
            is_remote  = int(rng.random() < cfg.remote_marginal_p)
            score = compute_fraud_score(txn_features, is_night, is_foreign, is_remote, cfg)

            rows.append({
                "period": period_idx, "customer_id": cid, "home_country": home,
                "txn_id": txn_id, "timestamp_h": round(ts, 4), "time_hour": round(time_hour, 3),
                "amount": round(amount, 2), "country": ctry, "is_remote": is_remote,
                "velocity_1h": round(txn_features["velocity_1h"], 1),
                "velocity_24h": round(txn_features["velocity_24h"], 1),
                "amount_zscore": txn_features["amount_zscore"],
                "is_foreign": txn_features["is_foreign"],
                "hours_since_last": txn_features["hours_since_last"],
                "fraud_score": float(score),
            })
            txn_id += 1
            hist.record(ts, amount, ctry)

    df = pd.DataFrame(rows)

    df["is_fraud"] = rng.random(len(df)) < df["fraud_score"].to_numpy()
    df["is_fraud"] = df["is_fraud"].astype(int)

    return (
        df.sort_values(["customer_id", "timestamp_h"])
          .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame):
    n, nf = len(df), int(df["is_fraud"].sum())
    print(f"\n{'='*55}")
    print(f"  n={n:,}  fraud={nf/n*100:.2f}%  amt=EUR{df['amount'].mean():.2f}"
          f"  remote={df['is_remote'].mean()*100:.1f}%  avg_P={df['fraud_score'].mean():.4f}")
    print(f"{'='*55}\n")


def generate(cfg: Optional[GeneratorConfig] = None, seed: int = 42,
             verbose: bool = True) -> pd.DataFrame:
    """Generate a single stationary period with warmup."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(seed)
    profiles  = _build_profiles(cfg, rng)
    histories = {p["customer_id"]: CustomerHistory(p["home_country"]) for p in profiles}
    _run_warmup(profiles, histories, cfg, rng)
    df = _generate_one_period(cfg, histories, profiles, 0.0, 0, 1, rng).drop(columns=["period"])
    if verbose:
        _print_summary(df)
    return df


def generate_multiperiod(n_periods: int = 10,
                         cfg: Optional[GeneratorConfig] = None,
                         cfg_list: Optional[List[GeneratorConfig]] = None,
                         seed: int = 42, verbose: bool = True,
                         label: str = "") -> pd.DataFrame:
    """Generate multiple periods. Pass cfg_list for drift experiments."""
    base     = cfg or GeneratorConfig()
    cfg_list = cfg_list or [base] * n_periods
    rng      = np.random.default_rng(seed)
    profiles  = _build_profiles(base, rng)
    histories = {p["customer_id"]: CustomerHistory(p["home_country"]) for p in profiles}
    _run_warmup(profiles, histories, base, rng)

    frames, txn_id = [], 0
    _W = ["w_amt","w_nt","w_fgn","w_v1h","w_v24h","w_gap","w_rmt","w_r_fg","w_r_nt","w_v_gp"]
    _W_ATTRS = ["w_amount","w_night","w_foreign","w_velocity_1h","w_velocity_24h",
                "w_gap_short","w_remote","w_remote_x_foreign","w_remote_x_night","w_velocity_x_gap"]

    if verbose:
        hdr = f"  {label}  ({len(cfg_list)} periods)" if label else f"  ({len(cfg_list)} periods)"
        print(f"\n{'='*100}\n{hdr}\n{'='*100}")
        header = f"  {'Per':>3}  {'Txns':>6}  {'Fraud%':>7}"
        for name in _W:
            header += f"  {name:>7}"
        print(header)
        print("  " + "-"*80)

    for p, pcfg in enumerate(cfg_list, start=1):
        period_start = (p - 1) * base.period_duration_hours
        # Each period gets its own RNG derived from the parent, keeping periods
        # independently reproducible while the overall sequence stays deterministic.
        df_p = _generate_one_period(pcfg, histories, profiles,
                                    period_start, txn_id, p,
                                    np.random.default_rng(rng.integers(0, 2**31)))
        txn_id += len(df_p)
        frames.append(df_p)

        if verbose:
            row = f"  {p:>3}  {len(df_p):>6,}  {df_p['is_fraud'].mean()*100:>6.2f}%"
            for attr in _W_ATTRS:
                row += f"  {getattr(pcfg, attr):>7.3f}"
            print(row)

    df = pd.concat(frames, ignore_index=True)
    if verbose:
        print("  " + "-"*80)
        print(f"  Overall fraud rate: {df['is_fraud'].mean()*100:.2f}%  |  Total rows: {len(df):,}\n")
    return df


# ---------------------------------------------------------------------------
# Diagnostics

def check_calibration(cfg: Optional[GeneratorConfig] = None,
                      seeds: Optional[List[int]] = None,
                      verbose: bool = True) -> pd.DataFrame:
    """Fraud rate and segment breakdown across seeds."""
    cfg, seeds = cfg or GeneratorConfig(), seeds or [0, 1, 2, 3, 4]
    rows = []
    for s in seeds:
        df  = generate(cfg, seed=s, verbose=False)
        nm  = (df["time_hour"] >= 22) | (df["time_hour"] < 6)
        fg  = df["is_foreign"] == 1
        rm  = df["is_remote"] == 1
        rows.append({
            "seed":             s,
            "n_txn":            len(df),
            "fraud_rate":       round(df["is_fraud"].mean(), 4),
            "fraud_night":      round(df.loc[nm,  "is_fraud"].mean(), 4),
            "fraud_day":        round(df.loc[~nm, "is_fraud"].mean(), 4),
            "fraud_foreign":    round(df.loc[fg,  "is_fraud"].mean(), 4) if fg.any()  else None,
            "fraud_domestic":   round(df.loc[~fg, "is_fraud"].mean(), 4),
            "fraud_remote":     round(df.loc[rm,  "is_fraud"].mean(), 4) if rm.any()  else None,
            "fraud_inperson":   round(df.loc[~rm, "is_fraud"].mean(), 4),
            "fraud_top_amt":    round(df.loc[df["amount"] >= df["amount"].quantile(.9), "is_fraud"].mean(), 4),
        })
    result = pd.DataFrame(rows)
    if verbose:
        print(f"\n{'='*72}\n  Calibration check — {len(seeds)} seeds\n{'='*72}")
        print(result.to_string(index=False))
        print(f"\n  fraud_rate  mean={result['fraud_rate'].mean():.4f}  "
              f"std={result['fraud_rate'].std():.4f}\n")
    return result


# ---------------------------------------------------------------------------
# Drift experiment helpers
# ---------------------------------------------------------------------------

def _build_probe_matrix(cfg: GeneratorConfig, seed: int, n_probe_customers: int = 2000) -> pd.DataFrame:
    """Run one warmup + one period for a small probe population and return the
    pre-computed feature matrix. Used by the binary-search calibration routines
    so weights can be evaluated cheaply without re-running warmup each iteration."""
    rng = np.random.default_rng(seed)
    probe_cfg = replace(cfg, n_customers=n_probe_customers)

    profiles = _build_profiles(probe_cfg, rng)
    hists = {p["customer_id"]: CustomerHistory(p["home_country"]) for p in profiles}
    _run_warmup(profiles, hists, probe_cfg, rng)

    rows = []

    for prof in profiles:
        hist = hists[prof["customer_id"]]
        n = max(1, int(rng.poisson(probe_cfg.txns_per_customer_lambda)))

        ts_arr = _sample_arrival_times(
            n, prof["time_peak_hr"], probe_cfg.time_kappa,
            probe_cfg.period_duration_hours, 0.0, rng
        )
        amounts = np.exp(rng.normal(prof["amount_mu"], probe_cfg.amount_sigma, n))
        ctrs = _sample_countries(probe_cfg, prof["home_idx"], n, rng)

        for i in range(n):
            ts = float(ts_arr[i])
            amount = float(amounts[i])
            ctry = ctrs[i]

            is_night = int(ts % 24.0 >= 22.0 or ts % 24.0 < 6.0)
            txn_f = hist.features(ts, amount, ctry, probe_cfg.new_country_window_h)
            is_remote = int(rng.random() < probe_cfg.remote_marginal_p)

            gap = float(np.log1p(1.0 / max(txn_f["hours_since_last"], 0.01)))
            v1 = float(np.log1p(txn_f["velocity_1h"]))
            v24 = float(np.log1p(txn_f["velocity_24h"]))
            is_foreign = txn_f["is_foreign"]

            rows.append({
                "amount_zscore": txn_f["amount_zscore"],
                "is_night": is_night,
                "is_foreign": is_foreign,
                "is_remote": is_remote,
                "v1": v1,
                "v24": v24,
                "gap": gap,
                "remote_x_foreign": is_remote * is_foreign,
                "remote_x_night": is_remote * is_night,
                "velocity_x_gap": v1 * gap,
            })

            hist.record(ts, amount, ctry)

    return pd.DataFrame(rows)


def _estimate_fraud_rate_from_probe(cfg: GeneratorConfig, probe: pd.DataFrame) -> float:
    logit = (
        cfg.fraud_intercept
        + cfg.w_amount * probe["amount_zscore"].to_numpy()
        + cfg.w_night * probe["is_night"].to_numpy()
        + cfg.w_foreign * probe["is_foreign"].to_numpy()
        + cfg.w_remote * probe["is_remote"].to_numpy()
        + cfg.w_velocity_1h * probe["v1"].to_numpy()
        + cfg.w_velocity_24h * probe["v24"].to_numpy()
        + cfg.w_gap_short * probe["gap"].to_numpy()
        + cfg.w_remote_x_foreign * probe["remote_x_foreign"].to_numpy()
        + cfg.w_remote_x_night * probe["remote_x_night"].to_numpy()
        + cfg.w_velocity_x_gap * probe["velocity_x_gap"].to_numpy()
    )

    return float(np.mean(1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))))


def _calibrate_B_weights(pattern_A: GeneratorConfig, seed: int,
                          target_rate: float = 0.01) -> GeneratorConfig:
    """Scale pattern B's active weights until mean P(fraud) ≈ target_rate.

    Intercept is fixed at pattern_A.fraud_intercept. Only the B-side feature
    weights (foreign, velocity, gap) are scaled by a multiplier m found via
    binary search. Structural zeros (night, remote, interactions) are enforced.
    """
    probe = _build_probe_matrix(pattern_A, seed)

    raw = dict(w_foreign=3.0, w_velocity_1h=3.0, w_velocity_24h=2.0,
               w_gap_short=1.5, w_velocity_x_gap=1.5)

    lo, hi = 0.01, 5.0
    best_cfg, best_err = None, float("inf")

    for _ in range(40):
        m = 0.5 * (lo + hi)
        candidate = replace(pattern_A,
            w_night=0.0, w_remote=0.0,
            w_remote_x_foreign=0.0, w_remote_x_night=0.0,
            w_foreign        = round(raw["w_foreign"]        * m, 4),
            w_velocity_1h    = round(raw["w_velocity_1h"]    * m, 4),
            w_velocity_24h   = round(raw["w_velocity_24h"]   * m, 4),
            w_gap_short      = round(raw["w_gap_short"]      * m, 4),
            w_velocity_x_gap = round(raw["w_velocity_x_gap"] * m, 4),
        )
        rate = _estimate_fraud_rate_from_probe(candidate, probe)
        err  = abs(rate - target_rate)
        if err < best_err:
            best_cfg, best_err = candidate, err
        if err < 1e-5:
            break
        if rate > target_rate:
            hi = m
        else:
            lo = m

    m_final = 0.5 * (lo + hi)
    print(f"  Pattern B weight multiplier: {m_final:.4f}  (estimated rate: {rate:.4f})")
    return best_cfg

def _calibrate_intermediate_weights(
    cfg_candidate: GeneratorConfig,
    seed: int,
    target_rate: float = 0.01,
) -> GeneratorConfig:
    """Scale all feature weights of an interpolated config by a single multiplier m
    (found via binary search) so that mean P(fraud) ≈ target_rate with the
    intercept held fixed. Zero weights stay zero because 0 * m = 0."""
    probe = _build_probe_matrix(cfg_candidate, seed)

    fields = [
        "w_amount",
        "w_night",
        "w_foreign",
        "w_velocity_1h",
        "w_velocity_24h",
        "w_gap_short",
        "w_remote",
        "w_remote_x_foreign",
        "w_remote_x_night",
        "w_velocity_x_gap",
    ]

    original = {f: getattr(cfg_candidate, f) for f in fields}

    lo, hi = 0.05, 10.0
    best_cfg = cfg_candidate
    best_err = float("inf")
    best_rate = None
    best_m = None

    for _ in range(50):
        m = 0.5 * (lo + hi)

        scaled_cfg = replace(
            cfg_candidate,
            **{f: round(original[f] * m, 4) for f in fields}
        )

        rate = _estimate_fraud_rate_from_probe(scaled_cfg, probe)
        err = abs(rate - target_rate)

        if err < best_err:
            best_cfg = scaled_cfg
            best_err = err
            best_rate = rate
            best_m = m

        if rate > target_rate:
            hi = m
        else:
            lo = m

    print(
        f"  Intermediate config calibrated: "
        f"m={best_m:.4f}, estimated rate={best_rate:.4f}"
    )

    return best_cfg

def _build_axis12_endpoints(seed: int) -> tuple[GeneratorConfig, GeneratorConfig]:
    """Build the two endpoint configs used by Axis 1 and Axis 2.

    Pattern A: fraud driven by remote channel and night-time hour.
    Pattern B: fraud driven by foreign country and high velocity.
    Both are calibrated to ~1% fraud rate using the same intercept.
    """
    pattern_A = calibrate_intercept(replace(GeneratorConfig(),
        w_amount=1.50,
        w_night=3.00,
        w_foreign=0.00,
        w_velocity_1h=0.00,
        w_velocity_24h=0.00,
        w_gap_short=0.50,
        w_remote=3.00,
        w_remote_x_foreign=0.00,
        w_remote_x_night=1.00,
        w_velocity_x_gap=0.00,
    ), target_rate=0.01, seed=seed)

    pattern_B = _calibrate_B_weights(
        pattern_A=pattern_A,
        seed=seed,
        target_rate=0.01,
    )

    return pattern_A, pattern_B

def _build_cfg_list_axis12(
    pattern_A: GeneratorConfig,
    pattern_B: GeneratorConfig,
    d: float,
    speed: str,
    seed: int,
    max_d: float = 1.50,
) -> List[GeneratorConfig]:
    """
    Interpolate from Pattern A toward Pattern B for Axis 1 and Axis 2.

    Pattern A and Pattern B are both around 1% fraud.
    Intermediate configs are also locally calibrated to stay around 1% fraud.

    Important:
    - The intercept is never changed.
    - The interpolation fraction still controls how far we move from A to B.
    - Local calibration only rescales overall weight strength.
    """

    t = d / max_d
    n = 10

    def _interp(a: float, b: float, frac: float) -> float:
        return round(a + frac * (b - a), 4)

    def _cfg(frac: float) -> GeneratorConfig:
        if frac <= 0.0:
            return pattern_A

        if frac >= 1.0:
            return pattern_B

        candidate = replace(
            pattern_A,

            # Keep intercept fixed
            fraud_intercept=pattern_A.fraud_intercept,

            # Interpolate only weights
            w_night=_interp(pattern_A.w_night, pattern_B.w_night, frac),
            w_foreign=_interp(pattern_A.w_foreign, pattern_B.w_foreign, frac),
            w_velocity_1h=_interp(pattern_A.w_velocity_1h, pattern_B.w_velocity_1h, frac),
            w_velocity_24h=_interp(pattern_A.w_velocity_24h, pattern_B.w_velocity_24h, frac),
            w_gap_short=_interp(pattern_A.w_gap_short, pattern_B.w_gap_short, frac),
            w_remote=_interp(pattern_A.w_remote, pattern_B.w_remote, frac),
            w_remote_x_foreign=_interp(
                pattern_A.w_remote_x_foreign,
                pattern_B.w_remote_x_foreign,
                frac,
            ),
            w_remote_x_night=_interp(
                pattern_A.w_remote_x_night,
                pattern_B.w_remote_x_night,
                frac,
            ),
            w_velocity_x_gap=_interp(
                pattern_A.w_velocity_x_gap,
                pattern_B.w_velocity_x_gap,
                frac,
            ),
        )

        return _calibrate_intermediate_weights(
            cfg_candidate=candidate,
            seed=seed,
            target_rate=0.01,
        )

    if speed == "sudden":
        return [pattern_A if i < 5 else _cfg(t) for i in range(n)]

    if speed == "gradual":
        return [_cfg(t * i / (n - 1)) for i in range(n)]

    raise ValueError(f"speed must be 'sudden' or 'gradual', got {speed!r}")


# ---------------------------------------------------------------------------
# Dataset saving
# ---------------------------------------------------------------------------

def _finish_dataset(label: str, df: pd.DataFrame, meta: dict) -> None:
    """Save CSV + metadata JSON."""
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{label}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    json_path = os.path.join(OUT_DIR, f"{label}_metadata.json")
    with open(json_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Metadata: {json_path}")

# ---------------------------------------------------------------------------
# Baseline: Stationary Pattern A
#

def run_baseline(seed: int = SEED) -> None:
    """Baseline — stationary Pattern A across all 10 periods.

    No drift is introduced. This gives the no-drift reference condition.
    """
    pattern_A, _ = _build_axis12_endpoints(seed)

    cfg_list = [pattern_A for _ in range(10)]
    label = "baseline_pattern_A_stationary"

    print(f"\n{'='*60}\nBaseline — stationary Pattern A\n{'='*60}")

    df = generate_multiperiod(
        cfg_list=cfg_list,
        cfg=pattern_A,
        seed=seed,
        verbose=True,
        label=label,
    )

    _finish_dataset(label, df, {
        "axis": "baseline",
        "label": label,
        "seed": seed,
        "n_periods": len(cfg_list),
        "period_config": {
            str(p): "A"
            for p in range(1, len(cfg_list) + 1)
        },
    })
# ---------------------------------------------------------------------------
# Axis 1 — Drift Magnitude
# ---------------------------------------------------------------------------

DELTAS_AXIS1 = [0.30, 0.60, 0.90, 1.20, 1.50]


def run_axis1(deltas: List[float] = DELTAS_AXIS1, seed: int = SEED) -> None:
    """Axis 1 — Drift Magnitude: sudden drift across 5 delta values.

    Pattern A (baseline): remote + night drive fraud; foreign + velocity = 0.
    Pattern B (drifted):  foreign + velocity drive fraud; remote + night = 0.
    delta scales the interpolation fraction A→B (max delta = full swap).
    """
    pattern_A, pattern_B = _build_axis12_endpoints(seed)
    base = pattern_A
    for delta in deltas:
        cfg_list = _build_cfg_list_axis12(
            pattern_A=pattern_A,
            pattern_B=pattern_B,
            d=delta,
            speed="sudden",
            seed=seed,
        )
        label = f"axis1_delta{delta}_sudden"
        print(f"\n{'='*60}\nAxis 1 — delta={delta}  sudden\n{'='*60}")
        df = generate_multiperiod(cfg_list=cfg_list, cfg=base, seed=seed,
                                  verbose=True, label=label)
        _finish_dataset(label, df, {
            "axis": "axis1", "label": label, "seed": seed,
            "n_periods": len(cfg_list), "drift_magnitude_d": delta, "speed": "sudden",
            "period_config": {str(p): "A" if p <= 5 else "B"
                              for p in range(1, len(cfg_list) + 1)},
        })


# ---------------------------------------------------------------------------
# Axis 2 — Drift Speed
# ---------------------------------------------------------------------------

DELTAS_AXIS2 = [0.30, 0.60, 0.90, 1.20, 1.50]


def run_axis2(deltas: List[float] = DELTAS_AXIS2, seed: int = SEED) -> None:
    """Axis 2 — Gradual Drift: gradual A→B drift across the same delta values.

    Pattern A: remote + night drive fraud.
    Pattern B: foreign + velocity drive fraud.
    The drift is introduced gradually over periods 1–10.
    """
    pattern_A, pattern_B = _build_axis12_endpoints(seed)
    base = pattern_A

    for delta in deltas:
        speed = "gradual"

        cfg_list = _build_cfg_list_axis12(
            pattern_A=pattern_A,
            pattern_B=pattern_B,
            d=delta,
            speed=speed,
            seed=seed,
        )

        label = f"axis2_delta{delta}_gradual"

        print(f"\n{'='*60}\nAxis 2 — delta={delta}  {speed}\n{'='*60}")

        df = generate_multiperiod(
            cfg_list=cfg_list,
            cfg=base,
            seed=seed,
            verbose=True,
            label=label,
        )

        _finish_dataset(label, df, {
            "axis": "axis2",
            "label": label,
            "seed": seed,
            "n_periods": len(cfg_list),
            "drift_magnitude_d": delta,
            "speed": speed,
            "period_config": {
                str(p): "gradual_A_to_B"
                for p in range(1, len(cfg_list) + 1)
            },
        })


# ---------------------------------------------------------------------------
# Axis 3 — Freeze Duration
# ---------------------------------------------------------------------------

FREEZE_DURATIONS = [1, 2, 3, 4, 5, 6]


def run_axis3(freeze_durations: List[int] = FREEZE_DURATIONS, seed: int = SEED) -> None:
    n_total = 10  # 3 baseline + up to 6 frozen + at least 1 recovery = 10
    n_baseline = 3
    for k in freeze_durations:
        base = calibrate_intercept(GeneratorConfig(), target_rate=0.01, seed=seed)
        frozen = replace(base, fraud_intercept=-50.0)

        cfg_list: List[GeneratorConfig] = []
        period_config: Dict[str, str] = {}
        for p in range(1, n_total + 1):
            if p <= n_baseline:
                cfg_list.append(base);   period_config[str(p)] = "baseline"
            elif p <= n_baseline + k:
                cfg_list.append(frozen); period_config[str(p)] = "frozen"
            else:
                cfg_list.append(base);   period_config[str(p)] = "recovery"

        label = f"axis3_freeze_k{k}"
        print(f"\n{'='*60}\nAxis 3 — freeze_duration k={k}  ({n_total} periods)\n{'='*60}")
        df = generate_multiperiod(cfg_list=cfg_list, cfg=base, seed=seed,
                                  verbose=True, label=label)
        _finish_dataset(label, df, {
            "axis": "axis3", "label": label, "seed": seed,
            "n_periods": n_total, "freeze_duration_k": k,
            "freeze_intercept": -50.0,
            "baseline_intercept": float(base.fraud_intercept),
            "period_config": period_config,
        })


# ---------------------------------------------------------------------------
# Axis 4 — Freeze Depth
# ---------------------------------------------------------------------------

FREEZE_OFFSETS = [0.2, 1.0, 2.0, 4.0, 8.0, 40.0]


def run_axis4(offsets: List[float] = FREEZE_OFFSETS, seed: int = SEED) -> None:
    """Axis 4 — Freeze Depth: varying fraud suppression intensity.

    Freeze intercepts are defined relative to the calibrated baseline intercept:
    alpha_freeze = alpha_baseline - offset.
    Larger offsets imply stronger fraud suppression.
    """
    base = calibrate_intercept(GeneratorConfig(), target_rate=0.01, seed=seed)

    for offset in offsets:
        freeze_alpha = base.fraud_intercept - offset
        frozen = replace(base, fraud_intercept=freeze_alpha)

        cfg_list: List[GeneratorConfig] = []
        period_config: Dict[str, str] = {}

        for p in range(1, 11):
            if p <= 4:
                cfg_list.append(base)
                period_config[str(p)] = "baseline"
            elif p <= 7:
                cfg_list.append(frozen)
                period_config[str(p)] = "frozen"
            else:
                cfg_list.append(base)
                period_config[str(p)] = "recovery"

        label = f"axis4_offset{offset}"

        print(f"\n{'='*60}\nAxis 4 — freeze_offset={offset}, freeze_alpha={freeze_alpha:.4f}\n{'='*60}")

        df = generate_multiperiod(
            cfg_list=cfg_list,
            cfg=base,
            seed=seed,
            verbose=True,
            label=label,
        )

        _finish_dataset(label, df, {
            "axis": "axis4",
            "label": label,
            "seed": seed,
            "n_periods": 10,
            "freeze_offset": offset,
            "freeze_intercept_alpha": float(freeze_alpha),
            "baseline_intercept": float(base.fraud_intercept),
            "period_config": period_config,
        })


# ---------------------------------------------------------------------------
# Axis 5 — Pattern Rotation
# ---------------------------------------------------------------------------

def _build_axis5_patterns(seed: int) -> Dict[str, GeneratorConfig]:
    """Axis 5 patterns with non-focus weights kept near zero, not exactly zero."""

    # Pattern A — remote/night-focused
    pattern_A = calibrate_intercept(replace(GeneratorConfig(),
        w_amount=0.50,
        w_night=3.00,
        w_foreign=0.15,
        w_velocity_1h=0.15,
        w_velocity_24h=0.15,
        w_gap_short=0.15,
        w_remote=3.00,
        w_remote_x_foreign=0.10,
        w_remote_x_night=1.00,
        w_velocity_x_gap=0.10,
    ), target_rate=0.01, seed=seed)

    # Pattern B — velocity/foreign-focused
    pattern_B = calibrate_intercept(replace(GeneratorConfig(),
        w_amount=0.50,
        w_night=0.15,
        w_foreign=3.00,
        w_velocity_1h=3.00,
        w_velocity_24h=2.00,
        w_gap_short=1.50,
        w_remote=0.15,
        w_remote_x_foreign=0.10,
        w_remote_x_night=0.10,
        w_velocity_x_gap=1.50,
    ), target_rate=0.01, seed=seed)

    # Pattern C — high-amount/night-focused
    pattern_C = calibrate_intercept(replace(GeneratorConfig(),
        w_amount=4.00,
        w_night=3.50,
        w_foreign=0.50,
        w_velocity_1h=0.15,
        w_velocity_24h=0.15,
        w_gap_short=0.15,
        w_remote=1.00,
        w_remote_x_foreign=0.10,
        w_remote_x_night=1.50,
        w_velocity_x_gap=0.10,
    ), target_rate=0.01, seed=seed)

    return {"A": pattern_A, "B": pattern_B, "C": pattern_C}

_ROTATION_SCHEDULE = {
    1: "A", 2: "A", 3: "B", 4: "B", 5: "C",
    6: "C", 7: "A", 8: "A", 9: "B", 10: "B",
}

def run_axis5(seed: int = SEED) -> None:
    patterns = _build_axis5_patterns(seed)
    base = patterns["A"]

    cfg_list = [patterns[_ROTATION_SCHEDULE[p]] for p in range(1, 11)]

    label = "axis5_pattern_rotation"

    print(f"\n{'='*60}\nAxis 5 — Pattern Rotation\n{'='*60}")

    df = generate_multiperiod(
        cfg_list=cfg_list,
        cfg=base,
        seed=seed,
        verbose=True,
        label=label
    )

    df["active_pattern"] = df["period"].map(_ROTATION_SCHEDULE)

    _finish_dataset(label, df, {
        "axis": "axis5",
        "label": label,
        "seed": seed,
        "rotation_schedule": _ROTATION_SCHEDULE,
    })

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(SEED)
    run_baseline()
    run_axis1()
    run_axis2()
    run_axis3()
    run_axis4()
    run_axis5()


if __name__ == "__main__":
    main()
