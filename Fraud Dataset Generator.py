from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# SECTION 0 - Configuration
# =============================================================================

@dataclass
class GeneratorConfig:

    n_customers: int = 10000
    txns_per_customer_lambda: float = 10.0
    amount_mu_global:       float = 2.95
    amount_sigma:           float = 1.2    
    amount_mu_personal_std: float = 0.4
    time_peak_hour:       float = 13.0
    time_kappa:           float = 1.5   
    time_personal_std_hr: float = 3.0   
    countries:            List[str]   = field(default_factory=lambda: ["A", "B", "C"])
    country_probs:        List[float] = field(default_factory=lambda: [0.85, 0.08, 0.07])
    home_country_loyalty: float = 0.88  
    remote_marginal_p:     float = 0.36
    remote_fraud_lift:     float = 0.4
    fraud_intercept: float = -13.8  # recalibrated: interactions raise avg logit ~1.3
    w_amount:       float = 0.7
    w_night:        float = 0.6
    w_foreign:      float = 1.3
    w_new_country:  float = 1.0
    w_velocity_1h:  float = 0.9
    w_velocity_24h: float = 0.3
    w_delta_amount: float = 0.6
    w_gap_short:    float = 0.8
    w_remote:       float = 1.4

    # -- Interaction terms ------------------------------------------------
    
    w_remote_x_foreign:    float = 0.8
    w_remote_x_night:      float = 0.5
    w_velocity_x_gap:      float = 0.4
    w_newcountry_x_foreign: float = 0.6

    fraud_noise_std: float = 0.5

    period_duration_hours: float = 168.0
    n_warmup_periods:      int   = 20

# =============================================================================
# SECTION 1 - Customer history
# =============================================================================

class CustomerHistory:

    def __init__(self, home_country: str):
        self.home_country = home_country
        self.timestamps:  List[float] = []
        self.amounts:     List[float] = []
        self.countries:   List[str]   = []

    def features(self, ts: float, amount: float, country: str) -> dict:

        if not self.timestamps:
            return dict(
                velocity_1h=0.0, velocity_24h=0.0,
                delta_amount=0.0, amount_zscore=0.0,
                is_new_country=1, hours_since_last=48.0,
            )

        gap  = ts - self.timestamps[-1]
        arr  = np.asarray(self.timestamps)
        v1h  = len(arr) - int(arr.searchsorted(ts - 1.0,  side='left'))
        v24h = len(arr) - int(arr.searchsorted(ts - 24.0, side='left'))

        log_amts = np.log(np.clip(self.amounts, 1e-6, None))
        mu_log   = float(log_amts.mean())
        std_log  = float(log_amts.std()) + 1e-6
        z        = (float(np.log(max(amount, 1e-6))) - mu_log) / std_log

        mean_amt = float(np.mean(self.amounts))
        delta    = (amount - mean_amt) / (mean_amt + 1.0)

        return dict(
            velocity_1h      = float(v1h),
            velocity_24h     = float(v24h),
            delta_amount     = round(delta, 4),
            amount_zscore    = round(z, 4),
            is_new_country   = int(country not in self.countries),
            hours_since_last = round(max(gap, 0.0), 3),
        )

    def record(self, ts: float, amount: float, country: str):
        """Commit a completed transaction to history."""
        self.timestamps.append(ts)
        self.amounts.append(amount)
        self.countries.append(country)

# =============================================================================
# SECTION 2 - Fraud scoring
# =============================================================================

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0))))


def compute_fraud_score(
    beh:        dict,
    is_night:   int,
    is_foreign: int,
    is_remote:  int,
    cfg:        GeneratorConfig,
    noise_std:  float,
    rng:        np.random.Generator,
) -> float:
    gap_signal = float(np.log1p(1.0 / max(beh["hours_since_last"], 0.01)))

    # Main effects
    logit = (
        cfg.fraud_intercept
        + cfg.w_amount       * beh["amount_zscore"]
        + cfg.w_night        * is_night
        + cfg.w_foreign      * is_foreign
        + cfg.w_remote       * is_remote
        + cfg.w_new_country  * beh["is_new_country"]
        + cfg.w_velocity_1h  * float(np.log1p(beh["velocity_1h"]))
        + cfg.w_velocity_24h * float(np.log1p(beh["velocity_24h"]))
        + cfg.w_delta_amount * beh["delta_amount"]
        + cfg.w_gap_short    * gap_signal
    )

    # Interaction terms — joint effects stronger than main effects alone
    # remote × foreign: CNP cross-border, dominant fraud channel (ECB, 2023)
    logit += cfg.w_remote_x_foreign * (is_remote * is_foreign)
    # remote × night: automated overnight CNP scripts (Whitrow et al., 2009)
    logit += cfg.w_remote_x_night * (is_remote * is_night)
    # velocity × gap: burst activity with rapid-fire timing = card-testing
    logit += cfg.w_velocity_x_gap * (float(np.log1p(beh["velocity_1h"])) * gap_signal)
    # new country × foreign: first-ever foreign country far more suspicious
    logit += cfg.w_newcountry_x_foreign * (beh["is_new_country"] * is_foreign)

    logit += float(rng.normal(0.0, noise_std))
    return _sigmoid(logit)


# =============================================================================
# SECTION 3 - Arrival process
# =============================================================================

def _sample_arrival_times(
    n_txn:        int,
    peak_hour:    float,
    kappa:        float,
    duration_h:   float,
    period_start: float,
    rng:          np.random.Generator,
) -> np.ndarray:

    peak_rad    = 2.0 * np.pi * (peak_hour / 24.0)
    n_candidates = max(n_txn * 6, 50)

    accepted: List[float] = []
    while len(accepted) < n_txn:
        candidates = rng.uniform(
            period_start, period_start + duration_h, size=n_candidates
        )
        hours_rad = 2.0 * np.pi * ((candidates % 24.0) / 24.0)
        intensity = np.exp(kappa * np.cos(hours_rad - peak_rad))
        intensity /= intensity.max()
        u = rng.uniform(0.0, 1.0, size=n_candidates)
        accepted.extend(candidates[u < intensity].tolist())

    accepted = sorted(accepted)[:n_txn]
    return np.array(accepted)


# =============================================================================
# SECTION 4 - Main generator
# =============================================================================

def _generate_one_period(
    cfg:          GeneratorConfig,
    histories:    Dict[int, CustomerHistory],
    profiles:     list,
    period_start: float,
    txn_id_start: int,
    period_idx:   int,
    rng:          np.random.Generator,
) -> pd.DataFrame:

    rows: List[dict] = []
    txn_id = txn_id_start

    for prof in profiles:
        cid      = prof["customer_id"]
        history  = histories[cid]
        home     = prof["home_country"]
        home_idx = cfg.countries.index(home)

        n_txn = max(1, int(rng.poisson(cfg.txns_per_customer_lambda)))

        # Arrival times: personal time preference encoded in arrival process
        timestamps = _sample_arrival_times(
            n_txn        = n_txn,
            peak_hour    = prof["time_peak_hr"],
            kappa        = cfg.time_kappa,
            duration_h   = cfg.period_duration_hours,
            period_start = period_start,
            rng          = rng,
        )

        # Amounts: personal log-normal
        amounts = rng.lognormal(prof["amount_mu"], cfg.amount_sigma, size=n_txn)

        # Countries: biased toward home country
        home_p = np.full(
            len(cfg.countries),
            (1.0 - cfg.home_country_loyalty) / (len(cfg.countries) - 1)
        )
        home_p[home_idx] = cfg.home_country_loyalty
        countries = [
            cfg.countries[rng.choice(len(cfg.countries), p=home_p)]
            for _ in range(n_txn)
        ]

        noise_std = cfg.fraud_noise_std + prof["risk_score"] * 0.5

        for i in range(n_txn):
            ts        = float(timestamps[i])
            amount    = float(amounts[i])
            country   = countries[i]

            # time_hour derived directly from timestamp (not drawn independently)
            time_hour  = ts % 24.0
            is_night   = int(time_hour >= 22.0 or time_hour < 6.0)
            is_foreign = int(country != home)

            beh = history.features(ts, amount, country)

            # Remote flag: drawn conditionally on fraud propensity.
            # Compute partial logit (excluding remote term) centred at the
            # intercept so that sigmoid(0) = 0.5 for a typical transaction.
            gap_signal = float(np.log1p(1.0 / max(beh["hours_since_last"], 0.01)))
            partial_logit = (
                cfg.fraud_intercept
                + cfg.w_amount       * beh["amount_zscore"]
                + cfg.w_night        * is_night
                + cfg.w_foreign      * is_foreign
                + cfg.w_new_country  * beh["is_new_country"]
                + cfg.w_velocity_1h  * float(np.log1p(beh["velocity_1h"]))
                + cfg.w_velocity_24h * float(np.log1p(beh["velocity_24h"]))
                + cfg.w_delta_amount * beh["delta_amount"]
                + cfg.w_gap_short    * gap_signal
            )
            # fraud_propensity in (0,1); centred so 0.5 = typical transaction.
            # p_remote rises above remote_marginal_p for high-propensity txns
            # and falls below it for low-propensity txns.
            fraud_propensity = _sigmoid(partial_logit - cfg.fraud_intercept)
            p_remote = float(np.clip(
                cfg.remote_marginal_p + cfg.remote_fraud_lift * (fraud_propensity - 0.5),
                0.01, 0.99,
            ))
            is_remote = int(rng.random() < p_remote)
            score = compute_fraud_score(
                beh, is_night, is_foreign, is_remote, cfg, noise_std, rng
            )
            is_fraud = int(rng.random() < score)

            rows.append({
                "period"          : period_idx,
                "customer_id"     : cid,
                "home_country"    : home,
                "txn_id"          : txn_id,
                "timestamp_h"     : round(ts, 4),
                "time_hour"       : round(time_hour, 3),
                "amount"          : round(amount, 2),
                "country"         : country,
                "is_remote"       : is_remote,
                "velocity_1h"     : round(beh["velocity_1h"], 1),
                "velocity_24h"    : round(beh["velocity_24h"], 1),
                "delta_amount"    : beh["delta_amount"],
                "amount_zscore"   : beh["amount_zscore"],
                "is_new_country"  : beh["is_new_country"],
                "hours_since_last": beh["hours_since_last"],
                "fraud_score"     : round(score, 6),
                "is_fraud"        : is_fraud,
            })
            txn_id += 1
            history.record(ts, amount, country)

    df = pd.DataFrame(rows).sort_values(
        ["customer_id", "timestamp_h"]
    ).reset_index(drop=True)
    return df


def _build_profiles(cfg: GeneratorConfig, rng: np.random.Generator) -> list:
    """Sample customer profiles from population priors."""
    country_p = np.array(cfg.country_probs, dtype=float)
    country_p /= country_p.sum()
    profiles = []
    for cid in range(cfg.n_customers):
        home = cfg.countries[rng.choice(len(cfg.countries), p=country_p)]
        profiles.append({
            "customer_id" : cid,
            "home_country": home,
            "amount_mu"   : float(rng.normal(
                cfg.amount_mu_global, cfg.amount_mu_personal_std
            )),
            "time_peak_hr": float(np.clip(
                rng.normal(cfg.time_peak_hour, cfg.time_personal_std_hr),
                0.0, 23.99
            )),

            "risk_score"  : float(rng.beta(1, 20)),
        })
    return profiles


def _run_warmup(
    profiles:  list,
    histories: Dict[int, CustomerHistory],
    cfg:       GeneratorConfig,
    rng:       np.random.Generator,
):

    for w in range(cfg.n_warmup_periods):
        warmup_start = -(cfg.n_warmup_periods - w) * cfg.period_duration_hours
        for prof in profiles:
            home     = prof["home_country"]
            home_idx = cfg.countries.index(home)
            n_txn    = max(1, int(rng.poisson(cfg.txns_per_customer_lambda)))

            timestamps = _sample_arrival_times(
                n_txn=n_txn, peak_hour=prof["time_peak_hr"],
                kappa=cfg.time_kappa, duration_h=cfg.period_duration_hours,
                period_start=warmup_start, rng=rng,
            )
            amounts = rng.lognormal(prof["amount_mu"], cfg.amount_sigma, size=n_txn)
            home_p  = np.full(
                len(cfg.countries),
                (1.0 - cfg.home_country_loyalty) / (len(cfg.countries) - 1)
            )
            home_p[home_idx] = cfg.home_country_loyalty
            countries = [
                cfg.countries[rng.choice(len(cfg.countries), p=home_p)]
                for _ in range(n_txn)
            ]
            hist = histories[prof["customer_id"]]
            for i in range(n_txn):
                hist.record(float(timestamps[i]), float(amounts[i]), countries[i])


# =============================================================================
# SECTION 5 - Public API
# =============================================================================

def generate(
    cfg:     Optional[GeneratorConfig] = None,
    seed:    int = 42,
    verbose: bool = True,
) -> pd.DataFrame:

    if cfg is None:
        cfg = GeneratorConfig()

    rng = np.random.default_rng(seed)

    profiles  = _build_profiles(cfg, rng)
    histories = {p["customer_id"]: CustomerHistory(p["home_country"])
                 for p in profiles}

    _run_warmup(profiles, histories, cfg, rng)

    df = _generate_one_period(
        cfg          = cfg,
        histories    = histories,
        profiles     = profiles,
        period_start = 0.0,
        txn_id_start = 0,
        period_idx   = 1,
        rng          = rng,
    )
    df = df.drop(columns=["period"])

    if verbose:
        _print_summary(df, cfg)

    return df


def generate_multiperiod(
    n_periods: int = 10,
    cfg:       Optional[GeneratorConfig] = None,
    cfg_list:  Optional[List[GeneratorConfig]] = None,
    seed:      int = 42,
    verbose:   bool = True,
    label:     str = "",
) -> pd.DataFrame:
    """Generate multiple periods.
    Pass cfg_list (one config per period) for drift experiments.
    If omitted, cfg is used for every period (stationary baseline).
    """
    base_cfg = cfg if cfg is not None else GeneratorConfig()
    if cfg_list is None:
        cfg_list = [base_cfg] * n_periods
    else:
        n_periods = len(cfg_list)

    rng = np.random.default_rng(seed)
    profiles  = _build_profiles(base_cfg, rng)
    histories = {p["customer_id"]: CustomerHistory(p["home_country"])
                 for p in profiles}
    _run_warmup(profiles, histories, base_cfg, rng)

    all_frames = []
    txn_id = 0

    hdr = f"  {label}  ({n_periods} periods)" if label else f"  Multi-period  ({n_periods} periods)"
    if verbose:
        print(f"\n{'═'*72}")
        print(hdr)
        print(f"{'═'*72}")
        print(f"  {'Per':>3}  {'Txns':>6}  {'Fraud%':>7}  {'AvgAmt':>8}  "
              f"{'NightFr%':>9}  {'ForeignFr%':>11}  {'RemoteFr%':>10}")
        print("  " + "-" * 62)

    for p in range(1, n_periods + 1):
        period_cfg   = cfg_list[p - 1]
        period_start = (p - 1) * base_cfg.period_duration_hours
        period_rng   = np.random.default_rng(rng.integers(0, 2**31))

        df_p = _generate_one_period(
            cfg          = period_cfg,
            histories    = histories,
            profiles     = profiles,
            period_start = period_start,
            txn_id_start = txn_id,
            period_idx   = p,
            rng          = period_rng,
        )
        txn_id += len(df_p)
        all_frames.append(df_p)

        if verbose:
            fraud_pct  = df_p["is_fraud"].mean() * 100
            avg_amt    = df_p["amount"].mean()
            night_mask = (df_p["time_hour"] >= 22) | (df_p["time_hour"] < 6)
            fgn_mask   = df_p["country"] != df_p["home_country"]
            rem_mask   = df_p["is_remote"] == 1
            night_fr   = df_p.loc[night_mask, "is_fraud"].mean() * 100 if night_mask.any() else 0.0
            fgn_fr     = df_p.loc[fgn_mask,  "is_fraud"].mean() * 100 if fgn_mask.any() else 0.0
            rem_fr     = df_p.loc[rem_mask,   "is_fraud"].mean() * 100 if rem_mask.any() else 0.0
            print(f"  {p:>3}  {len(df_p):>6,}  {fraud_pct:>6.2f}%  "
                  f"EUR{avg_amt:>6.1f}  {night_fr:>8.1f}%  {fgn_fr:>10.1f}%  {rem_fr:>9.1f}%")

    df = pd.concat(all_frames, ignore_index=True)
    if verbose:
        overall = df["is_fraud"].mean() * 100
        print("  " + "-" * 62)
        print(f"  Overall fraud rate: {overall:.2f}%  |  Total rows: {len(df):,}\n")
    return df

# =============================================================================
# SECTION 6 - Calibration diagnostics
# =============================================================================

def check_calibration(
    cfg:     Optional[GeneratorConfig] = None,
    seeds:   List[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:

    if cfg   is None: cfg   = GeneratorConfig()
    if seeds is None: seeds = [0, 1, 2, 3, 4]

    records = []
    for s in seeds:
        df = generate(cfg, seed=s, verbose=False)

        night   = (df["time_hour"] >= 22) | (df["time_hour"] < 6)
        foreign = df["country"] != df["home_country"]
        remote  = df["is_remote"] == 1
        top_amt = df["amount"] >= df["amount"].quantile(0.90)
        bot_amt = df["amount"] <= df["amount"].quantile(0.10)

        records.append({
            "seed"              : s,
            "n_txn"             : len(df),
            "fraud_rate"        : round(df["is_fraud"].mean(), 4),
            "fraud_night"       : round(df.loc[night,    "is_fraud"].mean(), 4),
            "fraud_day"         : round(df.loc[~night,   "is_fraud"].mean(), 4),
            "fraud_foreign"     : round(df.loc[foreign,  "is_fraud"].mean(), 4) if foreign.any() else None,
            "fraud_domestic"    : round(df.loc[~foreign, "is_fraud"].mean(), 4),
            "fraud_remote"      : round(df.loc[remote,   "is_fraud"].mean(), 4) if remote.any() else None,
            "fraud_inperson"    : round(df.loc[~remote,  "is_fraud"].mean(), 4),
            "fraud_top_amt"     : round(df.loc[top_amt,  "is_fraud"].mean(), 4),
            "fraud_bot_amt"     : round(df.loc[bot_amt,  "is_fraud"].mean(), 4),
            "fraud_new_country" : round(df.loc[df["is_new_country"]==1, "is_fraud"].mean(), 4)
                                  if (df["is_new_country"]==1).any() else None,
        })

    results = pd.DataFrame(records)

    if verbose:
        print(f"\n{'═'*72}")
        print(f"  Calibration check across {len(seeds)} seeds")
        print(f"{'═'*72}")
        print(results.to_string(index=False))
        print(f"\n  fraud_rate  mean={results['fraud_rate'].mean():.4f}  "
              f"std={results['fraud_rate'].std():.4f}\n")

    return results


# =============================================================================
# SECTION 7 - Summary printer
# =============================================================================

def _print_summary(df: pd.DataFrame, cfg: GeneratorConfig):
    n         = len(df)
    n_fraud   = int(df["is_fraud"].sum())
    fraud_pct = n_fraud / n * 100
    avg_amt   = df["amount"].mean()
    pct_a     = (df["country"] == "A").mean() * 100
    pct_rem   = df["is_remote"].mean() * 100
    avg_score = df["fraud_score"].mean()

    # Timestamp consistency check
    if "timestamp_h" in df.columns:
        derived  = (df["timestamp_h"] % 24).round(3)
        mismatch = (derived - df["time_hour"]).abs().max()
        ts_check = (f"consistent (max diff={mismatch:.6f}h)"
                    if mismatch < 0.01 else f"MISMATCH max={mismatch:.4f}h")
    else:
        ts_check = "n/a"

    print(f"\n{'═'*60}")
    print(f"  Fraud Generator v3")
    print(f"{'═'*60}")
    print(f"  Customers    : {cfg.n_customers}")
    print(f"  Transactions : {n:,}")
    print(f"  Fraud rate   : {fraud_pct:.2f}%  ({n_fraud} fraudulent)")
    print(f"  Avg amount   : EUR{avg_amt:.2f}")
    print(f"  Remote txns  : {pct_rem:.1f}%")
    print(f"  Country A%   : {pct_a:.1f}%")
    print(f"  Avg P(fraud) : {avg_score:.4f}")
    print(f"  ts/hour check: {ts_check}")
    print(f"{'═'*60}\n")


# =============================================================================
# SECTION 8 - Entry point
# =============================================================================
#
# SENSITIVITY ANALYSIS — "When does the model break?"
# ====================================================

if __name__ == "__main__":
    import os
    from dataclasses import replace

    out = r"C:\Users\julie\OneDrive\Documenten\Thesis\outputs"
    os.makedirs(out, exist_ok=True)

    N    = 10
    base = GeneratorConfig()

    # Full-drift target weights (100% = this destination)
    W_START = dict(w_remote=1.4, w_velocity_1h=0.9, w_amount=0.7,
                   w_foreign=1.3, w_night=0.6)
    W_END   = dict(w_remote=2.5, w_velocity_1h=2.0, w_amount=1.5,
                   w_foreign=0.4, w_night=0.1)

    def scaled_cfg(scale):
        """Config where weights moved `scale` fraction toward W_END."""
        return replace(base,
            w_remote      = round(W_START["w_remote"]      + scale*(W_END["w_remote"]      - W_START["w_remote"]),      4),
            w_velocity_1h = round(W_START["w_velocity_1h"] + scale*(W_END["w_velocity_1h"] - W_START["w_velocity_1h"]), 4),
            w_amount      = round(W_START["w_amount"]       + scale*(W_END["w_amount"]      - W_START["w_amount"]),      4),
            w_foreign     = round(W_START["w_foreign"]      + scale*(W_END["w_foreign"]     - W_START["w_foreign"]),     4),
            w_night       = round(W_START["w_night"]        + scale*(W_END["w_night"]       - W_START["w_night"]),       4),
        )

    def gradual_list(scale, n_periods):
        """Linear ramp from baseline (p1) to scaled_cfg (p_n)."""
        return [
            scaled_cfg(scale * (p-1) / (n_periods-1))
            for p in range(1, n_periods+1)
        ]

    saved = []

    # -------------------------------------------------------------------
    # 0. BASELINE
    # -------------------------------------------------------------------
    print("\n" + "="*72)
    print("  0. BASELINE — stationary reference")
    print("="*72)
    df = generate_multiperiod(cfg_list=[base]*N, seed=42, verbose=True,
                               label="Baseline")
    fname = "drift_baseline.csv"
    df.to_csv(f"{out}/{fname}", index=False)
    saved.append(fname)
    print(f"  -> fraud={df['is_fraud'].mean()*100:.3f}%")

    # -------------------------------------------------------------------
    # AXIS 1 — DRIFT MAGNITUDE (16 steps, sudden)
    # Fine-grained at low end to capture the break point precisely
    # -------------------------------------------------------------------
    print("\n" + "="*72)
    print("  AXIS 1 — DRIFT MAGNITUDE (sudden, 16 steps)")
    print("  Periods 1-5: baseline.  Periods 6-10: drifted.")
    print("="*72)

    magnitude_steps = [
        0.10, 0.20, 0.30, 0.40, 0.50,   # fine: 10-50%  (where break likely is)
        0.60, 0.75, 0.90, 1.00,           # medium: 60-100%
        1.20, 1.40, 1.60, 1.80,           # large: 120-180%
        2.00, 2.25, 2.50,                 # extreme: 200-250%
    ]

    for scale in magnitude_steps:
        post = scaled_cfg(scale)
        pct  = int(round(scale * 100))
        df = generate_multiperiod(
            cfg_list=[base]*5 + [post]*5,
            seed=42, verbose=True,
            label=f"Sudden {pct}%"
        )
        fname = f"drift_sudden_mag{pct:03d}pct.csv"
        df.to_csv(f"{out}/{fname}", index=False)
        saved.append(fname)
        pre = df[df.period<=5]["is_fraud"].mean()*100
        pst = df[df.period>5 ]["is_fraud"].mean()*100
        print(f"  -> {pct:3d}%  pre={pre:.3f}%  post={pst:.3f}%")

    # -------------------------------------------------------------------
    # AXIS 2 — DRIFT SPEED (sudden vs gradual, 4 magnitudes)
    # -------------------------------------------------------------------
    print("\n" + "="*72)
    print("  AXIS 2 — DRIFT SPEED (sudden vs gradual, 4 magnitudes)")
    print("="*72)

    speed_pairs = [
        (0.50, "mag050pct"),
        (1.00, "mag100pct"),
        (1.50, "mag150pct"),
        (2.00, "mag200pct"),
    ]

    for scale, slabel in speed_pairs:
        post = scaled_cfg(scale)

        # Sudden
        df = generate_multiperiod(
            cfg_list=[base]*5 + [post]*5,
            seed=42, verbose=True,
            label=f"Speed sudden {int(scale*100)}%"
        )
        fname = f"drift_speed_sudden_{slabel}.csv"
        df.to_csv(f"{out}/{fname}", index=False)
        saved.append(fname)
        print(f"  sudden {int(scale*100)}%: "
              f"pre={df[df.period<=5]['is_fraud'].mean()*100:.3f}%  "
              f"post={df[df.period>5]['is_fraud'].mean()*100:.3f}%")

        # Gradual — same destination
        df = generate_multiperiod(
            cfg_list=gradual_list(scale, N),
            seed=42, verbose=True,
            label=f"Speed gradual {int(scale*100)}%"
        )
        fname = f"drift_speed_gradual_{slabel}.csv"
        df.to_csv(f"{out}/{fname}", index=False)
        saved.append(fname)
        print(f"  gradual {int(scale*100)}%: "
              f"p1={df[df.period==1]['is_fraud'].mean()*100:.3f}%  "
              f"p10={df[df.period==10]['is_fraud'].mean()*100:.3f}%")

    # -------------------------------------------------------------------
    # AXIS 3 — FREEZE DURATION (1 to 7 frozen periods from period 3)
    # -------------------------------------------------------------------
    print("\n" + "="*72)
    print("  AXIS 3 — FREEZE DURATION (1 to 7 periods, starting at period 3)")
    print("="*72)

    frozen_deep = replace(base, fraud_intercept=-50.0)

    # Periods 1-4: baseline (model learns fraud patterns)
    # Periods 5 to 5+k-1: frozen (fraud disappears)
    # Periods 5+k to 10: baseline returns (does model remember?)
    # Max k = 6 so there is always at least 1 recovery period
    for k in range(1, 7):
        cfg_list = [base]*4 + [frozen_deep]*k + [base]*(N-4-k)
        assert len(cfg_list) == N, f"length {len(cfg_list)} != {N}"
        df = generate_multiperiod(
            cfg_list=cfg_list, seed=42, verbose=True,
            label=f"Freeze duration {k}p (p5-p{4+k} frozen, baseline resumes p{5+k})"
        )
        fname = f"drift_freeze_duration_{k}p.csv"
        df.to_csv(f"{out}/{fname}", index=False)
        saved.append(fname)
        fr = df[df.period.isin(range(5, 5+k))]["is_fraud"].mean()*100
        po = df[df.period >= 5+k]["is_fraud"].mean()*100
        print(f"  {k}p frozen (p5-p{4+k}) -> freeze={fr:.4f}%  post-freeze={po:.3f}%")

    # -------------------------------------------------------------------
    # AXIS 4 — FREEZE DEPTH (6 levels, always 3 frozen periods)
    # How much suppression is needed before models forget?
    # intercept goes from -14 (mild) to -50 (near-zero fraud)
    # -------------------------------------------------------------------
    print("\n" + "="*72)
    print("  AXIS 4 — FREEZE DEPTH (3 frozen periods p5-p7, 6 suppression levels)")
    print("  Structure: 4 normal | 3 frozen | 3 normal")
    print(f"  Baseline intercept = {base.fraud_intercept}")
    print("="*72)

    freeze_intercepts = [-14.0, -16.0, -18.0, -20.0, -30.0, -50.0]

    # Same 3-period freeze, always periods 5-7
    # Periods 1-4: baseline, periods 5-7: frozen at varying depth,
    # periods 8-10: baseline returns
    for intercept in freeze_intercepts:
        frozen_shallow = replace(base, fraud_intercept=intercept)
        cfg_list = [base]*4 + [frozen_shallow]*3 + [base]*3
        assert len(cfg_list) == N
        df = generate_multiperiod(
            cfg_list=cfg_list, seed=42, verbose=True,
            label=f"Freeze depth intercept={intercept} (p5-p7 frozen)"
        )
        label_int = str(intercept).replace("-","neg").replace(".","p")
        fname = f"drift_freeze_depth_{label_int}.csv"
        df.to_csv(f"{out}/{fname}", index=False)
        saved.append(fname)
        fr = df[df.period.isin([5,6,7])]["is_fraud"].mean()*100
        po = df[df.period >= 8]["is_fraud"].mean()*100
        print(f"  intercept={intercept:6.1f} -> freeze={fr:.4f}%  post={po:.3f}%")

    # -------------------------------------------------------------------
    # AXIS 5 — FRAUD PATTERN ROTATION
    # 4 named fraud patterns, each active for 3 periods.
    # Pattern A is the baseline. B, C, D are distinct fraud types
    # that each move weights in a different direction from baseline.
    # After cycling through B, C, D the fraudsters return to A.
    # Question: has the model forgotten Pattern A?
    #
    # Pattern A — baseline (CNP moderate, cross-border moderate)
    # Pattern B — CNP surge:     w_remote ↑, w_velocity_1h ↑, w_foreign ↓, w_night ↓
    # Pattern C — cross-border:  w_foreign ↑, w_new_country ↑, w_remote ↓, w_night ↑
    # Pattern D — burst/amount:  w_velocity_1h ↑, w_amount ↑, w_remote ↓, w_foreign ↓
    #
    # Sensitivity: scale how far B, C, D deviate from A.
    # Structure (12 periods): A,A,A | B,B,B | C,C,C | A,A,A (return)
    # -------------------------------------------------------------------
    print("\n" + "="*72)
    print("  AXIS 5 — FRAUD PATTERN ROTATION (A→B→C→A→B, 10 periods)")
    print("  2 periods per pattern. Sensitivity = how different patterns are.")
    print("  Pattern A: baseline.  B: CNP surge.  C: cross-border.")
    print("  Return to A at period 7-8, B at period 9-10 — does the model still recognise them?")
    print("="*72)

    N_ROT = 10   # 10 periods: 2 per pattern × 5 blocks (A, B, C, A, B)

    # Direction vectors for each pattern relative to baseline
    # Includes both main effect weights AND interaction weights
    # Pattern B (CNP surge): strong remote×foreign and remote×night interactions
    # Pattern C (cross-border): strong newcountry×foreign interaction, no remote
    PATTERN_B_DIR = dict(
        w_remote=+1.1, w_velocity_1h=+1.1, w_amount=+0.8,
        w_foreign=-0.9, w_night=-0.5, w_new_country=-0.2,
        # CNP surge activates remote interactions strongly
        w_remote_x_foreign=+0.7, w_remote_x_night=+0.6,
        w_velocity_x_gap=+0.5,   w_newcountry_x_foreign=-0.3,
    )
    PATTERN_C_DIR = dict(
        w_remote=-0.6, w_velocity_1h=-0.3, w_amount=+0.3,
        w_foreign=+1.0, w_night=+0.8, w_new_country=+0.8,
        # Cross-border activates geographic novelty interactions
        w_remote_x_foreign=-0.4, w_remote_x_night=-0.2,
        w_velocity_x_gap=-0.2,   w_newcountry_x_foreign=+0.9,
    )

    def pattern_cfg(direction, scale):
        """Return a config shifted in `direction` by `scale` from baseline."""
        return replace(base,
            w_remote      = round(max(0.01, base.w_remote      + scale * direction.get("w_remote",      0)), 4),
            w_velocity_1h = round(max(0.01, base.w_velocity_1h + scale * direction.get("w_velocity_1h", 0)), 4),
            w_amount      = round(max(0.01, base.w_amount       + scale * direction.get("w_amount",      0)), 4),
            w_foreign     = round(max(0.01, base.w_foreign      + scale * direction.get("w_foreign",     0)), 4),
            w_night       = round(max(0.01, base.w_night        + scale * direction.get("w_night",       0)), 4),
            w_new_country = round(max(0.01, base.w_new_country  + scale * direction.get("w_new_country", 0)), 4),
            # Interaction weights also shift per pattern
            w_remote_x_foreign    = round(max(0.0, base.w_remote_x_foreign    + scale * direction.get("w_remote_x_foreign",    0)), 4),
            w_remote_x_night      = round(max(0.0, base.w_remote_x_night      + scale * direction.get("w_remote_x_night",      0)), 4),
            w_velocity_x_gap      = round(max(0.0, base.w_velocity_x_gap      + scale * direction.get("w_velocity_x_gap",      0)), 4),
            w_newcountry_x_foreign= round(max(0.0, base.w_newcountry_x_foreign+ scale * direction.get("w_newcountry_x_foreign", 0)), 4),
        )

    rotation_scales = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50]

    for scale in rotation_scales:
        pat_A = base                              # Pattern A: always baseline
        pat_B = pattern_cfg(PATTERN_B_DIR, scale) # CNP surge
        pat_C = pattern_cfg(PATTERN_C_DIR, scale) # cross-border

        # A,A | B,B | C,C | A,A | B,B
        cfg_list = [pat_A]*2 + [pat_B]*2 + [pat_C]*2 + [pat_A]*2 + [pat_B]*2
        assert len(cfg_list) == N_ROT

        pct = int(round(scale * 100))
        df = generate_multiperiod(
            cfg_list=cfg_list, seed=42, verbose=True,
            label=f"Rotation {pct}% — A(p1-2) B(p3-4) C(p5-6) A-return(p7-8) B-return(p9-10)"
        )
        fname = f"drift_rotation_{pct:03d}pct.csv"
        df.to_csv(f"{out}/{fname}", index=False)
        saved.append(fname)

        # Key comparison: Pattern A and B fraud rates at start vs at return
        fr_A_start  = df[df.period.isin([1,2])]["is_fraud"].mean()*100
        fr_B        = df[df.period.isin([3,4])]["is_fraud"].mean()*100
        fr_C        = df[df.period.isin([5,6])]["is_fraud"].mean()*100
        fr_A_return = df[df.period.isin([7,8])]["is_fraud"].mean()*100
        fr_B_return = df[df.period.isin([9,10])]["is_fraud"].mean()*100
        print(f"  {pct:3d}%  A-start={fr_A_start:.3f}%  B={fr_B:.3f}%  "
              f"C={fr_C:.3f}%  A-return={fr_A_return:.3f}%  B-return={fr_B_return:.3f}%")

    # -------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------
    print("\n" + "="*72)
    print(f"  DONE — {len(saved)} datasets")
    print("="*72)

    axis1 = [f for f in saved if "sudden_mag" in f]
    axis2 = [f for f in saved if "speed" in f]
    axis3 = [f for f in saved if "duration" in f]
    axis4 = [f for f in saved if "depth" in f]
    axis5 = [f for f in saved if "rotation" in f]

    print(f"\n  Axis 1 — magnitude  : {len(axis1):2d} datasets  (drift_sudden_magXXXpct.csv)")
    print(f"  Axis 2 — speed      : {len(axis2):2d} datasets  (drift_speed_sudden/gradual_magXXXpct.csv)")
    print(f"  Axis 3 — freeze dur.: {len(axis3):2d} datasets  (drift_freeze_duration_Np.csv)")
    print(f"  Axis 4 — freeze dep.: {len(axis4):2d} datasets  (drift_freeze_depth_*.csv)")
    print(f"  Axis 5 — rotation   : {len(axis5):2d} datasets  (drift_rotation_XXXpct.csv)")
    print(f"  Baseline            :  1 dataset   (drift_baseline.csv)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Total               : {len(saved):2d} datasets\n")
    print(f"  Output: {out}\n")
    print("""  HOW TO USE FOR YOUR THESIS:
  Axis 1: x = drift magnitude,      y = avg PR-AUC + forgetting → break point
  Axis 2: sudden vs gradual lines,   y = avg PR-AUC + forgetting → does speed matter?
  Axis 3: x = freeze duration,       y = post-freeze avg PR-AUC  → how fast do models forget?
  Axis 4: x = freeze depth,          y = post-freeze avg PR-AUC  → does partial quiet matter?
  Axis 5: x = pattern separation,    y = avg PR-AUC on A-return  → does rotation fool the model?
  A bank can locate their scenario on these curves and read off which CL method survives.
""")
