"""
JST APEX QUANT — ADVANCED PROBABILITY & STATISTICS MODULE
Version: 1.0
Purpose: Research add-on for the existing XAU/USD V8.5 agent.
Mode: Research / diagnostics only. No broker orders.

Install:
    pip install numpy pandas scipy statsmodels

Optional:
    pip install arch hmmlearn

Quick test:
    python jst_advanced_quant.py

Integration examples:
    from jst_advanced_quant import (
        advanced_statistics, advanced_probability,
        monte_carlo_gbm, risk_report, validate_probabilities
    )

    returns = m5_df["close"].astype(float).pct_change().dropna()
    stats = advanced_statistics(returns)
    prob = advanced_probability(wins=24, losses=16)
    mc = monte_carlo_gbm(returns, horizon=20, paths=5000)
    risk = risk_report(returns)
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import genpareto
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _clean_returns(values: Iterable[float]) -> pd.Series:
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    return s


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _require_sample(s: pd.Series, minimum: int = 3) -> None:
    if len(s) < minimum:
        raise ValueError(f"Minimal {minimum} observasi valid; tersedia {len(s)}.")


# ---------------------------------------------------------------------
# 1. Advanced descriptive and distribution statistics
# ---------------------------------------------------------------------

def distribution_report(returns: Iterable[float]) -> Dict[str, Any]:
    r = _clean_returns(returns)
    _require_sample(r, 4)

    x = r.to_numpy()
    q = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    std = float(np.std(x, ddof=1))
    jb_stat, jb_p = stats.jarque_bera(x) if len(x) >= 8 else (np.nan, np.nan)

    # Student-t degrees of freedom fit is descriptive, not a forecast.
    try:
        t_df, t_loc, t_scale = stats.t.fit(x)
    except Exception:
        t_df, t_loc, t_scale = np.nan, np.nan, np.nan

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "variance": float(np.var(x, ddof=1)),
        "std": std,
        "skewness": float(stats.skew(x, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(x, fisher=True, bias=False)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "quantiles": dict(zip(
            ["q01", "q05", "q25", "q50", "q75", "q95", "q99"],
            map(float, q)
        )),
        "jarque_bera_stat": _finite(jb_stat, np.nan),
        "jarque_bera_pvalue": _finite(jb_p, np.nan),
        "student_t_fit": {
            "df": _finite(t_df, np.nan),
            "loc": _finite(t_loc, np.nan),
            "scale": _finite(t_scale, np.nan),
        },
        "warning": "Distribution fit describes the sample; it does not prove future returns follow that distribution."
    }


def volatility_report(returns: Iterable[float], ewma_lambda: float = 0.94) -> Dict[str, Any]:
    r = _clean_returns(returns)
    _require_sample(r, 5)
    if not 0.0 < ewma_lambda < 1.0:
        raise ValueError("ewma_lambda harus berada di antara 0 dan 1.")

    x = r.to_numpy()
    ewma_var = float(np.var(x[: min(20, len(x))], ddof=1))
    for value in x:
        ewma_var = ewma_lambda * ewma_var + (1.0 - ewma_lambda) * float(value) ** 2

    rolling = r.rolling(min(20, len(r))).std().dropna()
    return {
        "realized_volatility_per_bar": float(r.std(ddof=1)),
        "ewma_volatility_per_bar": float(math.sqrt(max(ewma_var, 0.0))),
        "rolling_volatility_latest": float(rolling.iloc[-1]) if len(rolling) else np.nan,
        "ewma_lambda": float(ewma_lambda),
        "annualized": False,
        "note": "Tidak diannualisasi karena frekuensi bar dan kalender trading harus ditentukan secara eksplisit."
    }


# ---------------------------------------------------------------------
# 2. Correlation, regression and time-series diagnostics
# ---------------------------------------------------------------------

def correlation_report(data: pd.DataFrame) -> Dict[str, Any]:
    numeric = data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if numeric.shape[1] < 2:
        raise ValueError("Masukkan minimal dua kolom numerik.")
    return {
        "pearson": numeric.corr(method="pearson").to_dict(),
        "spearman": numeric.corr(method="spearman").to_dict(),
        "covariance": numeric.cov().to_dict(),
        "observations": int(numeric.dropna().shape[0]),
    }


def ols_regression(y: Iterable[float], predictors: pd.DataFrame) -> Dict[str, Any]:
    frame = predictors.select_dtypes(include=[np.number]).copy()
    frame["__target__"] = pd.Series(y, index=frame.index)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) <= frame.shape[1] + 2:
        raise ValueError("Observasi tidak cukup untuk OLS dengan jumlah prediktor ini.")

    target = frame.pop("__target__")
    X = sm.add_constant(frame, has_constant="add")
    fit = sm.OLS(target, X).fit()

    return {
        "n": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "adjusted_r_squared": float(fit.rsquared_adj),
        "f_statistic": _finite(fit.fvalue, np.nan),
        "f_pvalue": _finite(fit.f_pvalue, np.nan),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "coefficients": fit.params.to_dict(),
        "pvalues": fit.pvalues.to_dict(),
        "confidence_intervals": fit.conf_int().to_dict(),
        "residual_std": float(np.std(fit.resid, ddof=1)),
        "diagnostic_warning": "OLS association is not causation. Check residual autocorrelation, heteroskedasticity, collinearity and out-of-sample stability."
    }


def time_series_diagnostics(returns: Iterable[float], max_lags: int = 20) -> Dict[str, Any]:
    r = _clean_returns(returns)
    _require_sample(r, max(30, max_lags + 5))
    x = r.to_numpy()
    nlags = min(max_lags, max(1, len(x) // 4))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(x, autolag="AIC")
        try:
            kp = kpss(x, regression="c", nlags="auto")
            kpss_result = {"statistic": float(kp[0]), "pvalue": float(kp[1])}
        except Exception as exc:
            kpss_result = {"error": str(exc)}

    return {
        "adf_statistic": float(adf[0]),
        "adf_pvalue": float(adf[1]),
        "adf_used_lag": int(adf[2]),
        "kpss": kpss_result,
        "acf": acf(x, nlags=nlags, fft=True).tolist(),
        "pacf": pacf(x, nlags=nlags, method="ywm").tolist(),
        "interpretation": "ADF null: unit root. KPSS null: stationarity. Interpret both tests together."
    }


def structural_break_cusum(returns: Iterable[float]) -> Dict[str, Any]:
    r = _clean_returns(returns)
    _require_sample(r, 30)
    x = r.to_numpy()
    centered = x - np.mean(x)
    scale = np.std(centered, ddof=1)
    if scale <= 0:
        return {"statistic": 0.0, "flag": False, "reason": "Zero variance"}
    path = np.cumsum(centered) / (scale * math.sqrt(len(x)))
    statistic = float(np.max(np.abs(path)))
    # Conservative screening threshold; not a formal p-value.
    return {
        "statistic": statistic,
        "flag": bool(statistic > 1.36),
        "threshold": 1.36,
        "note": "Screening diagnostic only; use a formally specified CUSUM test for inferential claims."
    }


# ---------------------------------------------------------------------
# 3. Inference, bootstrap and robust statistics
# ---------------------------------------------------------------------

def mean_confidence_interval(returns: Iterable[float], confidence: float = 0.95) -> Dict[str, float]:
    r = _clean_returns(returns)
    _require_sample(r, 2)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence harus antara 0 dan 1.")
    n = len(r)
    mean = float(r.mean())
    se = float(stats.sem(r))
    critical = float(stats.t.ppf((1 + confidence) / 2, n - 1))
    return {
        "mean": mean,
        "standard_error": se,
        "confidence": confidence,
        "lower": mean - critical * se,
        "upper": mean + critical * se,
        "n": int(n)
    }


def block_bootstrap_mean(
    returns: Iterable[float],
    iterations: int = 2000,
    block_size: int = 5,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    r = _clean_returns(returns).to_numpy()
    _require_sample(pd.Series(r), 10)
    if iterations < 100 or block_size < 1:
        raise ValueError("iterations minimal 100 dan block_size minimal 1.")

    rng = np.random.default_rng(seed)
    n = len(r)
    n_blocks = int(math.ceil(n / block_size))
    estimates = np.empty(iterations)

    for i in range(iterations):
        sampled = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, n))
            sampled.extend(r[(start + np.arange(block_size)) % n])
        estimates[i] = np.mean(sampled[:n])

    alpha = 1.0 - confidence
    return {
        "bootstrap_mean": float(np.mean(estimates)),
        "lower": float(np.quantile(estimates, alpha / 2)),
        "upper": float(np.quantile(estimates, 1 - alpha / 2)),
        "confidence": confidence,
        "iterations": iterations,
        "block_size": block_size,
        "method": "circular moving-block bootstrap"
    }


def hypothesis_test_mean(returns: Iterable[float], null_mean: float = 0.0) -> Dict[str, float]:
    r = _clean_returns(returns)
    _require_sample(r, 3)
    result = stats.ttest_1samp(r.to_numpy(), popmean=null_mean)
    return {
        "t_statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "sample_mean": float(r.mean()),
        "null_mean": float(null_mean),
        "n": int(len(r))
    }


def robust_statistics(returns: Iterable[float]) -> Dict[str, float]:
    r = _clean_returns(returns)
    _require_sample(r, 3)
    median = float(r.median())
    mad = float(np.median(np.abs(r.to_numpy() - median)))
    return {
        "median": median,
        "mad": mad,
        "robust_sigma": float(1.4826 * mad),
        "trimmed_mean_10pct": float(stats.trim_mean(r.to_numpy(), 0.10)),
        "winsorized_std_5pct": float(np.std(stats.mstats.winsorize(r.to_numpy(), limits=0.05), ddof=1))
    }


# ---------------------------------------------------------------------
# 4. Advanced probability and Bayesian inference
# ---------------------------------------------------------------------

def bayesian_win_probability(
    wins: int,
    losses: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_level: float = 0.95
) -> Dict[str, float]:
    if min(wins, losses) < 0 or prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Counts tidak boleh negatif; prior alpha/beta harus positif.")
    if not 0 < credible_level < 1:
        raise ValueError("credible_level harus antara 0 dan 1.")

    a = prior_alpha + wins
    b = prior_beta + losses
    tail = (1 - credible_level) / 2
    return {
        "posterior_alpha": float(a),
        "posterior_beta": float(b),
        "posterior_mean": float(a / (a + b)),
        "posterior_median": float(stats.beta.median(a, b)),
        "credible_lower": float(stats.beta.ppf(tail, a, b)),
        "credible_upper": float(stats.beta.ppf(1 - tail, a, b)),
        "credible_level": credible_level,
        "observations": int(wins + losses)
    }


def conditional_probability(
    frame: pd.DataFrame,
    condition: pd.Series,
    event: pd.Series,
    smoothing: float = 1.0
) -> Dict[str, float]:
    if len(frame) != len(condition) or len(frame) != len(event):
        raise ValueError("Panjang condition/event harus sama dengan frame.")
    c = pd.Series(condition, index=frame.index).fillna(False).astype(bool)
    e = pd.Series(event, index=frame.index).fillna(False).astype(bool)
    n = int(c.sum())
    successes = int((c & e).sum())
    p = (successes + smoothing) / (n + 2 * smoothing) if n + 2 * smoothing else np.nan
    return {
        "probability": float(p),
        "successes": successes,
        "condition_count": n,
        "smoothing": float(smoothing),
        "source": "conditional empirical frequency with beta(1,1)-style smoothing"
    }


def markov_transition_matrix(states: Sequence[str]) -> Dict[str, Any]:
    s = pd.Series(states).dropna().astype(str)
    _require_sample(s, 3)
    labels = sorted(s.unique().tolist())
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype=float)
    for a, b in zip(s.iloc[:-1], s.iloc[1:]):
        counts.loc[a, b] += 1
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    probs = counts.div(row_totals, axis=0).fillna(0.0)
    return {
        "states": labels,
        "counts": counts.to_dict(),
        "transition_probabilities": probs.to_dict(),
        "note": "First-order Markov assumption; test stability and compare with higher-order alternatives."
    }


# ---------------------------------------------------------------------
# 5. Predictive probability calibration
# ---------------------------------------------------------------------

def probability_validation(probabilities: Iterable[float], outcomes: Iterable[int]) -> Dict[str, Any]:
    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    if len(p) != len(y) or len(p) == 0:
        raise ValueError("Probabilities dan outcomes harus sama panjang dan tidak kosong.")
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if len(p) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Outcomes harus berupa 0/1 dan data harus valid.")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilitas harus berada di rentang 0 sampai 1.")

    eps = 1e-12
    clipped = np.clip(p, eps, 1 - eps)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))

    edges = np.linspace(0, 1, 11)
    bins = []
    for i in range(10):
        mask_bin = (p >= edges[i]) & ((p < edges[i + 1]) if i < 9 else (p <= edges[i + 1]))
        if mask_bin.any():
            bins.append({
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "count": int(mask_bin.sum()),
                "mean_prediction": float(p[mask_bin].mean()),
                "observed_frequency": float(y[mask_bin].mean())
            })
    return {
        "n": int(len(p)),
        "brier_score": brier,
        "log_loss": logloss,
        "reliability_bins": bins,
        "note": "Lower Brier/log-loss is better; calibration needs enough observations per bin."
    }


# ---------------------------------------------------------------------
# 6. Monte Carlo and jump diffusion
# ---------------------------------------------------------------------

def monte_carlo_gbm(
    returns: Iterable[float],
    horizon: int = 20,
    paths: int = 5000,
    initial_price: float = 1.0,
    seed: int = 42
) -> Dict[str, Any]:
    r = _clean_returns(returns).to_numpy()
    _require_sample(pd.Series(r), 20)
    if horizon < 1 or paths < 100 or initial_price <= 0:
        raise ValueError("horizon >= 1, paths >= 100, initial_price > 0.")

    rng = np.random.default_rng(seed)
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    shocks = rng.normal(mu, sigma, size=(paths, horizon))
    log_paths = np.cumsum(shocks, axis=1)
    price_paths = initial_price * np.exp(log_paths)
    terminal = price_paths[:, -1]
    return {
        "initial_price": float(initial_price),
        "horizon_bars": int(horizon),
        "paths": int(paths),
        "mean_terminal": float(np.mean(terminal)),
        "median_terminal": float(np.median(terminal)),
        "q05_terminal": float(np.quantile(terminal, 0.05)),
        "q95_terminal": float(np.quantile(terminal, 0.95)),
        "probability_above_initial": float(np.mean(terminal > initial_price)),
        "sample_paths": price_paths[: min(100, paths)].tolist(),
        "assumptions": "IID Gaussian log-return shocks; baseline scenario model, not a calibrated market forecast."
    }


def jump_diffusion_simulation(
    returns: Iterable[float],
    horizon: int = 20,
    paths: int = 5000,
    jump_probability_per_bar: float = 0.01,
    jump_mean: float = 0.0,
    jump_std: float = 0.01,
    initial_price: float = 1.0,
    seed: int = 42
) -> Dict[str, Any]:
    r = _clean_returns(returns).to_numpy()
    _require_sample(pd.Series(r), 20)
    if not 0 <= jump_probability_per_bar <= 1 or jump_std < 0:
        raise ValueError("Parameter jump tidak valid.")
    rng = np.random.default_rng(seed)
    base = rng.normal(np.mean(r), np.std(r, ddof=1), (paths, horizon))
    jump_mask = rng.random((paths, horizon)) < jump_probability_per_bar
    jumps = rng.normal(jump_mean, jump_std, (paths, horizon)) * jump_mask
    terminal = initial_price * np.exp(np.sum(base + jumps, axis=1))
    return {
        "mean_terminal": float(terminal.mean()),
        "median_terminal": float(np.median(terminal)),
        "q01_terminal": float(np.quantile(terminal, 0.01)),
        "q05_terminal": float(np.quantile(terminal, 0.05)),
        "q95_terminal": float(np.quantile(terminal, 0.95)),
        "q99_terminal": float(np.quantile(terminal, 0.99)),
        "probability_above_initial": float(np.mean(terminal > initial_price)),
        "jump_probability_per_bar": float(jump_probability_per_bar),
        "paths": int(paths),
        "horizon_bars": int(horizon),
        "note": "Jump parameters must be estimated/validated from appropriate high-frequency or event data."
    }


# ---------------------------------------------------------------------
# 7. EVT tail risk, VaR, Expected Shortfall
# ---------------------------------------------------------------------

def risk_report(returns: Iterable[float], confidence: float = 0.95) -> Dict[str, float]:
    r = _clean_returns(returns)
    _require_sample(r, 20)
    if not 0.5 < confidence < 1:
        raise ValueError("confidence harus di atas 0.5 dan di bawah 1.")

    x = r.to_numpy()
    loss = -x
    var = float(np.quantile(loss, confidence))
    tail = loss[loss >= var]
    es = float(np.mean(tail)) if len(tail) else var

    downside = x[x < 0]
    downside_dev = float(np.sqrt(np.mean(np.minimum(x, 0) ** 2)))
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))

    wealth = np.cumprod(1 + x)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1

    return {
        "confidence": confidence,
        "historical_var_loss_per_bar": var,
        "historical_expected_shortfall_per_bar": es,
        "mean_return_per_bar": mean,
        "volatility_per_bar": std,
        "sharpe_per_bar_unannualized": mean / std if std > 0 else np.nan,
        "sortino_per_bar_unannualized": mean / downside_dev if downside_dev > 0 else np.nan,
        "max_drawdown": float(np.min(drawdown)),
        "downside_observations": int(len(downside))
    }


def evt_gpd_tail_report(returns: Iterable[float], threshold_quantile: float = 0.90) -> Dict[str, Any]:
    r = _clean_returns(returns)
    _require_sample(r, 100)
    if not 0.80 <= threshold_quantile < 1:
        raise ValueError("threshold_quantile disarankan 0.80 sampai kurang dari 1.")

    losses = -r.to_numpy()
    threshold = float(np.quantile(losses, threshold_quantile))
    excess = losses[losses > threshold] - threshold
    if len(excess) < 20:
        return {
            "status": "INSUFFICIENT_TAIL_SAMPLE",
            "threshold": threshold,
            "exceedances": int(len(excess))
        }

    shape, loc, scale = genpareto.fit(excess, floc=0)
    return {
        "status": "FITTED_DIAGNOSTIC",
        "threshold_loss": threshold,
        "exceedances": int(len(excess)),
        "shape_xi": float(shape),
        "scale_beta": float(scale),
        "note": "GPD tail extrapolation is highly threshold-sensitive; validate stability and do not extrapolate blindly."
    }


# ---------------------------------------------------------------------
# 8. Optional GARCH model
# ---------------------------------------------------------------------

def garch_volatility_report(returns: Iterable[float]) -> Dict[str, Any]:
    r = _clean_returns(returns)
    _require_sample(r, 100)
    try:
        from arch import arch_model
    except ImportError:
        return {
            "status": "OPTIONAL_DEPENDENCY_MISSING",
            "install": "pip install arch"
        }

    # Scale returns for numerical stability; forecast variance is rescaled.
    scaled = r * 100.0
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    fit = model.fit(disp="off", show_warning=False)
    forecast = fit.forecast(horizon=1, reindex=False)
    variance_scaled = float(forecast.variance.iloc[-1, 0])
    return {
        "status": "FITTED",
        "model": "GARCH(1,1)-Student-t",
        "conditional_volatility_latest": float(fit.conditional_volatility.iloc[-1] / 100.0),
        "next_bar_volatility_forecast": float(math.sqrt(max(variance_scaled, 0.0)) / 100.0),
        "parameters": {k: float(v) for k, v in fit.params.items()},
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "convergence_flag": int(fit.convergence_flag)
    }


# ---------------------------------------------------------------------
# 9. One-call research report
# ---------------------------------------------------------------------

def advanced_statistics(returns: Iterable[float]) -> Dict[str, Any]:
    r = _clean_returns(returns)
    return {
        "distribution": distribution_report(r),
        "volatility": volatility_report(r),
        "robust": robust_statistics(r),
        "confidence_interval": mean_confidence_interval(r),
        "mean_hypothesis_test": hypothesis_test_mean(r),
        "time_series": time_series_diagnostics(r),
        "structural_break_screen": structural_break_cusum(r),
        "risk": risk_report(r),
        "evt_tail": evt_gpd_tail_report(r) if len(r) >= 100 else {
            "status": "INSUFFICIENT_DATA",
            "required_observations": 100,
            "available": int(len(r))
        }
    }


def advanced_probability(wins: int, losses: int) -> Dict[str, Any]:
    return {
        "bayesian_win_probability": bayesian_win_probability(wins, losses),
        "interpretation": "Posterior estimate of win probability under a Beta-Binomial model; it is not a forward guarantee."
    }


# ---------------------------------------------------------------------
# 10. Synthetic self-test — verifies code paths, not trading edge
# ---------------------------------------------------------------------

def self_test() -> None:
    rng = np.random.default_rng(7)
    synthetic_returns = pd.Series(rng.standard_t(df=5, size=600) * 0.001)
    report = advanced_statistics(synthetic_returns)
    bayes = advanced_probability(24, 16)
    mc = monte_carlo_gbm(synthetic_returns, horizon=10, paths=1000)
    risk = risk_report(synthetic_returns)
    calibration = probability_validation(
        probabilities=np.clip(rng.normal(0.55, 0.15, 200), 0.01, 0.99),
        outcomes=rng.binomial(1, 0.55, 200)
    )

    print("=" * 64)
    print("JST APEX QUANT — ADVANCED MODULE SELF-TEST")
    print("=" * 64)
    print("Distribution observations:", report["distribution"]["n"])
    print("Skewness:", round(report["distribution"]["skewness"], 5))
    print("Excess kurtosis:", round(report["distribution"]["excess_kurtosis"], 5))
    print("ADF p-value:", round(report["time_series"]["adf_pvalue"], 5))
    print("Bayesian posterior win probability:", round(bayes["bayesian_win_probability"]["posterior_mean"] * 100, 2), "%")
    print("Monte Carlo probability above initial:", round(mc["probability_above_initial"] * 100, 2), "%")
    print("Historical VaR:", round(risk["historical_var_loss_per_bar"], 6))
    print("Brier score:", round(calibration["brier_score"], 5))
    print("SELF-TEST PASSED — synthetic data only; no evidence of trading profitability.")
    print("=" * 64)


if __name__ == "__main__":
    self_test()
