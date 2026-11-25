import math
import threading
import time

import numpy as np
from numba import njit, prange

from app.simulation import GPU_AVAILABLE, cp
from app.schemas import EvaluateRequest

SQRT2 = math.sqrt(2.0)
LOG_LO = math.log(1e-3)  # log-spot search bounds for barrier bisection
LOG_HI = math.log(1e5)
BISECT_STEPS = 80
BLOCK = 256
MAX_KEYS = 32  # spreads per GPU launch; must match MAX_KEYS in _CUDA_SOURCE
GPU_BATCH = 1_000_000
CPU_BATCH = 200_000
SEED = 42
CHANNELS = ("sum", "sumsq", "wins", "profit", "stop", "days")

_lock = threading.Lock()
_gpu_kernel = None


_CUDA_SOURCE = r"""
#define MAX_KEYS 32
#define BLOCK 256

__device__ float put_value(float spot, float strike, float tau, float rate, float q, float sigma) {
    if (tau <= 0.0f) return fmaxf(strike - spot, 0.0f);
    float s = spot * expf(-q * tau);
    float vol = sigma * sqrtf(tau);
    float d1 = (logf(s / strike) + (rate + 0.5f * sigma * sigma) * tau) / vol;
    float d2 = d1 - vol;
    return strike * expf(-rate * tau) * 0.5f * erfcf(d2 * 0.70710678f)
         - s * 0.5f * erfcf(d1 * 0.70710678f);
}

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long &s) {
    unsigned long long z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__device__ __forceinline__ unsigned long long init_state(unsigned long long seed, unsigned long long tid) {
    return mix64(mix64(seed ^ 0xD1B54A32D192ED03ull) + tid * 0x9E3779B97F4A7C15ull);
}

__device__ __forceinline__ float uniform01(unsigned long long &s) {
    return ((float)(splitmix64(s) >> 41) + 0.5f) * (1.0f / 8388608.0f);
}

extern "C" __global__ void evaluate_paths(
        const unsigned long long n_paths, const unsigned long long seed,
        const int n_k, const int days, const int has_stop,
        const float* shortk, const float* longk, const float* sig_s, const float* sig_l,
        const float* credit, const float* pnl_profit, const float* pnl_stop,
        const float* bprofit, const float* bstop,
        const float spot0, const float rate, const float q, const float sigma, const float mu,
        const float width, const float tau_close, double* totals) {
    const float unset = __int_as_float(0x7fc00000);
    const float var_day = sigma * sigma / 365.0f;
    const float drift_day = (mu - 0.5f * sigma * sigma) / 365.0f;
    const float vol_day = sigma / sqrtf(365.0f);

    double acc[6][MAX_KEYS];
    for (int c = 0; c < 6; c++)
        for (int k = 0; k < MAX_KEYS; k++) acc[c][k] = 0.0;
    double unclosed = 0.0;

    const unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long state = init_state(seed, tid);

    for (unsigned long long p = tid; p < n_paths; p += stride) {
        float pnl[MAX_KEYS];
        int day_k[MAX_KEYS];
        unsigned char hit_p[MAX_KEYS];
        unsigned char hit_s[MAX_KEYS];
        for (int k = 0; k < n_k; k++) {
            pnl[k] = unset;
            day_k[k] = 0;
            hit_p[k] = 0;
            hit_s[k] = 0;
        }

        float xa = logf(spot0);
        for (int t = 0; t < days; t++) {
            float zt = normcdfinvf(uniform01(state));
            if (!isfinite(zt)) unclosed += 1.0;
            float ut = uniform01(state);
            float xb = xa + drift_day + vol_day * zt;
            float d = xb - xa;
            float m = xa + 0.5f * (d + sqrtf(d * d - 2.0f * var_day * logf(ut)));
            float mn = 0.0f;
            if (has_stop) {
                float ul = uniform01(state);
                mn = xa + 0.5f * (d - sqrtf(d * d - 2.0f * var_day * logf(ul)));
            }
            for (int k = 0; k < n_k; k++) {
                if (!isnan(pnl[k])) continue;
                if (has_stop && mn <= bstop[k * days + t]) {
                    pnl[k] = pnl_stop[k];
                    hit_s[k] = 1;
                    day_k[k] = t + 1;
                } else if (m >= bprofit[k * days + t]) {
                    pnl[k] = pnl_profit[k];
                    hit_p[k] = 1;
                    day_k[k] = t + 1;
                }
            }
            xa = xb;
        }

        float price = expf(xa);
        for (int k = 0; k < n_k; k++) {
            if (isnan(pnl[k])) {
                float v = put_value(price, shortk[k], tau_close, rate, q, sig_s[k])
                        - put_value(price, longk[k], tau_close, rate, q, sig_l[k]);
                pnl[k] = credit[k] - v;
                day_k[k] = days;
            }
            if (!isfinite(pnl[k])) { unclosed += 1.0; continue; }
            double v = pnl[k];
            acc[0][k] += v;
            acc[1][k] += v * v;
            acc[2][k] += (v > 0.0) ? 1.0 : 0.0;
            acc[3][k] += hit_p[k];
            acc[4][k] += hit_s[k];
            acc[5][k] += day_k[k];
        }
    }

    __shared__ double red[BLOCK];
    const int n_vals = 6 * n_k + 1;
    for (int j = 0; j < n_vals; j++) {
        double v = (j < 6 * n_k) ? acc[j / n_k][j % n_k] : unclosed;
        red[threadIdx.x] = v;
        __syncthreads();
        for (unsigned int off = BLOCK / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) atomicAdd(&totals[j], red[0]);
        __syncthreads();
    }
}
"""


@njit(cache=True)
def _put_value(spot, strike, tau, rate, q, sigma):
    if tau <= 0.0:
        return max(strike - spot, 0.0)
    s = spot * math.exp(-q * tau)
    vol = sigma * math.sqrt(tau)
    d1 = (math.log(s / strike) + (rate + 0.5 * sigma * sigma) * tau) / vol
    d2 = d1 - vol
    return (strike * math.exp(-rate * tau) * 0.5 * math.erfc(d2 / SQRT2)
            - s * 0.5 * math.erfc(d1 / SQRT2))


@njit(cache=True)
def _spread_value(spot, short_k, long_k, tau, rate, q, sig_s, sig_l):
    return (_put_value(spot, short_k, tau, rate, q, sig_s)
            - _put_value(spot, long_k, tau, rate, q, sig_l))


@njit(cache=True)
def _crossing(short_k, long_k, tau, rate, q, sig_s, sig_l, target):
    # log-spot where the spread value equals target; value falls as spot rises
    lo = LOG_LO
    hi = LOG_HI
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if _spread_value(math.exp(mid), short_k, long_k, tau, rate, q, sig_s, sig_l) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@njit(cache=True)
def _barriers(short_k, long_k, sig_s, sig_l, credit, rate, q, dte, days, profit_frac, stop_mult, has_stop):
    n = short_k.shape[0]
    bp = np.empty((n, days))
    bs = np.full((n, days), -np.inf)
    for k in range(n):
        target_p = (1.0 - profit_frac) * credit[k]
        target_s = (1.0 + stop_mult) * credit[k]
        for t in range(days):
            tau = (dte - (t + 0.5)) / 365.0  # mid-day approximation
            bp[k, t] = _crossing(short_k[k], long_k[k], tau, rate, q, sig_s[k], sig_l[k], target_p)
            if has_stop != 0:
                if _spread_value(math.exp(LOG_LO), short_k[k], long_k[k], tau, rate, q,
                                 sig_s[k], sig_l[k]) >= target_s:
                    bs[k, t] = _crossing(short_k[k], long_k[k], tau, rate, q, sig_s[k], sig_l[k], target_s)
    return bp, bs


@njit(parallel=True, cache=True)
def _cpu_paths(n, consts, short_k, long_k, sig_s, sig_l, credit, pnl_profit, pnl_stop,
               bp, bs, days, has_stop, pnl, dayout, hit_p, hit_s):
    spot0 = consts[0]
    mu = consts[1]
    sigma = consts[2]
    rate = consts[3]
    q = consts[4]
    tau_close = consts[5]
    var_day = sigma * sigma / 365.0
    drift_day = (mu - 0.5 * sigma * sigma) / 365.0
    vol_day = sigma / math.sqrt(365.0)
    n_s = short_k.shape[0]
    for i in prange(n):
        for k in range(n_s):
            pnl[i, k] = np.nan
            dayout[i, k] = 0
            hit_p[i, k] = 0
            hit_s[i, k] = 0
        xa = math.log(spot0)
        for t in range(days):
            z = np.random.standard_normal()
            u = 1.0 - np.random.random()
            xb = xa + drift_day + vol_day * z
            d = xb - xa
            m = xa + 0.5 * (d + math.sqrt(d * d - 2.0 * var_day * math.log(u)))
            mn = 0.0
            if has_stop != 0:
                u2 = 1.0 - np.random.random()
                mn = xa + 0.5 * (d - math.sqrt(d * d - 2.0 * var_day * math.log(u2)))
            for k in range(n_s):
                if np.isnan(pnl[i, k]):
                    if has_stop != 0 and mn <= bs[k, t]:
                        pnl[i, k] = pnl_stop[k]
                        hit_s[i, k] = 1
                        dayout[i, k] = t + 1
                    elif m >= bp[k, t]:
                        pnl[i, k] = pnl_profit[k]
                        hit_p[i, k] = 1
                        dayout[i, k] = t + 1
            xa = xb
        price = math.exp(xa)
        for k in range(n_s):
            if np.isnan(pnl[i, k]):
                v = (_spread_value(price, short_k[k], long_k[k], tau_close, rate, q, sig_s[k], sig_l[k]))
                pnl[i, k] = credit[k] - v
                dayout[i, k] = days


def _get_gpu_kernel():
    global _gpu_kernel
    if _gpu_kernel is None:
        _gpu_kernel = cp.RawKernel(_CUDA_SOURCE, "evaluate_paths")
    return _gpu_kernel


def _build_spreads(req: EvaluateRequest):
    by_strike = {round(q.strike, 6): q for q in req.strikes}
    rows = []
    for short in req.strikes:
        long = by_strike.get(round(short.strike - req.width, 6))
        if long is None:
            continue
        credit = short.mid - long.mid
        if not (0.0 < credit < req.width):
            continue
        rows.append((short, long, credit))
    return rows


def _new_accumulator(n_k):
    return {name: np.zeros(n_k, dtype=np.float64) for name in CHANNELS}


def _accumulate(acc, pnl, dayout, hit_p, hit_s):
    if not np.isfinite(pnl).all():
        raise RuntimeError("simulation produced an unclosed or invalid spread")
    p = pnl.astype(np.float64)
    acc["sum"] += p.sum(axis=0)
    acc["sumsq"] += (p * p).sum(axis=0)
    acc["wins"] += (p > 0).sum(axis=0)
    acc["profit"] += hit_p.sum(axis=0, dtype=np.float64)
    acc["stop"] += hit_s.sum(axis=0, dtype=np.float64)
    acc["days"] += dayout.sum(axis=0, dtype=np.float64)


def _run_cpu(paths, cfg, arr):
    n_k = arr["short"].shape[0]
    acc = _new_accumulator(n_k)
    consts = np.array([cfg["spot"], cfg["mu"], cfg["sigma"], cfg["rate"], cfg["q"], cfg["tau_close"]],
                      dtype=np.float64)
    done = 0
    while done < paths:
        n = min(CPU_BATCH, paths - done)
        pnl = np.empty((n, n_k), dtype=np.float32)
        dayout = np.empty((n, n_k), dtype=np.int32)
        hit_p = np.empty((n, n_k), dtype=np.uint8)
        hit_s = np.empty((n, n_k), dtype=np.uint8)
        _cpu_paths(n, consts, arr["short"], arr["long"], arr["sig_s"], arr["sig_l"], arr["credit"],
                   arr["pnl_profit"], arr["pnl_stop"], arr["bp"], arr["bs"], cfg["days"],
                   cfg["has_stop"], pnl, dayout, hit_p, hit_s)
        _accumulate(acc, pnl, dayout, hit_p, hit_s)
        done += n
    return acc


def _run_gpu_chunk(paths, cfg, arr):
    n_k = arr["short"].shape[0]
    kernel = _get_gpu_kernel()
    dev = {name: cp.asarray(arr[name], dtype=cp.float32) for name in
           ("short", "long", "sig_s", "sig_l", "credit", "pnl_profit", "pnl_stop", "bp", "bs")}
    totals = cp.zeros(len(CHANNELS) * n_k + 1, dtype=cp.float64)
    sm_count = cp.cuda.Device().attributes["MultiProcessorCount"]
    grid = sm_count * 8
    done = 0
    batch_index = 0
    while done < paths:
        n = min(GPU_BATCH, paths - done)
        seed = (SEED * 1_000_003 + batch_index) & 0xFFFFFFFFFFFFFFFF
        kernel(
            (grid,), (BLOCK,),
            (np.uint64(n), np.uint64(seed), np.int32(n_k), np.int32(cfg["days"]),
             np.int32(cfg["has_stop"]),
             dev["short"], dev["long"], dev["sig_s"], dev["sig_l"], dev["credit"],
             dev["pnl_profit"], dev["pnl_stop"], dev["bp"], dev["bs"],
             np.float32(cfg["spot"]), np.float32(cfg["rate"]), np.float32(cfg["q"]),
             np.float32(cfg["sigma"]), np.float32(cfg["mu"]), np.float32(cfg["width"]),
             np.float32(cfg["tau_close"]), totals),
        )
        done += n
        batch_index += 1
    host = totals.get()
    if host[-1] > 0:
        raise RuntimeError("simulation produced an unclosed or invalid spread")
    return {name: host[i * n_k:(i + 1) * n_k] for i, name in enumerate(CHANNELS)}


def _run_gpu(paths, cfg, arr):
    n_s = arr["short"].shape[0]
    parts = []
    for start in range(0, n_s, MAX_KEYS):
        sl = slice(start, min(start + MAX_KEYS, n_s))
        chunk = {name: value[sl] for name, value in arr.items()}
        parts.append(_run_gpu_chunk(paths, cfg, chunk))
    return {name: np.concatenate([p[name] for p in parts]) for name in CHANNELS}


def _safe_round(value, digits):
    return round(float(value), digits)


def _build_response(req, rows, acc, cfg, device, elapsed):
    paths = req.paths
    spreads = []
    cap = req.width * math.exp(-req.risk_free_rate * req.close_dte / 365.0)
    for i, (short, long, credit) in enumerate(rows):
        mean = acc["sum"][i] / paths
        var = max(acc["sumsq"][i] / paths - mean * mean, 0.0)
        worst = cap - credit
        if req.stop_loss is not None:
            worst = min(worst, req.stop_loss * credit)
        profit_rate = acc["profit"][i] / paths
        stop_rate = acc["stop"][i] / paths
        spreads.append({
            "short_strike": short.strike,
            "long_strike": _safe_round(long.strike, 4),
            "short_delta": short.delta,
            "short_iv": short.iv,
            "long_iv": long.iv,
            "credit": _safe_round(credit, 4),
            "max_loss": _safe_round(worst, 4),
            "expected_pnl": _safe_round(mean, 4),
            "pnl_standard_error": _safe_round(math.sqrt(var / paths), 6),
            "win_rate": _safe_round(acc["wins"][i] / paths, 4),
            "pct_closed_at_profit_target": _safe_round(profit_rate, 4),
            "pct_closed_at_stop_loss": _safe_round(stop_rate, 4) if req.stop_loss is not None else None,
            "pct_closed_at_close_dte": _safe_round(1.0 - profit_rate - stop_rate, 4),
            "avg_days_held": _safe_round(acc["days"][i] / paths, 2),
        })
    return {
        "symbol": req.symbol,
        "option_type": req.option_type,
        "dte": req.dte,
        "close_dte": req.close_dte,
        "width": req.width,
        "device": device,
        "paths": paths,
        "elapsed_seconds": _safe_round(elapsed, 3),
        "assumptions": {
            "pricing": "Black-Scholes per leg at its own IV, sticky strike",
            "path_vol": req.realized_vol,
            "path_drift": cfg["mu"],
            "stop_loss": req.stop_loss,
            "tie_rule": "stop checked before profit target on the same day",
        },
        "spreads": spreads,
    }


def evaluate(req: EvaluateRequest) -> dict:
    rows = _build_spreads(req)
    if not rows:
        return _build_response(req, rows, _new_accumulator(0), {"mu": req.drift - req.dividend_yield},
                               "none", 0.0)
    with _lock:
        n_k = len(rows)
        days = req.dte - req.close_dte
        short_k = np.array([r[0].strike for r in rows], dtype=np.float64)
        long_k = np.array([r[1].strike for r in rows], dtype=np.float64)
        sig_s = np.array([r[0].iv for r in rows], dtype=np.float64)
        sig_l = np.array([r[1].iv for r in rows], dtype=np.float64)
        credit = np.array([r[2] for r in rows], dtype=np.float64)
        stop_mult = req.stop_loss if req.stop_loss is not None else 0.0
        bp, bs = _barriers(short_k, long_k, sig_s, sig_l, credit, req.risk_free_rate,
                           req.dividend_yield, req.dte, days, req.profit_target, stop_mult,
                           1 if req.stop_loss is not None else 0)
        cfg = {
            "spot": req.spot,
            "mu": req.drift - req.dividend_yield,
            "sigma": req.realized_vol,
            "rate": req.risk_free_rate,
            "q": req.dividend_yield,
            "width": req.width,
            "tau_close": req.close_dte / 365.0,
            "days": days,
            "has_stop": 1 if req.stop_loss is not None else 0,
        }
        arr = {
            "short": short_k, "long": long_k, "sig_s": sig_s, "sig_l": sig_l, "credit": credit,
            "pnl_profit": req.profit_target * credit,
            "pnl_stop": -stop_mult * credit,
            "bp": bp, "bs": bs,
        }
        device = "cuda" if GPU_AVAILABLE else "cpu"
        start = time.perf_counter()
        if GPU_AVAILABLE:
            acc = _run_gpu(req.paths, cfg, arr)
        else:
            acc = _run_cpu(req.paths, cfg, arr)
        elapsed = time.perf_counter() - start
    return _build_response(req, rows, acc, cfg, device, elapsed)
