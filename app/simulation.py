import math
import threading
import time

import numpy as np
from numba import njit, prange

try:
    import cupy as cp

    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    GPU_AVAILABLE = False

SPOT = 100.0
RATE = 0.05
VOLATILITY = 0.50
REAL_WORLD_DRIFT = 0.08
WIDTH = 10.0
ENTRY_DTES = (45,)
CLOSE_DTE = 21
DAYS = max(ENTRY_DTES) - CLOSE_DTE  # calendar days simulated per path
SHORT_DELTAS = tuple(round(0.36 - 0.02 * i, 2) for i in range(14))
N_SPREADS = len(SHORT_DELTAS)
N_KEYS = len(ENTRY_DTES) * N_SPREADS
PROFIT_TARGET = 0.5

PATHS = 1_000_000_000  # untuned: retune so the Ryzen 7900 estimate is about 120-130 s
BATCH_SIZE = 1_000_000
CALIBRATION_PATHS = 2_000_000
SEED = 42
HOST_TO_RYZEN_7900_RATIO = 1.0

SQRT2 = math.sqrt(2.0)
_lock = threading.Lock()
_setup = None
_calibration_seconds = None
_gpu_kernel = None


_CUDA_SOURCE = r"""
__device__ float put_value(float spot, float strike, float tau, float rate, float sigma) {
    float vol = sigma * sqrtf(tau);
    float d1 = (logf(spot / strike) + (rate + 0.5f * sigma * sigma) * tau) / vol;
    float d2 = d1 - vol;
    return strike * expf(-rate * tau) * 0.5f * erfcf(d2 * 0.70710678f)
         - spot * 0.5f * erfcf(d1 * 0.70710678f);
}

extern "C" __global__ void managed_paths(
        const int n, const float* z, const float* u, const float* credit,
        const float* shortk, const float* barrier, const int* close_t,
        const float spot0, const float rate, const float sigma, const float mu,
        const float width, const int days, const int n_s, const int n_e,
        float* pnl, int* day_out, unsigned char* hit) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const int n_k = n_e * n_s;
    const float unset = __int_as_float(0x7fc00000);
    const float var_day = sigma * sigma / 365.0f;
    const float drift_day = (mu - 0.5f * sigma * sigma) / 365.0f;
    const float vol_day = sigma * 0.05234834f;
    const float tau_close = 21.0f / 365.0f;

    for (int k = 0; k < n_k; k++) {
        pnl[i * n_k + k] = unset;
        day_out[i * n_k + k] = 0;
        hit[i * n_k + k] = 0;
    }

    float xa = logf(spot0);
    for (int t = 0; t < days; t++) {
        float zt = z[i * days + t];
        float ut = 1.0f - u[i * days + t];
        float xb = xa + drift_day + vol_day * zt;
        float d = xb - xa;
        float m = xa + 0.5f * (d + sqrtf(d * d - 2.0f * var_day * logf(ut)));
        for (int e = 0; e < n_e; e++) {
            if (t >= close_t[e]) continue;
            for (int s = 0; s < n_s; s++) {
                int k = e * n_s + s;
                int idx = i * n_k + k;
                if (isnan(pnl[idx]) && m >= barrier[k * days + t]) {
                    pnl[idx] = 0.5f * credit[k];
                    day_out[idx] = t + 1;
                    hit[idx] = 1;
                }
            }
            if (t + 1 == close_t[e]) {
                float price = expf(xb);
                for (int s = 0; s < n_s; s++) {
                    int k = e * n_s + s;
                    int idx = i * n_k + k;
                    if (isnan(pnl[idx])) {
                        float ks = shortk[k];
                        float v = put_value(price, ks, tau_close, rate, sigma)
                                - put_value(price, ks - width, tau_close, rate, sigma);
                        pnl[idx] = credit[k] - v;
                        day_out[idx] = close_t[e];
                    }
                }
            }
        }
        xa = xb;
    }
}
"""


@njit(cache=True)
def _put_value(spot, strike, tau, rate, sigma):
    if tau <= 0.0:
        return max(strike - spot, 0.0)
    vol = sigma * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / vol
    d2 = d1 - vol
    return (strike * math.exp(-rate * tau) * 0.5 * math.erfc(d2 / SQRT2)
            - spot * 0.5 * math.erfc(d1 / SQRT2))


@njit(parallel=True, cache=True)
def _cpu_paths(n, consts, credit, short, barrier, close_t, pnl, days, hit):
    spot0 = consts[0]
    rate = consts[1]
    sigma = consts[2]
    mu = consts[3]
    width = consts[4]
    var_day = sigma * sigma / 365.0
    drift_day = (mu - 0.5 * sigma * sigma) / 365.0
    vol_day = sigma / math.sqrt(365.0)
    tau_close = CLOSE_DTE / 365.0
    n_s = credit.shape[1]
    n_k = pnl.shape[1]
    n_days = barrier.shape[2]
    for i in prange(n):
        for kk in range(n_k):
            pnl[i, kk] = np.nan
            days[i, kk] = 0
            hit[i, kk] = 0
        xa = math.log(spot0)
        for t in range(n_days):
            z = np.random.standard_normal()
            u = 1.0 - np.random.random()
            xb = xa + drift_day + vol_day * z
            d = xb - xa
            m = xa + 0.5 * (d + math.sqrt(d * d - 2.0 * var_day * math.log(u)))
            for e in range(credit.shape[0]):
                if t >= close_t[e]:
                    continue
                for s in range(n_s):
                    k = e * n_s + s
                    if np.isnan(pnl[i, k]) and m >= barrier[e, s, t]:
                        pnl[i, k] = 0.5 * credit[e, s]
                        days[i, k] = t + 1
                        hit[i, k] = 1
                if t + 1 == close_t[e]:
                    price = math.exp(xb)
                    for s in range(n_s):
                        k = e * n_s + s
                        if np.isnan(pnl[i, k]):
                            ks = short[e, s]
                            value = (_put_value(price, ks, tau_close, rate, sigma)
                                     - _put_value(price, ks - width, tau_close, rate, sigma))
                            pnl[i, k] = credit[e, s] - value
                            days[i, k] = close_t[e]
            xa = xb


def _put_delta_abs(strike, tau):
    vol = VOLATILITY * math.sqrt(tau)
    d1 = (math.log(SPOT / strike) + (RATE + 0.5 * VOLATILITY**2) * tau) / vol
    return 0.5 * math.erfc(d1 / SQRT2)


def _strike_for_delta(delta, tau):
    lo, hi = 20.0, 200.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _put_delta_abs(mid, tau) < delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _build_setup():
    credit = np.zeros((len(ENTRY_DTES), N_SPREADS))
    short = np.zeros_like(credit)
    close_t = np.array([dte - CLOSE_DTE for dte in ENTRY_DTES], dtype=np.int64)
    barrier = np.zeros((len(ENTRY_DTES), N_SPREADS, DAYS))
    for e, dte in enumerate(ENTRY_DTES):
        tau = dte / 365.0
        for s, delta in enumerate(SHORT_DELTAS):
            ks = _strike_for_delta(delta, tau)
            kl = ks - WIDTH
            short[e, s] = ks
            credit[e, s] = (_put_value(SPOT, ks, tau, RATE, VOLATILITY)
                            - _put_value(SPOT, kl, tau, RATE, VOLATILITY))
            target = PROFIT_TARGET * credit[e, s]
            for t in range(close_t[e]):
                # log-spot at which the spread is worth the profit target, mid-day
                tau_mid = (dte - (t + 0.5)) / 365.0
                lo, hi = math.log(1.0), math.log(1000.0)
                for _ in range(100):
                    mid = 0.5 * (lo + hi)
                    spot = math.exp(mid)
                    value = (_put_value(spot, ks, tau_mid, RATE, VOLATILITY)
                             - _put_value(spot, kl, tau_mid, RATE, VOLATILITY))
                    if value > target:
                        lo = mid
                    else:
                        hi = mid
                barrier[e, s, t] = 0.5 * (lo + hi)
    return credit, short, barrier, close_t


def _get_setup():
    global _setup
    if _setup is None:
        _setup = _build_setup()
    return _setup


def _device_setup(setup):
    credit, short, barrier, close_t = setup
    return (cp.asarray(credit, dtype=cp.float32).ravel(),
            cp.asarray(short, dtype=cp.float32).ravel(),
            cp.asarray(barrier, dtype=cp.float32).ravel(),
            cp.asarray(close_t, dtype=cp.int32))


def _get_gpu_kernel():
    global _gpu_kernel
    if _gpu_kernel is None:
        _gpu_kernel = cp.RawKernel(_CUDA_SOURCE, "managed_paths")
    return _gpu_kernel


def _batch_cpu(n, mu, setup):
    credit, short, barrier, close_t = setup
    consts = np.array([SPOT, RATE, VOLATILITY, mu, WIDTH], dtype=np.float64)
    pnl = np.empty((n, N_KEYS), dtype=np.float32)
    days = np.empty((n, N_KEYS), dtype=np.int32)
    hit = np.empty((n, N_KEYS), dtype=np.uint8)
    _cpu_paths(n, consts, credit, short, barrier, close_t, pnl, days, hit)
    return pnl, days, hit


def _batch_gpu(n, seed, mu, dev):
    credit, short, barrier, close_t = dev
    rng = cp.random.default_rng(seed)
    z = rng.standard_normal((n, DAYS), dtype=cp.float32)
    u = rng.random((n, DAYS), dtype=cp.float32)
    pnl = cp.empty((n, N_KEYS), dtype=cp.float32)
    days = cp.empty((n, N_KEYS), dtype=cp.int32)
    hit = cp.empty((n, N_KEYS), dtype=cp.uint8)
    block = 256
    grid = (n + block - 1) // block
    _get_gpu_kernel()(
        (grid,), (block,),
        (np.int32(n), z, u, credit, short, barrier, close_t,
         np.float32(SPOT), np.float32(RATE), np.float32(VOLATILITY), np.float32(mu),
         np.float32(WIDTH), np.int32(DAYS), np.int32(N_SPREADS), np.int32(len(ENTRY_DTES)),
         pnl, days, hit),
    )
    return pnl, days, hit


def _new_accumulator(xp):
    return {key: xp.zeros(N_KEYS, dtype=xp.float64)
            for key in ("sum", "sumsq", "wins", "hits", "days")}


def _accumulate(acc, pnl, days, hit, xp):
    p = pnl.astype(xp.float64)
    if bool(xp.isnan(p).any()):
        raise RuntimeError("simulation produced an unclosed spread")
    acc["sum"] += p.sum(axis=0)
    acc["sumsq"] += (p * p).sum(axis=0)
    acc["wins"] += (p > 0).sum(axis=0)
    acc["hits"] += hit.sum(axis=0, dtype=xp.float64)
    acc["days"] += days.sum(axis=0, dtype=xp.float64)


def _run(xp, paths, mu, setup, dev):
    acc = _new_accumulator(xp)
    done = 0
    batch_index = 0
    while done < paths:
        n = min(BATCH_SIZE, paths - done)
        if xp is cp:
            pnl, days, hit = _batch_gpu(n, SEED + batch_index, mu, dev)
        else:
            pnl, days, hit = _batch_cpu(n, mu, setup)
        _accumulate(acc, pnl, days, hit, xp)
        done += n
        batch_index += 1
    if xp is cp:
        return {key: value.get() for key, value in acc.items()}
    return acc


def _cpu_calibration_seconds(setup):
    global _calibration_seconds
    if _calibration_seconds is None:
        _run(np, 10_000, REAL_WORLD_DRIFT, setup, None)  # first call compiles the Numba kernel
        start = time.perf_counter()
        _run(np, CALIBRATION_PATHS, REAL_WORLD_DRIFT, setup, None)
        elapsed = time.perf_counter() - start
        _calibration_seconds = elapsed * (PATHS / CALIBRATION_PATHS) * HOST_TO_RYZEN_7900_RATIO
    return _calibration_seconds


def _build_response(device, elapsed, estimated, acc, setup):
    credit, short, _, _ = setup
    mean = acc["sum"] / PATHS
    variance = np.maximum(acc["sumsq"] / PATHS - mean * mean, 0.0)
    standard_error = np.sqrt(variance / PATHS)
    win_rate = acc["wins"] / PATHS
    hit_rate = acc["hits"] / PATHS
    avg_days = acc["days"] / PATHS

    entries = []
    for e, dte in enumerate(ENTRY_DTES):
        spreads = []
        for s, delta in enumerate(SHORT_DELTAS):
            k = e * N_SPREADS + s
            c = float(credit[e, s])
            ks = float(short[e, s])
            spreads.append({
                "delta": int(round(delta * 100)),
                "short_strike": round(ks, 2),
                "long_strike": round(ks - WIDTH, 2),
                "credit": round(c, 4),
                "max_loss": round(WIDTH - c, 4),
                "win_rate": round(float(win_rate[k]), 4),
                "pct_closed_at_profit_target": round(float(hit_rate[k]), 4),
                "pct_closed_at_21_dte": round(1.0 - float(hit_rate[k]), 4),
                "avg_days_held": round(float(avg_days[k]), 2),
                "expected_pnl": round(float(mean[k]), 4),
                "pnl_standard_error": round(float(standard_error[k]), 6),
            })
        entries.append({"dte": dte, "spreads": spreads})

    return {
        "device": device,
        "paths": PATHS,
        "dtype": "float32",
        "parameters": {
            "spot": SPOT,
            "rate": RATE,
            "volatility": VOLATILITY,
            "real_world_drift": REAL_WORLD_DRIFT,
            "width": WIDTH,
            "profit_target": PROFIT_TARGET,
            "close_dte": CLOSE_DTE,
        },
        "elapsed_seconds": round(elapsed, 4),
        "estimated_ryzen_7900_seconds": round(estimated, 4),
        "speedup_vs_ryzen_7900": round(estimated / elapsed, 2),
        "entries": entries,
    }


def run_compute() -> dict:
    with _lock:
        setup = _get_setup()
        estimated = _cpu_calibration_seconds(setup)
        if GPU_AVAILABLE:
            device, xp = "cuda", cp
            dev = _device_setup(setup)
            _run(cp, 1_000, REAL_WORLD_DRIFT, setup, dev)  # first call compiles the CUDA kernel
        else:
            device, xp, dev = "cpu", np, None
        start = time.perf_counter()
        acc = _run(xp, PATHS, REAL_WORLD_DRIFT, setup, dev)
        elapsed = time.perf_counter() - start
    return _build_response(device, elapsed, estimated, acc, setup)
