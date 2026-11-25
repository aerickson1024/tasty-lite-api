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

PATHS = 200_000_000  # about 1.8 s warm on an RTX 3060
MIN_PATHS = 1_000
MAX_PATHS = 2_000_000_000
BATCH_SIZE = 1_000_000
BLOCK = 256  # must match BLOCK in _CUDA_SOURCE
CALIBRATION_PATHS = 2_000_000
SEED = 42
HOST_TO_RYZEN_7900_RATIO = 1.0

SQRT2 = math.sqrt(2.0)
_lock = threading.Lock()
_setup = None
_device = None
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

// seed ^ (tid * golden) left streams of nearby threads/batches overlapping; hash both instead.
__device__ __forceinline__ unsigned long long init_state(unsigned long long seed, unsigned long long tid) {
    return mix64(mix64(seed ^ 0xD1B54A32D192ED03ull) + tid * 0x9E3779B97F4A7C15ull);
}

__device__ __forceinline__ float uniform01(unsigned long long &s) {
    return ((float)(splitmix64(s) >> 41) + 0.5f) * (1.0f / 8388608.0f);
}

#define MAX_KEYS 32
#define BLOCK 256

extern "C" __global__ void managed_paths(
        const unsigned long long n_paths, const unsigned long long seed,
        const float* credit, const float* shortk, const float* barrier, const int* close_t,
        const float spot0, const float rate, const float sigma, const float mu,
        const float width, const int days, const int n_s, const int n_e,
        const float tau_close, double* totals) {
    const int n_k = n_e * n_s;
    const float unset = __int_as_float(0x7fc00000);
    const float var_day = sigma * sigma / 365.0f;
    const float drift_day = (mu - 0.5f * sigma * sigma) / 365.0f;
    const float vol_day = sigma / sqrtf(365.0f);

    double acc[5][MAX_KEYS];
    for (int c = 0; c < 5; c++)
        for (int k = 0; k < MAX_KEYS; k++) acc[c][k] = 0.0;
    double unclosed = 0.0;

    const unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long state = init_state(seed, tid);

    for (unsigned long long p = tid; p < n_paths; p += stride) {
        float pnl[MAX_KEYS];
        int day_k[MAX_KEYS];
        unsigned char hit_k[MAX_KEYS];
        for (int k = 0; k < n_k; k++) {
            pnl[k] = unset;
            day_k[k] = 0;
            hit_k[k] = 0;
        }

        float xa = logf(spot0);
        for (int t = 0; t < days; t++) {
            float zt = normcdfinvf(uniform01(state));
            if (!isfinite(zt)) unclosed += 1.0;
            float ut = uniform01(state);
            float xb = xa + drift_day + vol_day * zt;
            float d = xb - xa;
            float m = xa + 0.5f * (d + sqrtf(d * d - 2.0f * var_day * logf(ut)));
            for (int e = 0; e < n_e; e++) {
                if (t >= close_t[e]) continue;
                for (int s = 0; s < n_s; s++) {
                    int k = e * n_s + s;
                    if (isnan(pnl[k]) && m >= barrier[k * days + t]) {
                        pnl[k] = 0.5f * credit[k];
                        day_k[k] = t + 1;
                        hit_k[k] = 1;
                    }
                }
                if (t + 1 == close_t[e]) {
                    float price = expf(xb);
                    for (int s = 0; s < n_s; s++) {
                        int k = e * n_s + s;
                        if (isnan(pnl[k])) {
                            float ks = shortk[k];
                            float v = put_value(price, ks, tau_close, rate, sigma)
                                    - put_value(price, ks - width, tau_close, rate, sigma);
                            pnl[k] = credit[k] - v;
                            day_k[k] = close_t[e];
                        }
                    }
                }
            }
            xa = xb;
        }

        for (int k = 0; k < n_k; k++) {
            if (isnan(pnl[k]) || !isfinite(pnl[k])) { unclosed += 1.0; continue; }
            double v = pnl[k];
            acc[0][k] += v;
            acc[1][k] += v * v;
            acc[2][k] += (v > 0.0) ? 1.0 : 0.0;
            acc[3][k] += hit_k[k];
            acc[4][k] += day_k[k];
        }
    }

    __shared__ double red[BLOCK];
    const int n_vals = 5 * n_k + 1;
    for (int j = 0; j < n_vals; j++) {
        double v = (j < 5 * n_k) ? acc[j / n_k][j % n_k] : unclosed;
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


def _get_device_setup(setup):
    global _device
    if _device is None:
        _device = _device_setup(setup)
    return _device


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


def _run_gpu(paths, mu, dev):
    credit, short, barrier, close_t = dev
    kernel = _get_gpu_kernel()
    n_vals = 5 * N_KEYS + 1
    totals = cp.zeros(n_vals, dtype=cp.float64)
    sm_count = cp.cuda.Device().attributes["MultiProcessorCount"]
    grid = sm_count * 8
    done = 0
    batch_index = 0
    while done < paths:
        n = min(BATCH_SIZE, paths - done)
        seed = (SEED * 1_000_003 + batch_index) & 0xFFFFFFFFFFFFFFFF
        kernel(
            (grid,), (BLOCK,),
            (np.uint64(n), np.uint64(seed), credit, short, barrier, close_t,
             np.float32(SPOT), np.float32(RATE), np.float32(VOLATILITY), np.float32(mu),
             np.float32(WIDTH), np.int32(DAYS), np.int32(N_SPREADS), np.int32(len(ENTRY_DTES)),
             np.float32(CLOSE_DTE / 365.0), totals),
        )
        done += n
        batch_index += 1
    host = totals.get()
    if host[-1] > 0:
        raise RuntimeError("simulation produced an invalid path or unclosed spread")
    k = N_KEYS
    return {"sum": host[0:k], "sumsq": host[k:2 * k], "wins": host[2 * k:3 * k],
            "hits": host[3 * k:4 * k], "days": host[4 * k:5 * k]}


def _new_accumulator():
    return {key: np.zeros(N_KEYS, dtype=np.float64)
            for key in ("sum", "sumsq", "wins", "hits", "days")}


def _accumulate(acc, pnl, days, hit):
    p = pnl.astype(np.float64)
    if bool(np.isnan(p).any()):
        raise RuntimeError("simulation produced an unclosed spread")
    acc["sum"] += p.sum(axis=0)
    acc["sumsq"] += (p * p).sum(axis=0)
    acc["wins"] += (p > 0).sum(axis=0)
    acc["hits"] += hit.sum(axis=0, dtype=np.float64)
    acc["days"] += days.sum(axis=0, dtype=np.float64)


def _run_cpu(paths, mu, setup):
    acc = _new_accumulator()
    done = 0
    while done < paths:
        n = min(BATCH_SIZE, paths - done)
        pnl, days, hit = _batch_cpu(n, mu, setup)
        _accumulate(acc, pnl, days, hit)
        done += n
    return acc


def _cpu_seconds_per_path(setup):
    global _calibration_seconds
    if _calibration_seconds is None:
        _run_cpu(10_000, REAL_WORLD_DRIFT, setup)  # first call compiles the Numba kernel
        start = time.perf_counter()
        _run_cpu(CALIBRATION_PATHS, REAL_WORLD_DRIFT, setup)
        elapsed = time.perf_counter() - start
        _calibration_seconds = elapsed / CALIBRATION_PATHS * HOST_TO_RYZEN_7900_RATIO
    return _calibration_seconds


def _build_response(device, paths, elapsed, estimated, acc, setup):
    credit, short, _, _ = setup
    mean = acc["sum"] / paths
    variance = np.maximum(acc["sumsq"] / paths - mean * mean, 0.0)
    standard_error = np.sqrt(variance / paths)
    win_rate = acc["wins"] / paths
    hit_rate = acc["hits"] / paths
    avg_days = acc["days"] / paths

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
        "paths": paths,
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


def run_compute(paths: int = PATHS) -> dict:
    with _lock:
        setup = _get_setup()
        estimated = _cpu_seconds_per_path(setup) * paths
        if GPU_AVAILABLE:
            device = "cuda"
            dev = _get_device_setup(setup)
            _run_gpu(1_000, REAL_WORLD_DRIFT, dev)  # first call compiles the CUDA kernel
            start = time.perf_counter()
            acc = _run_gpu(paths, REAL_WORLD_DRIFT, dev)
        else:
            device = "cpu"
            start = time.perf_counter()
            acc = _run_cpu(paths, REAL_WORLD_DRIFT, setup)
        elapsed = time.perf_counter() - start
    return _build_response(device, paths, elapsed, estimated, acc, setup)
