# tasty-lite-api

A small FastAPI service that runs a GPU-accelerated Monte Carlo simulation of a managed put credit spread strategy and reports how long it took, compared with an estimate of the same run on an AMD Ryzen 7900 CPU.

The market data is simulated, not real. The parameters are chosen to be in a realistic range, but the results are not market predictions.

## What the API does

| Method | Path       | Purpose                                                              |
|--------|------------|----------------------------------------------------------------------|
| GET    | `/health`  | Liveness check. Returns `{"status": "ok"}`.                          |
| GET    | `/compute` | Runs the Monte Carlo simulation and returns the results and timings. Optional query parameter `paths`. |

`/compute` simulates `PATHS` price paths of the underlying and evaluates a put credit spread on each path. Fixed inputs:

| Parameter           | Value |
|---------------------|-------|
| Spot price          | 100   |
| Risk-free rate      | 5%    |
| Volatility (flat)   | 50%   |
| Real-world drift    | 8%    |
| Spread width        | 10 points |
| Entry DTEs          | 45 |
| Close DTE           | 21 |
| Profit target       | 50% of entry credit |

For each entry DTE, 14 put credit spreads are built. The short put is the strike whose absolute delta is 0.36, 0.34, ..., 0.10 (Black-Scholes, 50% vol). The long put is 10 points below the short put. The credit is the spread's Black-Scholes value at entry, and max loss is 10 minus the credit.

### Management rules

- Profit target: a good-till-cancelled buy-to-close limit at 50% of the entry credit. It fills exactly at that price when the spread trades there.
- Time exit: if the target is not hit, close at Black-Scholes fair value at 21 DTE.
- No stop loss.
- The 45 DTE entry is held for up to 24 calendar days.

### Method

The price path is a daily geometric Brownian motion with drift 8% and volatility 50%. Intraday touches are checked with a Brownian bridge sample of each day's high, given the day's open and close. The profit target is compared with a precomputed barrier for each spread and day. The barrier is the log-spot level at which the spread is worth the target at mid-day. This avoids simulating intraday steps.

The CPU kernel uses Numba with all logical cores. The GPU kernel is a CUDA kernel compiled by CuPy. Both run the same logic. The GPU is used when one is available. Otherwise the CPU runs. Random numbers are not seeded on the CPU, so CPU results vary between calls. GPU results are reproducible for a given `SEED` and batch size.

### Assumptions and limitations

- Volatility is flat at 50% and does not vary with strike or time (no skew or term structure).
- Results are per share. No commissions, slippage, or assignment risk are modelled.
- P&L is not discounted. `expected_pnl` is the mean P&L at close.
- Limit fills occur at the target price. The fill time is rounded to whole days.

The request takes no body. The only query parameter is `paths` (integer, 1,000 to 2,000,000,000, default 200,000,000). Values outside that range, or non-integers, return 422. All other inputs are fixed.

On the RTX 3060 a warm call at the default takes about 1.8 s (about 9 s per billion paths). The standard error of `expected_pnl` at the default is about 1.7e-4 for the 36 delta spread and 7.7e-5 for the 10 delta spread, shrinking with the square root of `paths`. The first call after startup is slower (about 3.6 s measured) because of CUDA kernel compilation and the CPU calibration.

## Requirements

- Windows (the project has been run on Windows only).
- Python 3.12. CuPy wheels were not available for Python 3.15 when this was set up.
- For GPU acceleration: an NVIDIA GPU and a driver that supports CUDA 12.x. The CPU fallback works without one.

## Setup

From the project root:

PowerShell:

```powershell
python -m pip install uv
python -m uv venv --python 3.12 .venv312
python -m uv pip install --python .venv312\Scripts\python.exe -r requirements.txt
```

Bash (Git Bash, or the bash terminal in VS Code):

```bash
python -m pip install uv
python -m uv venv --python 3.12 .venv312
python -m uv pip install --python .venv312/Scripts/python.exe -r requirements.txt
```

`uv` downloads Python 3.12 automatically if it is not already installed.

`requirements.txt` installs:

- `fastapi` and `uvicorn[standard]` for the web server.
- `numpy` and `numba` for the CPU path.
- `numba` runs the CPU kernel in parallel across cores.
- `cupy-cuda12x[ctk]` for the GPU path. The `[ctk]` extra installs the CUDA headers that CuPy needs.

## Running the server

PowerShell:

```powershell
.venv312\Scripts\python.exe -m uvicorn app.main:app --reload
```

Bash:

```bash
./.venv312/Scripts/python.exe -m uvicorn app.main:app --reload
```

Or use the helper script from the project root (Git Bash):

```bash
bash run.sh
```

The script passes any extra arguments through to uvicorn, for example `bash run.sh --port 8001`.

The server listens on `http://127.0.0.1:8000`. Interactive API docs are at `http://127.0.0.1:8000/docs`.

## Using the API

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Run the simulation:

```powershell
curl.exe http://127.0.0.1:8000/compute
curl.exe "http://127.0.0.1:8000/compute?paths=1000000000"
```

Response shape (values are illustrative; run `/compute` to populate them):

```json
{
  "device": "cuda",
  "paths": 200000000,
  "dtype": "float32",
  "parameters": {
    "spot": 100.0,
    "rate": 0.05,
    "volatility": 0.5,
    "real_world_drift": 0.08,
    "width": 10.0,
    "profit_target": 0.5,
    "close_dte": 21
  },
  "elapsed_seconds": 0.0,
  "estimated_ryzen_7900_seconds": 0.0,
  "speedup_vs_ryzen_7900": 0.0,
  "entries": [
    {
      "dte": 45,
      "spreads": [
        {
          "delta": 36,
          "short_strike": 0.0,
          "long_strike": 0.0,
          "credit": 0.0,
          "max_loss": 0.0,
          "win_rate": 0.0,
          "pct_closed_at_profit_target": 0.0,
          "pct_closed_at_21_dte": 0.0,
          "avg_days_held": 0.0,
          "expected_pnl": 0.0,
          "pnl_standard_error": 0.0
        }
      ]
    }
  ]
}
```

`entries` has 1 item (45 DTE) with 14 spreads. The example above shows one spread only.

### Response fields

| Field                          | Meaning |
|--------------------------------|---------|
| `device`                       | `cuda` if the GPU ran the simulation, `cpu` if it fell back to Numba. |
| `paths`                        | Number of simulated price paths. |
| `dtype`                        | Floating-point precision of the simulated P&L. |
| `parameters`                   | The fixed inputs used for this run. |
| `elapsed_seconds`              | Wall-clock time of the timed simulation on the device used. |
| `estimated_ryzen_7900_seconds` | Estimated time for the same simulation on a Ryzen 7900. |
| `speedup_vs_ryzen_7900`        | `estimated_ryzen_7900_seconds / elapsed_seconds`. |
| `entries[].dte`                | Entry days to expiration (45). |
| `entries[].spreads[].delta`    | Short put absolute delta times 100 (36 means 0.36). |
| `short_strike`, `long_strike`  | Strikes of the short and long puts. `long_strike` is `short_strike - 10`. |
| `credit`                       | Credit received at entry, per share. |
| `max_loss`                     | `10 - credit`, per share. |
| `win_rate`                     | Fraction of paths with positive P&L at close. |
| `pct_closed_at_profit_target`  | Fraction of paths closed at the 50% profit target. |
| `pct_closed_at_21_dte`         | Fraction of paths closed at 21 DTE at fair value. |
| `avg_days_held`                | Average calendar days held. |
| `expected_pnl`                 | Mean P&L per share (undiscounted). |
| `pnl_standard_error`           | Monte Carlo standard error of `expected_pnl`. |

### How the Ryzen 7900 estimate works

The host machine is itself a Ryzen 9 7900. The first `/compute` call simulates `CALIBRATION_PATHS` paths on the CPU, and the time per path is cached. The estimate is that per-path time multiplied by the requested `paths`. It is an estimate, not a measured run on a 7900.

The first GPU call includes a warm-up step that is excluded from `elapsed_seconds`.

## Project layout

```
app/
  main.py         FastAPI app and route definitions (/health, /compute)
  simulation.py   Monte Carlo model, CPU/GPU selection, timing and 7900 estimate
requirements.txt  Python dependencies
.gitignore        Ignores virtual environments, caches and .env files
README.md         This file
```

## Configuration

Model inputs and simulation size are constants at the top of `app/simulation.py`:

| Constant                | Default     | Purpose |
|-------------------------|-------------|---------|
| `SPOT`, `RATE`, `VOLATILITY`, `REAL_WORLD_DRIFT`, `WIDTH` | see table above | Market and spread inputs. |
| `ENTRY_DTES`, `CLOSE_DTE` | (45,), 21 | Entry days to expiration and time exit. |
| `SHORT_DELTAS`          | 0.36 to 0.10 in steps of 0.02 (14 spreads) | Short put deltas simulated. |
| `PROFIT_TARGET`         | 0.5         | Buy-to-close target as a fraction of entry credit. |
| `PATHS`                 | 200,000,000 | Default number of simulated paths when `paths` is not given. About 1.8 s warm on the GPU. |
| `MIN_PATHS`, `MAX_PATHS` | 1,000, 2,000,000,000 | Accepted range of the `paths` query parameter. |
| `BATCH_SIZE`            | 1,000,000   | Paths per GPU/CPU batch. Lower it if GPU memory runs out. |
| `CALIBRATION_PATHS`     | 2,000,000   | Paths used to estimate the Ryzen 7900 runtime. |
| `SEED`                  | 42          | GPU random seed. CPU is not seeded. |
| `HOST_TO_RYZEN_7900_RATIO` | 1.0      | Scales the host CPU time to a Ryzen 7900. 1.0 because the host is a 7900. |

## Notes

- A `CuPy` warning about `CUDA_PATH` may appear at startup. It is harmless when the GPU is used. Setting the `CUDA_PATH` environment variable removes it.
- The older `.venv` folder in this project uses Python 3.15 and does not contain the dependencies. Use `.venv312` instead.

## Changelog

- Initial version: `/health` route, and `/compute` GPU Monte Carlo endpoint with CPU fallback and Ryzen 7900 estimate.
- Changed `/compute` from POST to GET.
- Replaced the European call simulation with the managed put credit spread simulation (45 and 38 DTE, 21 spreads each, 50% profit target, 21 DTE time exit). Added Numba CPU kernel and CUDA kernel. `PATHS` is untuned and runs have not been measured yet.
- Scoped to 45 DTE only, short deltas 36 to 10 in 2-point steps (14 spreads). The CUDA kernel takes the entry count as a parameter.
- Rewrote the CUDA kernel: random numbers (splitmix64) are generated in-kernel, results are reduced on the device, and setup arrays are cached on the GPU. 1e9 paths went from about 47 s to about 9 s.
- Fixed overlapping random streams between threads and batches (start state is now a hash of seed and thread id). After the fix the GPU matches the CPU within statistical error (z-scores -0.96 to -0.42 across the 14 spreads at 1e9 GPU vs 5e8 CPU paths). Uniforms now use 23 bits so a draw can never be exactly 1, and non-finite draws raise an error.
- Added the optional `paths` query parameter. Default `PATHS` is now 200,000,000 (about 1.8 s warm). The Ryzen 7900 estimate scales with the requested paths. The CPU fallback uses the same default and is not tuned.
