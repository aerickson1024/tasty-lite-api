# tasty-lite-api

A small FastAPI service that runs a GPU-accelerated Monte Carlo simulation of a managed put credit spread strategy and reports how long it took, compared with an estimate of the same run on an AMD Ryzen 7900 CPU.

The market data is simulated, not real. The parameters are chosen to be in a realistic range, but the results are not market predictions.

## What the API does

| Method | Path       | Purpose                                                              |
|--------|------------|----------------------------------------------------------------------|
| GET    | `/health`  | Liveness check. Returns `{"status": "ok"}`.                          |
| GET    | `/compute` | Runs the Monte Carlo simulation and returns the results and timings. |

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

The request takes no body and no query parameters. The same inputs are used on every call.

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
```

Response shape (values are not yet measured for the spread simulation; run `/compute` to populate them):

```json
{
  "device": "cuda",
  "paths": 1000000000,
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

The host machine is itself a Ryzen 9 7900. The first `/compute` call simulates `CALIBRATION_PATHS` paths on the CPU, then scales the time linearly to `PATHS`. The result is an estimate, not a measured run on a 7900. The calibration is cached after the first call.

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
| `PATHS`                 | 1,000,000,000 | Number of simulated paths. Untuned. Retune so the Ryzen 7900 estimate is about 120 to 130 seconds. |
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
