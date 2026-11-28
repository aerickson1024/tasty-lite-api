# tasty-lite-api

A small FastAPI service that evaluates put credit verticals with a GPU-accelerated Monte Carlo simulation (`/evaluate`), derives return metrics from the results (`/analyze`), and returns real quotes and IV rank from Tastytrade (`/stocks`).

Simulated results are not market predictions.

## What the API does

| Method | Path       | Purpose                                                              |
|--------|------------|----------------------------------------------------------------------|
| GET    | `/health`  | Liveness check. Returns `{"status": "ok"}`.                          |
| GET    | `/stocks`  | Real Tastytrade data for the configured symbols: `symbol`, `last`, `volume`, `iv_rank`. Needs credentials, see below. |

## Tastytrade stock data

`GET /stocks` calls the Tastytrade REST API (production, `https://api.tastyworks.com`) for each symbol in `TASTY_SYMBOLS`. It returns one item per symbol:

```json
{"items": [{"symbol": "SPY", "last": 512.34, "volume": 1234567.0, "iv_rank": 0.4567}]}
```

- `last` and `volume` come from `GET /market-data/by-type`.
- `iv_rank` is `implied-volatility-index-rank` from `GET /market-metrics`, as a fraction (0.4567 means 45.67%). Missing values are `null`.
- Access tokens last 15 minutes. They are refreshed from the refresh token and cached in memory.
- Responses: 503 if credentials are not set, 502 if Tastytrade fails or cannot be reached.

Create `.env` in this project root (it is git-ignored). Never commit it.

```
TASTY_CLIENT_SECRET=your-oauth-client-secret
TASTY_REFRESH_TOKEN=your-refresh-token
TASTY_SYMBOLS=SPY,QQQ,AAPL,TSLA,NVDA
```

`TASTY_SYMBOLS` is optional and defaults to the list shown. Get the client secret and refresh token from your Tastytrade developer application. The sandbox (`api.cert.tastyworks.com`) does not serve market data, so production credentials are required.

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
- `httpx` for calls to Tastytrade and `python-dotenv` to read `.env`.
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

## Evaluating a put credit vertical

`POST /evaluate` returns the expected value of each put credit vertical in a request. The caller supplies the strikes (mid, IV, delta) from tastytrade. The endpoint does not choose candidates and does not sort results.

```powershell
curl.exe -X POST http://127.0.0.1:8000/evaluate -H "Content-Type: application/json" -d @request.json
```

Request fields:

| Field              | Required | Meaning |
|--------------------|----------|---------|
| `symbol`           | yes      | Underlying symbol (label only). |
| `option_type`      | yes      | Must be `put_credit_vertical`. |
| `width`            | yes      | Distance between short and long strike. A long strike of `short - width` must be in `strikes`. |
| `dte`              | yes      | Entry days to expiration. One DTE per request. |
| `close_dte`        | yes      | Days to expiration at the time exit. Must be below `dte`. |
| `strikes[]`        | yes      | `strike`, `mid`, `iv` (decimal), `delta` (put delta, negative). Strikes must be unique. |
| `spot`             | yes      | Underlying price. |
| `risk_free_rate`   | yes      | Decimal, for example 0.05. |
| `dividend_yield`   | no       | Decimal, default 0. |
| `realized_vol`     | yes      | Decimal volatility used for simulated paths, for example 0.5. Values above 5.0 are rejected to catch percent-vs-decimal mistakes. |
| `drift`            | yes      | Decimal real-world drift of the underlying, for example 0.08. |
| `profit_target`    | yes      | Fraction of entry credit to buy back at, above 0 and up to 1. |
| `stop_loss`        | no       | Multiple of entry credit. The trade closes when the spread is worth `(1 + stop_loss) * credit`. Omit for no stop. |
| `paths`            | no       | Simulated paths, 1,000 to 2,000,000,000. Default 200,000,000. |

A request is rejected with 422 when `option_type` is not `put_credit_vertical`, `close_dte` is not below `dte`, strikes repeat, a mid is negative, `realized_vol` exceeds 5.0, `profit_target` is outside (0, 1], `stop_loss` is 0 or below, or `paths` is out of range. Strikes without a partner at `strike - width`, and spreads whose credit is not between 0 and `width`, are skipped silently.

Response: each entry in `spreads` has `short_strike`, `long_strike`, `short_delta`, `short_iv`, `long_iv`, `credit`, `max_loss`, `expected_pnl`, `pnl_standard_error`, `win_rate`, `pct_closed_at_profit_target`, `pct_closed_at_stop_loss` (null without a stop), `pct_closed_at_close_dte`, and `avg_days_held`. The response also reports `device`, `paths`, `elapsed_seconds` and `assumptions`.

Model assumptions:

- Each leg is priced with Black-Scholes at its own IV, with the spot and rate from the request and the dividend yield applied. The IV is held fixed through the trade (sticky strike).
- Entry credit is the mid of the short minus the mid of the long. Profit and stop barriers are modelled prices, so a mid that differs from the model price shifts the barriers.
- Paths are GBM with daily steps. Barriers are checked against the Brownian-bridge daily high (profit) and low (stop), with barrier levels taken at mid-day.
- If the stop and the profit target are both hit on the same day, the stop counts first.
- `max_loss` is the smaller of the discounted width minus credit and `stop_loss * credit`.
- Results are undiscounted. The drift is the real-world drift less the dividend yield.
- Spreads are split into chunks of 32 per GPU launch, so runtime grows with spread count: about 1.8 s for 11 spreads and about 8.3 s for 41 spreads at the default paths.

## Analyzing a spread

`POST /analyze` takes one spread object from an `/evaluate` response and returns return and annualized metrics. It does no simulation and needs no GPU.

Request fields:

| Field           | Required | Meaning |
|-----------------|----------|---------|
| `short_strike`  | yes      | Strike sold. Must be above `long_strike`. |
| `long_strike`   | yes      | Strike bought. |
| `credit`        | yes      | Entry credit per share. Must be below the width. |
| `expected_pnl`  | yes      | Expected P&L per share, from `/evaluate`. |
| `avg_days_held` | yes      | Average days held, from `/evaluate`. |
| `max_loss`      | no       | Simulated worst case per share. Only adds a per-contract figure. |
| `profit_target` | no       | Fraction of credit, used for profit target dollars. |

Other fields in the `/evaluate` response (deltas, IVs, win rate) are ignored, so a spread object can be sent as is.

Formulas, with capital at risk $C = \text{width} - \text{credit}$ and $n = 365 / \text{avg\_days\_held}$:

- Return on capital: $r = \text{expected\_pnl} / C$
- APR (simple): $r \cdot n$
- CAGR (compounded): $(1 + r)^n - 1$, or $-1$ when $r \le -1$. Null if the power overflows.
- Breakeven price: short strike minus credit. Max profit: credit. Profit target dollars: credit times `profit_target`.
- Risk/reward: $C / \text{credit}$. EV per day: expected P&L / avg days held.

Assumptions: capital is width minus credit, the year is 365 days, there are no fees or slippage, and capital is redeployed back to back. When a stop loss is used, switch the capital basis to the supplied `max_loss`.

## Project layout

```
app/
  main.py         FastAPI app and route definitions (/health, /stocks, /evaluate, /analyze)
  schemas.py      Request models and validation for /evaluate and /analyze
  evaluation.py   Put credit vertical engine for /evaluate (CPU and CUDA kernels)
  analysis.py     Return, APR and CAGR metrics for /analyze
requirements.txt  Python dependencies
.gitignore        Ignores virtual environments, caches and .env files
README.md         This file
```

## Notes

- A `CuPy` warning about `CUDA_PATH` may appear at startup. It is harmless when the GPU is used. Setting the `CUDA_PATH` environment variable removes it.
- The older `.venv` folder in this project uses Python 3.15 and does not contain the dependencies. Use `.venv312` instead.

## Changelog

- Initial version: `/health` route, and `/compute` GPU Monte Carlo endpoint with CPU fallback and Ryzen 7900 estimate.
- Changed `/compute` from POST to GET.
- Replaced the European call simulation with the managed put credit spread simulation (45 and 38 DTE, 21 spreads each, 50% profit target, 21 DTE time exit). Added Numba CPU kernel and CUDA kernel. `PATHS` is untuned and runs have not been measured yet.
- Scoped to 45 DTE only, short deltas 36 to 10 in 2-point steps (14 spreads). The CUDA kernel takes the entry count as a parameter.
- `max_loss` now reports the simulated worst case (discounted width minus credit) instead of `10 - credit`.
- Rewrote the CUDA kernel: random numbers (splitmix64) are generated in-kernel, results are reduced on the device, and setup arrays are cached on the GPU. 1e9 paths went from about 47 s to about 9 s.
- Fixed overlapping random streams between threads and batches (start state is now a hash of seed and thread id). After the fix the GPU matches the CPU within statistical error (z-scores -0.96 to -0.42 across the 14 spreads at 1e9 GPU vs 5e8 CPU paths). Uniforms now use 23 bits so a draw can never be exactly 1, and non-finite draws raise an error.
- Added the optional `paths` query parameter. Default `PATHS` is now 200,000,000 (about 1.8 s warm). The Ryzen 7900 estimate scales with the requested paths. The CPU fallback uses the same default and is not tuned.
- Added `POST /evaluate`: expected value and odds for caller-supplied put credit verticals, with optional stop loss (multiple of credit), dividend yield, and per-leg IV. Engine in `app/evaluation.py` with CPU and CUDA kernels. Validated against the Black-Scholes closed form (max |z| 0.58 at 2e6 paths) and GPU vs CPU with and without a stop (max |z| 1.34).
- Added `POST /analyze`: return on capital, APR, CAGR, breakeven, risk/reward, EV per day and per-contract dollars for one spread from `/evaluate`. Pure arithmetic, no simulation.
- Removed `GET /compute` and `app/simulation.py`. `PATHS`, `MIN_PATHS`, `MAX_PATHS` now live in `app/schemas.py`; GPU detection lives in `app/evaluation.py`.
