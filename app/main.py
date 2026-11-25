from fastapi import FastAPI, Query

from app.simulation import MAX_PATHS, MIN_PATHS, PATHS, run_compute

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/compute")
def compute(paths: int = Query(PATHS, ge=MIN_PATHS, le=MAX_PATHS)) -> dict:
    return run_compute(paths)
