from fastapi import FastAPI

from app.simulation import run_compute

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/compute")
def compute() -> dict:
    return run_compute()
