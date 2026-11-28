import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.analysis import analyze
from app.evaluation import evaluate
from app.schemas import AnalyzeRequest, EvaluateRequest
from app.tastytrade import TastytradeError, TastytradeNotConfigured, fetch_stocks

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/health")
async def health_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            await ws.send_json({"status": "ok"})
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass


@app.get("/stocks")
async def stocks() -> dict:
    try:
        return {"items": await fetch_stocks()}
    except TastytradeNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TastytradeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/evaluate")
def evaluate_spreads(req: EvaluateRequest) -> dict:
    return evaluate(req)


@app.post("/analyze")
def analyze_spread(req: AnalyzeRequest) -> dict:
    return analyze(req)
