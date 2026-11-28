from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

PATHS = 200_000_000
MIN_PATHS = 1_000
MAX_PATHS = 2_000_000_000


class StrikeQuote(BaseModel):
    strike: float = Field(gt=0)
    mid: float = Field(ge=0)
    iv: float = Field(gt=0)
    delta: float = Field(ge=-1, le=1)


class EvaluateRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    option_type: Literal["put_credit_vertical"]
    width: float = Field(gt=0)
    dte: int = Field(ge=1)
    close_dte: int = Field(ge=0)
    strikes: list[StrikeQuote] = Field(min_length=1)
    spot: float = Field(gt=0)
    risk_free_rate: float = Field(ge=-0.1, le=1)
    dividend_yield: float = Field(default=0.0, ge=0, le=1)
    realized_vol: float = Field(gt=0, le=5.0)
    drift: float = Field(ge=-1, le=5)
    profit_target: float = Field(gt=0, le=1)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    paths: int = Field(default=PATHS, ge=MIN_PATHS, le=MAX_PATHS)

    @model_validator(mode="after")
    def check_consistency(self):
        if self.close_dte >= self.dte:
            raise ValueError("close_dte must be below dte")
        strikes = [q.strike for q in self.strikes]
        if len(set(strikes)) != len(strikes):
            raise ValueError("strikes must be unique")
        return self


class AnalyzeRequest(BaseModel):
    short_strike: float = Field(gt=0)
    long_strike: float = Field(gt=0)
    credit: float = Field(gt=0)
    expected_pnl: float
    avg_days_held: float = Field(gt=0)
    max_loss: Optional[float] = Field(default=None, gt=0)
    profit_target: Optional[float] = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def check_spread(self):
        if self.short_strike <= self.long_strike:
            raise ValueError("short_strike must be above long_strike")
        if self.credit >= self.short_strike - self.long_strike:
            raise ValueError("credit must be below the width")
        return self
