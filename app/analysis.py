from app.schemas import AnalyzeRequest

DAYS_PER_YEAR = 365.0


def analyze(req: AnalyzeRequest) -> dict:
    width = req.short_strike - req.long_strike
    capital = width - req.credit
    ror = req.expected_pnl / capital
    trades_per_year = DAYS_PER_YEAR / req.avg_days_held

    apr = ror * trades_per_year
    if ror <= -1.0:
        cagr = -1.0
    else:
        try:
            cagr = (1.0 + ror) ** trades_per_year - 1.0
        except OverflowError:
            cagr = None

    profit_target_dollars = None
    if req.profit_target is not None:
        profit_target_dollars = req.credit * req.profit_target

    per_share = {
        "width": width,
        "capital_at_risk": capital,
        "return_on_capital": ror,
        "breakeven_price": req.short_strike - req.credit,
        "max_profit": req.credit,
        "profit_target_dollars": profit_target_dollars,
        "risk_reward": capital / req.credit,
        "ev_per_day": req.expected_pnl / req.avg_days_held,
    }

    per_contract = {
        "capital_at_risk": capital * 100,
        "max_profit": req.credit * 100,
        "expected_pnl": req.expected_pnl * 100,
        "profit_target_dollars": None if profit_target_dollars is None else profit_target_dollars * 100,
        "max_loss": None if req.max_loss is None else req.max_loss * 100,
    }

    return {
        "inputs": {
            "short_strike": req.short_strike,
            "long_strike": req.long_strike,
            "credit": req.credit,
            "expected_pnl": req.expected_pnl,
            "avg_days_held": req.avg_days_held,
        },
        "per_share": per_share,
        "per_contract": per_contract,
        "annualized": {
            "trades_per_year": trades_per_year,
            "apr": apr,
            "cagr": cagr,
        },
        "assumptions": {
            "capital_basis": "width minus credit",
            "days_per_year": DAYS_PER_YEAR,
            "fees_and_slippage": "none",
            "redeployment": "back to back",
        },
    }
