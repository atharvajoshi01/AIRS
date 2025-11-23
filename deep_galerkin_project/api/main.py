"""
FastAPI application for option pricing inference.

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dgmlib.inference.service import PricingService
from dgmlib.utils.numerics import monte_carlo_european

# Create FastAPI app
app = FastAPI(
    title="Deep Galerkin Option Pricing API",
    description="REST API for option pricing using Deep Galerkin Methods",
    version="0.1.0",
)

# Global pricing service (load model on startup)
pricing_service = PricingService()


class PriceRequest(BaseModel):
    """Request model for option pricing."""

    S: float = Field(..., description="Current stock price", gt=0)
    K: float = Field(..., description="Strike price", gt=0)
    r: float = Field(..., description="Risk-free rate", ge=0, le=1)
    sigma: float = Field(..., description="Volatility", gt=0, le=2)
    T: float = Field(..., description="Time to maturity", gt=0, le=10)
    option_type: Literal["call", "put"] = Field("call", description="Option type")
    method: Literal["dgm", "bs", "mc", "fd"] = Field(
        "bs", description="Pricing method"
    )
    t: float = Field(0.0, description="Current time", ge=0)

    class Config:
        schema_extra = {
            "example": {
                "S": 100.0,
                "K": 100.0,
                "r": 0.05,
                "sigma": 0.2,
                "T": 1.0,
                "option_type": "call",
                "method": "bs",
                "t": 0.0,
            }
        }


class PriceResponse(BaseModel):
    """Response model for option pricing."""

    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    method: str
    parameters: dict


@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "Deep Galerkin Option Pricing API",
        "version": "0.1.0",
        "endpoints": {
            "/price": "POST - Price an option",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/price", response_model=PriceResponse)
def price_option(request: PriceRequest):
    """
    Price an option using specified method.

    Methods:
    - bs: Analytical Black-Scholes formula
    - mc: Monte Carlo simulation
    - dgm: Deep Galerkin Method (requires loaded model)
    - fd: Finite Difference (not yet implemented via API)
    """
    try:
        params = request.dict()
        method = params.pop("method")

        if method == "bs":
            # Analytical Black-Scholes
            result = pricing_service.price_analytical(
                S=request.S,
                K=request.K,
                r=request.r,
                sigma=request.sigma,
                T=request.T - request.t,
                option_type=request.option_type,
            )

        elif method == "mc":
            # Monte Carlo
            price, std_err = monte_carlo_european(
                S0=request.S,
                K=request.K,
                r=request.r,
                sigma=request.sigma,
                T=request.T - request.t,
                option_type=request.option_type,
                n_paths=100000,
            )
            result = {
                "price": price,
                "std_error": std_err,
            }

        elif method == "dgm":
            # Deep Galerkin Method
            if pricing_service.model is None:
                raise HTTPException(
                    status_code=400,
                    detail="No DGM model loaded. Use /load_model endpoint first.",
                )

            result = pricing_service.price(S=request.S, t=request.t)

        elif method == "fd":
            raise HTTPException(
                status_code=501,
                detail="Finite Difference method not yet implemented via API",
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown method: {method}",
            )

        return PriceResponse(
            price=result["price"],
            delta=result.get("delta"),
            gamma=result.get("gamma"),
            method=method,
            parameters=params,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model")
def load_model(checkpoint_path: str):
    """Load a DGM model from checkpoint."""
    try:
        pricing_service.load_model(checkpoint_path)
        return {"message": f"Model loaded from {checkpoint_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
