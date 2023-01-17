import logging
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from kedro.config import ConfigLoader
from kedro.io import DataCatalog

from .schemas import PredictPayload

APP_VERSION = "0.1"
app = FastAPI(
    title="my inference app",
    description="predict given a payload",
    version=APP_VERSION,
)

PROD_VERSION_MAPPING = {"0.1": "2023-01-17T15.47.21.700Z"}


#


def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.on_event("startup")
async def load_model() -> None:
    global model
    conf_path = Path(__file__).parents[1] / "conf"
    config = ConfigLoader(conf_source=conf_path, env="base")
    prod_version = {
        "model": PROD_VERSION_MAPPING[APP_VERSION],
    }
    logging.info("Loading model...")
    io = DataCatalog.from_config(
        catalog=config.get("catalog*"), load_versions=prod_version
    )
    model = io.load("model")
    logging.info("Model ready to use for inference!")


@construct_response
@app.get("/", tags=["API"])
def health(request: Request) -> dict:
    """health check"""
    return {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": {}}


@construct_response
@app.post("/predict", tags=["Pred"])
async def predict(request: Request, payload: PredictPayload) -> dict:
    """Return prediction for a given payload"""
    X = pd.DataFrame(payload.dict())

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": model.predict(X).tolist()},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
