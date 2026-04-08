"""GPU recommender endpoint."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from llm_optimizer.predefined.gpus import GPU_SPECS
from pydantic import BaseModel

from planner.api.routes.common import handle_hf_error
from planner.gpu_recommender import GPURecommender

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["gpu-recommender"])


class EstimateRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    input_len: int
    output_len: int
    max_gpus: int = 1
    max_gpus_per_type: dict[str, int] | None = None
    gpu_list: list[str] | None = None
    max_ttft: float | None = None
    max_itl: float | None = None
    max_latency: float | None = None
    custom_gpu_costs: dict[str, float] | None = None


class EstimateResponse(BaseModel):
    success: bool
    input_parameters: dict[str, Any]
    estimated_best_performance: dict[str, Any]
    gpu_results: dict[str, Any]
    failed_gpus: dict[str, str]
    summary: dict[str, int]


@router.post("/estimate")
async def estimate(request: EstimateRequest) -> EstimateResponse:
    """Run GPU performance estimation for a model and workload.

    Returns per-GPU performance data and ranked best recommendations.
    """
    gpu_list = request.gpu_list if request.gpu_list else sorted(GPU_SPECS.keys())

    try:
        recommender = GPURecommender(
            model_id=request.model_id,
            input_len=request.input_len,
            output_len=request.output_len,
            max_gpus=request.max_gpus,
            max_gpus_per_type=request.max_gpus_per_type,
            gpu_list=gpu_list,
            max_ttft=request.max_ttft,
            max_itl=request.max_itl,
            max_latency=request.max_latency,
            custom_gpu_costs=request.custom_gpu_costs,
        )
        _, failed_gpus = recommender.get_gpu_results()
        performance_summary = recommender.get_performance_summary()
    except Exception as e:
        msg = str(e).lower()
        if "gated" in msg or "403" in msg or "unauthorized" in msg:
            handle_hf_error(e)
        logger.exception("Unexpected error during GPU estimation for model %s", request.model_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during GPU estimation.",
        ) from e

    input_params: dict[str, Any] = {
        "model": request.model_id,
        "input_len": request.input_len,
        "output_len": request.output_len,
        "max_gpus": request.max_gpus,
        "gpu_list": gpu_list,
    }
    if request.max_ttft is not None:
        input_params["max_ttft_ms"] = request.max_ttft
    if request.max_itl is not None:
        input_params["max_itl_ms"] = request.max_itl
    if request.max_latency is not None:
        input_params["max_latency_s"] = request.max_latency
    if request.max_gpus_per_type:
        input_params["max_gpus_per_type"] = request.max_gpus_per_type

    return EstimateResponse(
        success=True,
        input_parameters=input_params,
        estimated_best_performance=performance_summary["estimated_best_performance"],
        gpu_results=performance_summary["gpu_results"],
        failed_gpus=failed_gpus,
        summary={
            "total_gpus_analyzed": len(gpu_list),
            "failed_gpus": len(failed_gpus),
        },
    )
