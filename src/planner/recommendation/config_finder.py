"""Capacity planning and GPU configuration recommendation.

IMPORTANT: PostgreSQL Migration (Phase 1):
- Uses traffic profile-based exact matching on (prompt_tokens, output_tokens)
- Queries benchmarks by exact traffic profile (512→256, 1024→1024, 4096→512, 10240→1536)
- Filters by p95 SLO compliance (TTFT, ITL, E2E)
- Uses pre-calculated e2e_p95 from benchmarks (not dynamic calculation)

Benchmarks collected using GuideLLM with fixed traffic profiles:
- Batching: vLLM continuous batching (dynamic, auto-configured)
- KV cache: enabled (vLLM default)
- Request pattern: steady-state load

TODO (Phase 2+): Parametric Performance Models
- Train regression models: f(prompt_tokens, output_tokens) → (ttft_p95, itl_p95, e2e_p95)
- Support arbitrary traffic profiles beyond the 4 GuideLLM defaults
- Interpolate for in-range predictions with confidence intervals
"""

import contextlib
import io
import logging
import math
import os
import time
from typing import Protocol

from planner.capacity_planner import check_model_fits_gpu, get_model_config_from_hf
from planner.gpu_recommender import GPURecommender
from planner.knowledge_base.benchmarks import BenchmarkData, BenchmarkRepository
from planner.knowledge_base.model_catalog import ModelCatalog, ModelInfo
from planner.shared.schemas import (
    ConfigurationScores,
    DeploymentIntent,
    DeploymentRecommendation,
    GPUConfig,
    SLOTargets,
    TrafficProfile,
)
from planner.shared.utils import normalize_gpu_types

from .analyzer import get_task_bonus
from .scorer import Scorer

logger = logging.getLogger(__name__)


class QualityScorer(Protocol):
    """Protocol for quality scoring backends."""

    def get_quality_score(self, model_name: str, use_case: str) -> float: ...


class ConfigFinder:
    """Plan GPU capacity to meet SLO targets and traffic requirements."""

    def __init__(
        self,
        benchmark_repo: BenchmarkRepository | None = None,
        catalog: ModelCatalog | None = None,
        quality_scorer: QualityScorer | None = None,
    ):
        """
        Initialize capacity planner.

        Args:
            benchmark_repo: PostgreSQL benchmark repository.
            catalog: Model catalog
            quality_scorer: Optional scorer with get_quality_score(model_name, use_case) method.
                           When provided, replaces the default UseCaseQualityScorer.
        """
        self.benchmark_repo = benchmark_repo or BenchmarkRepository()
        self.catalog = catalog or ModelCatalog()
        self._quality_scorer = quality_scorer

    def _calculate_required_replicas(self, qps_per_replica: float, required_qps: float) -> int:
        """
        Calculate number of replicas needed for traffic.

        Args:
            qps_per_replica: QPS capacity per replica
            required_qps: Required QPS to handle

        Returns:
            Number of replicas (minimum 1)
        """
        if qps_per_replica <= 0:
            return 0  # Infeasible: cannot serve positive throughput

        # Add 20% headroom for safety
        headroom_factor = 1.2
        required_capacity = required_qps * headroom_factor

        replicas = math.ceil(required_capacity / qps_per_replica)
        return max(1, replicas)

    def _generate_reasoning_from_bench(
        self,
        bench: BenchmarkData,
        gpu_config: GPUConfig,
        intent: DeploymentIntent,
        model: ModelInfo | None = None,
    ) -> str:
        """Generate explanation for recommendation from benchmark data.

        Args:
            bench: Benchmark data
            gpu_config: GPU configuration
            intent: Deployment intent
            model: Model info (optional, may be None if not in catalog)

        Returns:
            Reasoning string
        """
        reasons = []

        # Model selection
        if model:
            reasons.append(
                f"Selected {model.name} ({model.size_parameters}) for {intent.use_case} use case"
            )
        else:
            reasons.append(f"Selected {bench.model_hf_repo} for {intent.use_case} use case")

        # GPU configuration
        if gpu_config.tensor_parallel > 1:
            reasons.append(
                f"Using {gpu_config.tensor_parallel}x tensor parallelism on {gpu_config.gpu_type} "
                f"for improved latency"
            )
        else:
            reasons.append(f"Deploying on {gpu_config.gpu_type} GPUs")

        # Scaling
        if gpu_config.replicas > 1:
            reasons.append(
                f"{gpu_config.replicas} independent replicas to handle {intent.user_count} users"
            )

        # Performance
        ttft_p95 = int(bench.ttft_p95) if bench.ttft_p95 else 0
        itl_p95 = int(bench.itl_p95) if bench.itl_p95 else 0
        reasons.append(f"Expected performance: TTFT={ttft_p95}ms (p95), ITL={itl_p95}ms (p95)")

        return ". ".join(reasons)

    # Mapping from gpu_catalog.json gpu_type to llm_optimizer GPU_SPECS keys.
    # GPUs not in this map are not supported by the roofline model.
    _CATALOG_TO_ROOFLINE_GPU: dict[str, str] = {
        "H100": "H100",
        "H200": "H200",
        "A100-80": "A100",
        "A100-40": "A100-40GB",
        "L40": "L40",
        "L20": "L20",
        "B100": "B100",
        "B200": "B200",
    }

    @staticmethod
    def _convert_estimation_to_benchmark(
        model_id: str,
        gpu_type: str,
        gpu_count: int,
        prompt_tokens: int,
        output_tokens: int,
        ttft_ms: float,
        itl_ms: float,
        e2e_latency_ms: float,
        output_throughput_tps: float,
    ) -> BenchmarkData:
        """Convert GPU Recommender roofline output to BenchmarkData format.

        The roofline model produces single-point estimates (no percentile
        distribution), so the same value is used for mean/p90/p95/p99.
        """
        rps = output_throughput_tps / output_tokens if output_tokens > 0 else 0.0

        data = {
            "model_hf_repo": model_id,
            "hardware": gpu_type,
            "hardware_count": gpu_count,
            "framework": "vllm",
            "framework_version": "estimated",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "mean_input_tokens": prompt_tokens,
            "mean_output_tokens": output_tokens,
            "ttft_mean": ttft_ms,
            "ttft_p90": ttft_ms,
            "ttft_p95": ttft_ms,
            "ttft_p99": ttft_ms,
            "itl_mean": itl_ms,
            "itl_p90": itl_ms,
            "itl_p95": itl_ms,
            "itl_p99": itl_ms,
            "e2e_mean": e2e_latency_ms,
            "e2e_p90": e2e_latency_ms,
            "e2e_p95": e2e_latency_ms,
            "e2e_p99": e2e_latency_ms,
            "tps_mean": output_throughput_tps,
            "tps_p90": output_throughput_tps,
            "tps_p95": output_throughput_tps,
            "tps_p99": output_throughput_tps,
            "tokens_per_second": output_throughput_tps,
            "requests_per_second": rps,
            "estimated": True,
            "source": "llm-optimizer",
            "confidence_level": "estimated",
            "model_uri": None,
        }
        return BenchmarkData(data)

    def _generate_estimated_configs(
        self,
        traffic_profile: TrafficProfile,
        slo_targets: SLOTargets,
        preferred_models: list[str],
        existing_benchmarks: list[BenchmarkData],
        gpu_types: list[str] | None,
        estimate_all_catalog: bool = False,
    ) -> tuple[list[BenchmarkData], list[str]]:
        """Generate estimated BenchmarkData for (model, GPU) pairs without benchmarks.

        Uses the capacity planner for memory feasibility and the BentoML roofline
        model for synthetic performance estimation. Results are written to the DB
        for future cache hits.

        Args:
            traffic_profile: Current traffic profile (prompt_tokens, output_tokens)
            slo_targets: SLO targets (TTFT, ITL, E2E)
            preferred_models: User-specified model IDs (HuggingFace format)
            existing_benchmarks: Benchmark results already found from DB
            gpu_types: GPU types to evaluate (None = all catalog GPUs)
            estimate_all_catalog: If True, also estimate for catalog models
                                 without benchmarks (not just user-specified)

        Returns:
            Tuple of (list of new BenchmarkData, list of warning messages)
        """
        warnings: list[str] = []

        # 1. Build covered set from existing benchmarks (includes prior roofline estimates)
        # Key is (model, gpu, tp) so different TP values are estimated independently.
        covered: set[tuple[str, str, int]] = set()
        for bench in existing_benchmarks:
            covered.add((bench.model_hf_repo.lower(), bench.hardware.lower(), bench.hardware_count))

        # 2. Determine models to estimate
        models_to_estimate: list[str] = []
        for model_id in preferred_models:
            models_to_estimate.append(model_id)

        if estimate_all_catalog:
            for model_info in self.catalog.get_all_models():
                if model_info.model_id not in models_to_estimate:
                    models_to_estimate.append(model_info.model_id)

        if not models_to_estimate:
            return [], warnings

        # 3. Determine GPUs to evaluate
        if gpu_types:
            catalog_gpus = [
                gt for gt in self.catalog.get_all_gpu_types() if gt.gpu_type in gpu_types
            ]
        else:
            catalog_gpus = self.catalog.get_all_gpu_types()

        # Configurable limits
        max_models = int(os.getenv("PLANNER_ESTIMATED_MAX_MODELS", "5"))
        timeout_s = int(os.getenv("PLANNER_ESTIMATED_TIMEOUT_S", "60"))
        models_to_estimate = models_to_estimate[:max_models]

        hf_token = os.getenv("HF_TOKEN")
        new_benchmarks: list[BenchmarkData] = []
        start_time = time.monotonic()

        gpu_names = [g.gpu_type for g in catalog_gpus]
        logger.info(
            f"Estimation plan: {len(models_to_estimate)} models × "
            f"{len(catalog_gpus)} GPUs {gpu_names}, "
            f"{len(covered)} (model, GPU, TP) combinations already covered"
        )
        for model_id in models_to_estimate:
            logger.info(f"  model: {model_id}")

        # 4. For each model, check feasibility and estimate performance
        for model_idx, model_id in enumerate(models_to_estimate, 1):
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_s:
                remaining = len(models_to_estimate) - model_idx + 1
                msg = (
                    f"Estimation timeout ({timeout_s}s) reached after {elapsed:.0f}s. "
                    f"Skipping {remaining} remaining model(s)."
                )
                logger.warning(msg)
                warnings.append(msg)
                break
            logger.info(f"Estimating model {model_idx}/{len(models_to_estimate)}: {model_id}")
            # Fetch model config from HuggingFace (suppress noisy stdout
            # from safetensors tqdm progress bar)
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    model_config = get_model_config_from_hf(model_id, hf_token)
            except Exception as e:
                msg = f"Could not estimate performance for {model_id}: {e}"
                logger.warning(msg)
                warnings.append(msg)
                continue

            model_had_any_gpu = False
            model_checked_any_gpu = False
            model_had_error = False

            for gpu_info in catalog_gpus:
                # Map catalog GPU name to roofline model name
                roofline_gpu = self._CATALOG_TO_ROOFLINE_GPU.get(gpu_info.gpu_type)
                if not roofline_gpu:
                    continue  # GPU not supported by roofline model

                model_checked_any_gpu = True

                # Check memory feasibility (suppress safetensors tqdm output)
                try:
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        valid_tps = check_model_fits_gpu(
                            model_id, model_config, gpu_info.memory_gb, hf_token=hf_token
                        )
                except Exception as e:
                    msg = f"Could not check GPU fit for {model_id} on {gpu_info.gpu_type}: {e}"
                    logger.warning(msg)
                    warnings.append(msg)
                    model_had_error = True
                    logger.info(f"Skipping remaining GPUs for {model_id} due to model-level error")
                    break
                if not valid_tps:
                    logger.info(
                        f"  {gpu_info.gpu_type}: model does not fit "
                        f"({gpu_info.memory_gb}GB) at any TP"
                    )
                    continue

                model_had_any_gpu = True

                # Estimate performance at each valid TP value
                for tp in valid_tps:
                    # Skip if already covered (e.g., from prior DB benchmark or earlier estimate)
                    if (model_id.lower(), gpu_info.gpu_type.lower(), tp) in covered:
                        continue

                    # Run roofline estimation (suppress noisy stdout from
                    # llm_optimizer click.echo and safetensors tqdm progress)
                    try:
                        with (
                            contextlib.redirect_stdout(io.StringIO()),
                            contextlib.redirect_stderr(io.StringIO()),
                        ):
                            recommender = GPURecommender(
                                model_id=model_id,
                                input_len=traffic_profile.prompt_tokens,
                                output_len=traffic_profile.output_tokens,
                                max_gpus=tp,
                                gpu_list=[roofline_gpu],
                                max_ttft=slo_targets.ttft_p95_target_ms,
                                max_itl=slo_targets.itl_p95_target_ms,
                                max_latency=slo_targets.e2e_p95_target_ms / 1000,
                                catalog=self.catalog,
                            )
                            gpu_results, failed_gpus = recommender.get_gpu_results()
                    except Exception as e:
                        msg = f"Roofline estimation failed for {model_id} on {gpu_info.gpu_type} TP={tp}: {e}"
                        logger.warning(msg)
                        warnings.append(msg)
                        continue

                    if roofline_gpu in failed_gpus:
                        # Don't surface per-TP constraint failures as warnings —
                        # higher TP values may still succeed
                        continue

                    if roofline_gpu not in gpu_results:
                        continue

                    result = gpu_results[roofline_gpu]
                    best_latency = (
                        result.best_configs.get("best_latency")
                        if isinstance(result.best_configs, dict)
                        else None
                    )
                    if not best_latency or not hasattr(best_latency, "ttft_ms"):
                        continue

                    # Convert to BenchmarkData
                    bench = self._convert_estimation_to_benchmark(
                        model_id=model_id,
                        gpu_type=gpu_info.gpu_type,  # Use catalog name for DB consistency
                        gpu_count=tp,
                        prompt_tokens=traffic_profile.prompt_tokens,
                        output_tokens=traffic_profile.output_tokens,
                        ttft_ms=best_latency.ttft_ms,
                        itl_ms=best_latency.itl_ms,
                        e2e_latency_ms=best_latency.e2e_latency_s * 1000,
                        output_throughput_tps=best_latency.output_throughput_tps,
                    )
                    new_benchmarks.append(bench)
                    covered.add((model_id.lower(), gpu_info.gpu_type.lower(), tp))

            # Only warn "does not fit" when the model genuinely doesn't fit —
            # not when a network error prevented the check.
            if not model_had_any_gpu and model_checked_any_gpu and not model_had_error:
                msg = f"Model {model_id} does not fit on any available GPU"
                logger.warning(msg)
                warnings.append(msg)

        # 5. Write new estimates to DB for future cache hits
        if new_benchmarks:
            try:
                self.benchmark_repo.save_benchmarks(new_benchmarks)
                logger.info(f"Wrote {len(new_benchmarks)} roofline estimates to DB")
            except Exception as e:
                msg = f"Failed to persist roofline estimates to DB: {type(e).__name__}: {e}"
                logger.warning(msg)
                warnings.append(msg)

        return new_benchmarks, warnings

    def plan_all_capacities(
        self,
        traffic_profile: TrafficProfile,
        slo_targets: SLOTargets,
        intent: DeploymentIntent,
        include_near_miss: bool = False,  # Strict SLO filtering - no tolerance
        near_miss_tolerance: float = 0.0,  # No near-miss tolerance
        weights: dict[str, int] | None = None,  # Custom weights for balanced score
        cluster_gpu_types: list[str] | None = None,
        preferred_models: list[str] | None = None,
        enable_estimated: bool = True,
    ) -> tuple[list[DeploymentRecommendation], list[str]]:
        """
        Plan GPU capacity and return ALL viable configurations meeting SLO.

        Queries benchmarks for all (model, GPU) configurations meeting SLO targets,
        then scores each on accuracy, price, latency, and complexity.

        Args:
            traffic_profile: Traffic characteristics (prompt_tokens, output_tokens)
            slo_targets: p95 SLO targets
            intent: Original deployment intent
            include_near_miss: Whether to include configs within tolerance of SLO
            near_miss_tolerance: How much over SLO to allow (0.2 = 20%)
            weights: Custom weights for balanced score (0-10 scale)
                     Keys: accuracy, price, latency, complexity
            cluster_gpu_types: Detected GPU types from cluster (None = detection
                not attempted, [] = no GPUs detected, non-empty = hard filter
                intersected with user preferences)
            preferred_models: User-specified model IDs to include via estimated
                performance when no benchmark data exists
            enable_estimated: Whether to run roofline estimation for models/GPUs
                without benchmark data (default True)

        Returns:
            Tuple of (list of DeploymentRecommendations with scores, list of warning messages)
        """
        scorer = Scorer()
        all_configs: list[DeploymentRecommendation] = []

        # Determine SLO thresholds for query
        # If including near-miss, relax thresholds by tolerance
        if include_near_miss:
            query_ttft = int(slo_targets.ttft_p95_target_ms * (1 + near_miss_tolerance))
            query_itl = int(slo_targets.itl_p95_target_ms * (1 + near_miss_tolerance))
            query_e2e = int(slo_targets.e2e_p95_target_ms * (1 + near_miss_tolerance))
        else:
            query_ttft = slo_targets.ttft_p95_target_ms
            query_itl = slo_targets.itl_p95_target_ms
            query_e2e = slo_targets.e2e_p95_target_ms

        # Get percentile from SLO targets (default to p95 for backwards compatibility)
        percentile = getattr(slo_targets, "percentile", "p95")

        # Normalize user's preferred GPU types
        normalized_user_gpus = normalize_gpu_types(intent.preferred_gpu_types)

        # Determine effective GPU filter by intersecting cluster and user preferences
        # cluster_gpu_types semantics:
        #   None or [] = no cluster detection / detection failed -> use user prefs only
        #   non-empty list = detected cluster GPUs -> intersect with user prefs
        if cluster_gpu_types:
            if normalized_user_gpus:
                effective_gpus = sorted(set(cluster_gpu_types) & set(normalized_user_gpus))
                logger.info(
                    f"Cluster GPUs: {cluster_gpu_types}. "
                    f"User preference: {normalized_user_gpus}. "
                    f"Effective filter: {effective_gpus}"
                )
                if not effective_gpus:
                    logger.warning(
                        "No overlap between cluster GPUs and user preference — "
                        "no configurations possible"
                    )
                    return [], []
            else:
                effective_gpus = sorted(cluster_gpu_types)
                logger.info(f"Using cluster GPUs as filter: {effective_gpus}")
        elif normalized_user_gpus:
            effective_gpus = normalized_user_gpus
            logger.info(f"Filtering by user preferred GPUs: {effective_gpus}")
        else:
            effective_gpus = []

        normalized_gpus = effective_gpus

        # Track whether the GPU filter came from cluster detection (vs user preference)
        # so we can fall back to all GPUs if cluster GPUs have no benchmark data.
        gpu_filter_from_cluster = bool(cluster_gpu_types) and not normalized_user_gpus

        # Query PostgreSQL for configurations meeting relaxed SLO targets
        matching_configs = self.benchmark_repo.find_configurations_meeting_slo(
            prompt_tokens=traffic_profile.prompt_tokens,
            output_tokens=traffic_profile.output_tokens,
            ttft_p95_max_ms=query_ttft,
            itl_p95_max_ms=query_itl,
            e2e_p95_max_ms=query_e2e,
            min_qps=0,
            percentile=percentile,
            gpu_types=normalized_gpus if normalized_gpus else None,
        )

        # Fallback: if cluster-detected GPUs had no benchmark data, retry
        # without GPU filter so the user still gets recommendations.
        if not matching_configs and normalized_gpus and gpu_filter_from_cluster:
            logger.warning(
                f"No benchmarks found for cluster GPUs {normalized_gpus} — "
                f"falling back to all available GPUs"
            )
            matching_configs = self.benchmark_repo.find_configurations_meeting_slo(
                prompt_tokens=traffic_profile.prompt_tokens,
                output_tokens=traffic_profile.output_tokens,
                ttft_p95_max_ms=query_ttft,
                itl_p95_max_ms=query_itl,
                e2e_p95_max_ms=query_e2e,
                min_qps=0,
                percentile=percentile,
                gpu_types=None,
            )

        # Estimated performance flow: generate roofline estimates for
        # preferred models (and optionally catalog models) that lack benchmark data.
        all_warnings: list[str] = []
        if enable_estimated and preferred_models:
            estimated_configs, estimation_warnings = self._generate_estimated_configs(
                traffic_profile=traffic_profile,
                slo_targets=slo_targets,
                preferred_models=preferred_models,
                existing_benchmarks=matching_configs,
                gpu_types=normalized_gpus if normalized_gpus else None,
            )
            all_warnings.extend(estimation_warnings)
            if estimated_configs:
                matching_configs.extend(estimated_configs)
                logger.info(
                    f"Added {len(estimated_configs)} estimated configurations "
                    f"from roofline model"
                )

        # When the user specified preferred models, filter results to only
        # those models.  Fall back to all configs if none of the preferred
        # models produced viable results.
        if preferred_models:
            preferred_set = {m.lower() for m in preferred_models}
            preferred_configs = [
                c for c in matching_configs if c.model_hf_repo.lower() in preferred_set
            ]
            if preferred_configs:
                logger.info(
                    f"Filtering to {len(preferred_configs)} configs for "
                    f"preferred models (from {len(matching_configs)} total)"
                )
                matching_configs = preferred_configs
            else:
                logger.info("No configs for preferred models — showing all available")

        if not matching_configs:
            logger.warning(
                f"No configurations found for traffic profile "
                f"({traffic_profile.prompt_tokens}→{traffic_profile.output_tokens})"
                + (f" with GPUs {normalized_gpus}" if normalized_gpus else "")
            )
            return [], all_warnings

        # Build model lookup from catalog for scoring
        # Models not in catalog will get accuracy score = 0
        all_models = self.catalog.get_all_models()
        model_lookup = {m.model_id.lower(): m for m in all_models}

        # Process each matching benchmark (no pre-filtering by model list)
        for bench in matching_configs:
            # Look up model in catalog (may be None if not in catalog)
            model = model_lookup.get(bench.model_hf_repo.lower())

            # Calculate required replicas to handle traffic
            replicas = self._calculate_required_replicas(
                bench.requests_per_second, traffic_profile.expected_qps or 1.0
            )
            if replicas == 0:
                continue  # Zero-throughput benchmark — infeasible config

            # Create GPU config - gpu_count is PER REPLICA, not total
            gpu_config = GPUConfig(
                gpu_type=bench.hardware,
                gpu_count=bench.hardware_count,  # Per-replica GPU count
                tensor_parallel=bench.hardware_count,
                replicas=replicas,
            )

            # Calculate cost using TOTAL GPUs (per-replica * replicas)
            total_gpus = bench.hardware_count * replicas
            cost_per_hour = self.catalog.calculate_gpu_cost(
                bench.hardware, total_gpus, hours_per_month=1
            )

            if cost_per_hour is None:
                logger.warning(f"Could not calculate cost for {bench.hardware}")
                continue

            cost_per_month = cost_per_hour * 730  # ~30 days

            # Calculate latency score and SLO status
            predicted_ttft = int(bench.ttft_p95) if bench.ttft_p95 else 0
            predicted_itl = int(bench.itl_p95) if bench.itl_p95 else 0
            predicted_e2e = int(bench.e2e_p95) if bench.e2e_p95 else 0

            latency_score, slo_status = scorer.score_latency(
                predicted_ttft_ms=predicted_ttft,
                predicted_itl_ms=predicted_itl,
                predicted_e2e_ms=predicted_e2e,
                target_ttft_ms=slo_targets.ttft_p95_target_ms,
                target_itl_ms=slo_targets.itl_p95_target_ms,
                target_e2e_ms=slo_targets.e2e_p95_target_ms,
                use_case=intent.use_case,
                near_miss_tolerance=near_miss_tolerance,
            )

            # Skip if exceeds SLO and we're not including near-miss
            if slo_status == "exceeds" and not include_near_miss:
                continue

            # Calculate accuracy score - USE RAW BENCHMARK SCORE
            # This is the actual model accuracy from benchmarks (AA or Model Catalog)
            # NOT a composite score with latency/budget bonuses
            model_name_for_scoring = model.name if model else bench.model_hf_repo
            if self._quality_scorer is not None:
                raw_accuracy = self._quality_scorer.get_quality_score(
                    model_name_for_scoring, intent.use_case
                )
                if raw_accuracy == 0 and bench.model_hf_repo:
                    raw_accuracy = self._quality_scorer.get_quality_score(
                        bench.model_hf_repo, intent.use_case
                    )
            else:
                from .quality import score_model_quality

                raw_accuracy = score_model_quality(model_name_for_scoring, intent.use_case)
                if raw_accuracy == 0 and bench.model_hf_repo:
                    raw_accuracy = score_model_quality(bench.model_hf_repo, intent.use_case)

            accuracy_score = int(raw_accuracy)

            # Fallback: for models without accuracy benchmarks (e.g., estimated models),
            # use parameter-count-based heuristic so they aren't filtered by min_accuracy
            if accuracy_score == 0 and getattr(bench, "confidence_level", None) == "estimated":
                model_size = model.size_parameters if model else bench.model_hf_repo
                accuracy_score = scorer.score_accuracy_by_size(model_size)

            # Apply task-specific bonus to accuracy score
            # This boosts models that are well-suited for the specific use case
            task_bonus = get_task_bonus(model_name_for_scoring, intent.use_case)
            accuracy_score = min(accuracy_score + task_bonus, 100)  # Cap at 100

            complexity_score = scorer.score_complexity(total_gpus)  # Use total GPUs for complexity

            # Determine model_id and model_name
            # Use catalog info if available, otherwise use benchmark model_hf_repo
            model_id = model.model_id if model else bench.model_hf_repo
            model_name = model.name if model else bench.model_hf_repo

            # Build benchmark_metrics with all percentile values for UI display
            benchmark_metrics = {
                "ttft_mean": int(bench.ttft_mean) if bench.ttft_mean else 0,
                "ttft_p90": int(bench.ttft_p90) if bench.ttft_p90 else 0,
                "ttft_p95": int(bench.ttft_p95) if bench.ttft_p95 else 0,
                "ttft_p99": int(bench.ttft_p99) if bench.ttft_p99 else 0,
                "itl_mean": int(bench.itl_mean) if bench.itl_mean else 0,
                "itl_p90": int(bench.itl_p90) if bench.itl_p90 else 0,
                "itl_p95": int(bench.itl_p95) if bench.itl_p95 else 0,
                "itl_p99": int(bench.itl_p99) if bench.itl_p99 else 0,
                "e2e_mean": int(bench.e2e_mean) if bench.e2e_mean else 0,
                "e2e_p90": int(bench.e2e_p90) if bench.e2e_p90 else 0,
                "e2e_p95": int(bench.e2e_p95) if bench.e2e_p95 else 0,
                "e2e_p99": int(bench.e2e_p99) if bench.e2e_p99 else 0,
                "tps_mean": float(bench.tps_mean) if bench.tps_mean else 0,
                "tps_p90": float(bench.tps_p90) if bench.tps_p90 else 0,
                "tps_p95": float(bench.tps_p95) if bench.tps_p95 else 0,
                "tps_p99": float(bench.tps_p99) if bench.tps_p99 else 0,
                # RPS per replica from benchmark (for card display)
                "requests_per_second": float(bench.requests_per_second)
                if bench.requests_per_second
                else 0,
                # Data validation flag: True = estimated/interpolated, False = real benchmark
                "estimated": getattr(bench, "estimated", False),
                # Classification fields for UI badges
                "source": getattr(bench, "source", "other"),
                "confidence_level": getattr(bench, "confidence_level", "benchmarked"),
            }

            # Build recommendation (price score calculated later after we know min/max)
            recommendation = DeploymentRecommendation(
                intent=intent,
                traffic_profile=traffic_profile,
                slo_targets=slo_targets,
                model_id=model_id,
                model_name=model_name,
                model_uri=getattr(bench, "model_uri", None),
                gpu_config=gpu_config,
                predicted_ttft_p95_ms=predicted_ttft,
                predicted_itl_p95_ms=predicted_itl,
                predicted_e2e_p95_ms=predicted_e2e,
                predicted_throughput_qps=bench.requests_per_second * replicas,
                cost_per_hour_usd=cost_per_hour,
                cost_per_month_usd=cost_per_month,
                meets_slo=(slo_status == "compliant"),
                reasoning=self._generate_reasoning_from_bench(bench, gpu_config, intent, model),
                benchmark_metrics=benchmark_metrics,  # All percentile data for UI
                # Temporary scores without price (will be updated below)
                scores=ConfigurationScores(
                    accuracy_score=accuracy_score,
                    price_score=0,  # Placeholder
                    latency_score=latency_score,
                    complexity_score=complexity_score,
                    balanced_score=0.0,  # Placeholder
                    slo_status=slo_status,
                ),
            )

            all_configs.append(recommendation)

        if not all_configs:
            logger.warning("No viable configurations found for any model")
            return [], all_warnings

        # Now calculate price scores (need min/max across all configs)
        costs = [rec.cost_per_month_usd for rec in all_configs if rec.cost_per_month_usd]
        if costs:
            min_cost = min(costs)
            max_cost = max(costs)

            for rec in all_configs:
                if rec.scores and rec.cost_per_month_usd:
                    # Update price score
                    price_score = scorer.score_price(rec.cost_per_month_usd, min_cost, max_cost)
                    rec.scores.price_score = price_score

                    # Calculate base balanced score with user weights
                    # Weights from UI are 0-10 integers, normalize to fractions
                    normalized_weights = None
                    if weights:
                        total = sum(weights.values()) or 1  # Avoid division by zero
                        normalized_weights = {k: v / total for k, v in weights.items()}

                    base_balanced = scorer.score_balanced(
                        accuracy_score=rec.scores.accuracy_score,
                        price_score=price_score,
                        latency_score=rec.scores.latency_score,
                        complexity_score=rec.scores.complexity_score,
                        weights=normalized_weights,
                    )

                    # Apply scalability penalty based on replica count
                    # Configs needing many replicas are less efficient for high workloads
                    replicas = rec.gpu_config.replicas if rec.gpu_config else 1
                    if replicas <= 1:
                        scalability_factor = 1.0  # No penalty
                    elif replicas <= 3:
                        scalability_factor = 0.98  # 2% penalty
                    elif replicas <= 6:
                        scalability_factor = 0.95  # 5% penalty
                    elif replicas <= 10:
                        scalability_factor = 0.90  # 10% penalty
                    elif replicas <= 20:
                        scalability_factor = 0.80  # 20% penalty
                    else:
                        scalability_factor = 0.65  # 35% penalty for very large deployments

                    rec.scores.balanced_score = round(base_balanced * scalability_factor, 1)

        # Count unique models in configurations
        unique_models = {rec.model_id for rec in all_configs}
        logger.info(
            f"Found {len(all_configs)} viable configurations across {len(unique_models)} models"
        )
        return all_configs, all_warnings
