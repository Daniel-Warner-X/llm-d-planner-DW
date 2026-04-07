# Design: Estimated Performance Integration

**Issue**: [#137 — Integration of Config Explorer capabilities into recommendation pipeline](https://github.com/llm-d-incubation/llm-d-planner/issues/137)

**Status**: Phase 1 (Backend Core) and Phase 2 (UI) implemented. Phase 3 (Validation CLI) pending.

**Goal**: When benchmark data is unavailable for a model/GPU combination, generate estimated performance using the Capacity Planner (memory feasibility) and GPU Recommender (BentoML roofline model), and present those estimates alongside benchmark-based recommendations.

---

## 1. Problem Statement

The Planner recommendation pipeline currently relies exclusively on performance benchmarks stored in PostgreSQL. The benchmark database covers a limited matrix of (model, GPU, traffic profile) combinations. When a user requests a model or GPU that lacks benchmark data, the system returns zero results.

Two existing in-process components can fill this gap:

- **Capacity Planner** (`src/planner/capacity_planner.py`) — calculates GPU memory requirements and determines whether a model physically fits on a given GPU at various tensor parallelism (TP) degrees.
- **GPU Recommender** (`src/planner/gpu_recommender.py`) — uses BentoML's `llm_optimizer` roofline model to estimate TTFT, ITL, E2E latency, and throughput without running actual benchmarks.

---

## 2. Requirements

### Functional Requirements

| ID | Requirement | Details |
|----|-------------|---------|
| FR-1 | **Model selection in user intent** | Users can specify one or more models in their natural language description ("I want to deploy Llama 3.3 70B"). The LLM extraction prompt extracts `preferred_models` (analogous to the existing `preferred_gpu_types` extraction in `src/planner/llm/prompts.py`). Users can then review and modify the extracted models via the "Modify Extracted Context" form, which provides both a multi-select from `data/configuration/model_catalog.json` and free-text entry for arbitrary HuggingFace model IDs. |
| FR-2 | **GPU selection from catalog** | Replace the hardcoded GPU list in `ui/components/extraction.py:165` (`["Any GPU", "H100", "A100", "A10G", "L4", "T4"]`) with a dynamic list from `data/configuration/gpu_catalog.json`. Support multi-select. |
| FR-3 | **Display constraints in Technical Specification** | Show user-selected GPUs and models in the Technical Specification tab. Display "Any" when no filter is applied, or list specific selections. |
| FR-4 | **Editable GPU and model lists** | Both GPU and model lists are editable via multi-select widgets in the "Modify Extracted Context" form. |
| FR-5 | **Estimated performance fallback** | When benchmark data is missing for (model, GPU) combinations, generate estimated performance, store it in `exported_summaries` (with `source='llm-optimizer'`, `confidence_level='estimated'`), and merge results into the recommendation pipeline. On subsequent requests, prior roofline estimates are returned by the existing DB query — the roofline model only runs for genuinely new combinations. |
| FR-6 | **Visual differentiation** | Recommendation cards display a badge based on `confidence_level`: **"Benchmarked"** (green) for real or validated benchmarks, **"Estimated"** (amber) for roofline model estimates. |
| FR-7 | **Toggle for estimated recommendations** | Configuration tab includes a toggle to enable/disable estimated recommendations. Default: enabled. |
| FR-8 | **Configurable scope** | Default: estimated flow triggers only for user-specified models. Configurable: enable for all catalog models without benchmark data. |
| FR-9 | **Persistence of estimated results** | Estimated results are written to the existing `exported_summaries` table with `source='llm-optimizer'` and `confidence_level='estimated'`. On subsequent requests, the DB query returns them alongside other benchmarks — the roofline model only runs for (model, GPU, traffic profile) combinations not already in the DB. Roofline estimates can be cleared independently via `DELETE WHERE source = 'llm-optimizer'` (API/UI). |
| FR-10 | **DB source and confidence classification** | Repurpose the existing `source` column (currently `local`/`model_catalog`) to identify the benchmark tool/method: `guidellm`, `blis`, `llm-optimizer`, `manual`, `model_catalog`, `llmd-benchmark`, etc. Add a new `confidence_level` column with 2 values: `benchmarked` (real or validated benchmarks, including physics-based simulation like BLIS), `estimated` (analytical/roofline models). The mapping from source to confidence level is set at data load time. Both columns are free-form text for extensibility. |
| FR-11 | **Test coverage** | All new and modified functions must have unit tests. Existing functions that are touched by these changes must have tests added if they don't already exist. |

### Non-Functional Requirements

| ID | Requirement | Details |
|----|-------------|---------|
| NFR-1 | **Configurable latency budget** | HuggingFace API calls for model config may take 2-5s (cached after first call). Roofline estimation is ~0.5-2s per (model, GPU) pair. The maximum number of models to evaluate and the overall timeout must be configurable (environment variable or config file). Default cap: 5 models, 60s timeout. |
| NFR-2 | **Graceful degradation with user feedback** | All estimation failures (HF API unreachable, model doesn't exist, roofline failures) must be reported to the user in the response metadata, not just logged. The UI should display warnings for skipped models (e.g., "Could not estimate performance for model X: HuggingFace API unreachable"). |
| NFR-3 | **Estimated results persistence** | Estimated results are stored in `exported_summaries` (see FR-9). They can be cleared independently via `source` filter (`DELETE WHERE source = 'llm-optimizer'`) from the Configuration tab and API. |

---

## 3. Request Flow

### 3.1 Current Flow

```
User Input
  --> LLM Extraction
  --> DeploymentIntent (use_case, user_count, preferred_gpu_types)
  --> TrafficProfile + SLOTargets
  --> ConfigFinder.plan_all_capacities()
        [1] normalize_gpu_types()
        [2] find_configurations_meeting_slo() --> PostgreSQL
        [3] If no results AND GPU filter was from cluster detection:
            retry without GPU filter
        [4] If still no results: return []          <-- dead end
        [5] For each BenchmarkData: build GPUConfig, score, rank
  --> DeploymentRecommendation[]
```

### 3.2 Proposed Flow

```
User Input
  --> LLM Extraction
  --> DeploymentIntent (use_case, user_count, preferred_gpu_types,
                        preferred_models)                          <-- NEW
  --> TrafficProfile + SLOTargets
  --> ConfigFinder.plan_all_capacities()
        [1] normalize_gpu_types()
        [2] find_configurations_meeting_slo() --> PostgreSQL
        [3] If no results AND GPU filter was from cluster detection:
            retry without GPU filter
        [4] NEW: _generate_estimated_configs()                     <-- NEW
            [a] Determine which (model, GPU, TP) combos need estimation:
                - Build set of (model, GPU, TP) triples already in DB results
                  (includes prior roofline estimates from earlier requests)
                - User-specified models with uncovered combos
                - (Optionally) catalog models with uncovered combos
            [b] Determine GPUs to evaluate:
                - User-specified GPUs, or all catalog GPUs if none specified
            [c] For each model:
                - For each GPU:
                  - check_model_fits_gpu() --> list of valid TP values
                  - For each valid TP not already covered:
                    - GPURecommender.get_gpu_results() --> estimates
                    - Convert PerformanceEstimationResult --> BenchmarkData
                - Write new estimates to exported_summaries
                  (source='llm-optimizer', confidence_level='estimated')
            [d] Append new estimated configs to matching_configs
        [5] If no matching_configs (benchmark + estimated): return []
        [6] For each BenchmarkData: build GPUConfig, score, rank   (unchanged)
  --> DeploymentRecommendation[]
```

**Insertion point**: `src/planner/recommendation/config_finder.py`, after line 253 (cluster-GPU fallback), before line 255 (`if not matching_configs: return []`).

The key design principle: estimated results are injected as `BenchmarkData` objects before the scoring loop. The downstream scoring pipeline works unchanged. One change was made to the ranking layer: the Analyzer's ACCURACY-FIRST strategy (which pre-selected top N models by accuracy, then restricted all other cards to only those models) was removed. Each card (Best Accuracy, Lowest Cost, etc.) now sorts independently from all filtered configs. This prevents preferred models from being filtered out when they don't rank in the top N by accuracy.

---

## 4. APIs Between Components

### 4.1 New: `check_model_fits_gpu()`

**Location**: `src/planner/capacity_planner.py`

**Purpose**: Wraps existing building blocks into a single "does it fit?" check.

```python
def check_model_fits_gpu(
    model_name: str,
    model_config: AutoConfig,
    gpu_memory_gb: int,
    gpu_util: float = 0.9,
    hf_token: str | None = None,
) -> list[int]:
    """
    Check which tensor parallelism (TP) values allow the model to fit on the GPU.

    Returns sorted list of valid TP values where allocatable KV cache memory > 0.
    Empty list means the model does not fit at any TP.
    """
```

**Logic**:
1. Call `find_possible_tp(model_config)` to get architecturally valid TP divisors.
2. For each TP value, call `allocatable_kv_cache_memory(model_name, model_config, gpu_memory_gb, gpu_util, tp=tp)`.
3. Return sorted list of TP values where allocatable memory > 0.

**Examples**:
```
check_model_fits_gpu("meta-llama/Llama-3.3-70B-Instruct", config, 80)
  --> [2, 4, 8]     # 70B model needs TP>=2 on 80GB GPU

check_model_fits_gpu("meta-llama/Llama-3.1-8B-Instruct", config, 24)
  --> [1, 2, 4, 8]  # 8B model fits on 24GB GPU at TP=1

check_model_fits_gpu("meta-llama/Llama-3.3-70B-Instruct", config, 24)
  --> []             # Too large for L4 at any TP
```

**Existing functions used** (no changes needed):
- `find_possible_tp()` (`capacity_planner.py`)
- `allocatable_kv_cache_memory()` (`capacity_planner.py`)
- `get_model_config_from_hf()` (`capacity_planner.py`) — called by the orchestrator, not this function

### 4.2 New: `_generate_estimated_configs()`

**Location**: `src/planner/recommendation/config_finder.py` (private method on `ConfigFinder`)

```python
def _generate_estimated_configs(
    self,
    traffic_profile: TrafficProfile,
    slo_targets: SLOTargets,
    preferred_models: list[str],
    existing_benchmarks: list[BenchmarkData],
    gpu_types: list[str] | None,
    estimate_all_catalog: bool = False,
) -> tuple[list[BenchmarkData], list[str]]:
    """
    Generate estimated BenchmarkData for (model, GPU) pairs without benchmarks.

    Args:
        traffic_profile: Current traffic profile (prompt_tokens, output_tokens)
        slo_targets: SLO targets (TTFT, ITL, E2E)
        preferred_models: User-specified model IDs (HuggingFace format)
        existing_benchmarks: Benchmark results already found from DB
        gpu_types: GPU types to evaluate (None = all catalog GPUs)
        estimate_all_catalog: If True, also estimate for catalog models
                             without benchmarks (not just user-specified)

    Returns:
        Tuple of (list of BenchmarkData with estimated=True, list of warning strings)
    """
```

**Orchestration logic**:

1. Build the set of (model, GPU) pairs already covered by `existing_benchmarks` returned from the DB query. This includes any prior roofline estimates that were stored on earlier requests — they are already in `exported_summaries` and returned by `find_configurations_meeting_slo()`.
2. Determine models to estimate:
   - Always include `preferred_models` whose (model, GPU) pairs are not fully covered.
   - If `estimate_all_catalog`: also include catalog models with uncovered pairs.
3. Determine GPUs to evaluate:
   - If `gpu_types` is provided: use those.
   - Otherwise: use all GPUs from `gpu_catalog.json` via `self.catalog.get_all_gpu_types()`.
4. For each model:
   - Fetch model config via `get_model_config_from_hf()` (cached).
   - For each GPU:
     - Call `check_model_fits_gpu()` — get valid TP values.
     - If no valid TP: skip.
     - For each valid TP value:
       - Skip if (model, GPU, TP) already in covered set (from step 1).
       - Call `GPURecommender` with this model, GPU, TP, traffic profile, and SLO constraints.
       - Convert results to `BenchmarkData` with `source='llm-optimizer'`, `confidence_level='estimated'`.
     - Write all new estimates to `exported_summaries` so future requests get a DB hit.
5. Return list of newly estimated `BenchmarkData` objects (appended to `matching_configs`).

### 4.3 Existing: `GPURecommender` (modified)

**Location**: `src/planner/gpu_recommender.py`

The existing `GPURecommender` class provides the core estimation logic. One fix was required: the constraint string passed to `llm_optimizer` was concatenating parts without delimiters. The `llm_optimizer` library expects semicolon-separated constraints (`split(";")`), so constraint building was changed to use `";".join(constraint_parts)`.

```python
recommender = GPURecommender(
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    input_len=512,                    # From traffic_profile.prompt_tokens
    output_len=256,                   # From traffic_profile.output_tokens
    max_gpus=tp,                      # Each valid TP from check_model_fits_gpu()
    gpu_list=["H100"],                # Single GPU for this evaluation
    max_ttft=slo_targets.ttft_p95,    # SLO constraint
    max_itl=slo_targets.itl_p95,      # SLO constraint
    max_latency=slo_targets.e2e_p95 / 1000,  # SLO constraint (ms -> s)
)
gpu_results, failed_gpus = recommender.get_gpu_results()
```

**Output**: `dict[str, PerformanceEstimationResult]` keyed by GPU name.

Each `PerformanceEstimationResult.best_configs` is a dict with entries like `best_latency` containing:
- `ttft_ms` (float)
- `itl_ms` (float)
- `e2e_latency_s` (float)
- `output_throughput_tps` (float)

The constraint delimiter fix and GPU name mapping were the only changes needed.

### 4.4 New: `_convert_estimation_to_benchmark()`

**Location**: `src/planner/recommendation/config_finder.py` (static method on `ConfigFinder`)

```python
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
    """
    Convert GPU Recommender output to BenchmarkData format.

    The roofline model produces single-point estimates (no percentile
    distribution), so the same value is used for mean/p90/p95/p99.
    """
```

**Field mapping**:

| Source | BenchmarkData field(s) |
|--------|----------------------|
| `ttft_ms` | `ttft_mean`, `ttft_p90`, `ttft_p95`, `ttft_p99` (all same value) |
| `itl_ms` | `itl_mean`, `itl_p90`, `itl_p95`, `itl_p99` (all same value) |
| `e2e_latency_s * 1000` | `e2e_mean`, `e2e_p90`, `e2e_p95`, `e2e_p99` (all same, in ms) |
| `output_throughput_tps` | `tps_mean`, `tps_p90`, `tps_p95`, `tps_p99`, `tokens_per_second` |
| `output_throughput_tps / output_tokens` | `requests_per_second` |
| `model_id` | `model_hf_repo` |
| `gpu_type` | `hardware` |
| `gpu_count` | `hardware_count` |
| `prompt_tokens` | `prompt_tokens`, `mean_input_tokens` |
| `output_tokens` | `output_tokens`, `mean_output_tokens` |
| `"vllm"` | `framework` |
| `"estimated"` | `framework_version` |
| `True` | `estimated` |
| `"llm-optimizer"` | `source` |
| `"estimated"` | `confidence_level` |

### 4.5 Modified: `DeploymentIntent` Schema

**Location**: `src/planner/shared/schemas/intent.py`

Add field:
```python
preferred_models: list[str] = Field(
    default_factory=list,
    description="List of user's preferred model IDs (HuggingFace format, empty = any model). "
    "Can be catalog model_ids or arbitrary HF repo IDs.",
)
```

### 4.6 Modified: `plan_all_capacities()` Signature and Return Type

**Location**: `src/planner/recommendation/config_finder.py`

Add parameters:
```python
preferred_models: list[str] | None = None,
enable_estimated: bool = True,
```

**Return type changed** from `list[DeploymentRecommendation]` to `tuple[list[DeploymentRecommendation], list[str]]`. The second element contains estimation warnings (e.g., "Model X does not fit on any available GPU", "Estimation skipped for model X on GPU Y: ..."). All callers in `workflow.py` unpack this tuple and thread warnings through to `RankedRecommendationsResponse.warnings`.

**Preferred model filtering**: When `preferred_models` is specified, the final `matching_configs` list is filtered to only include configurations for those models. If no preferred models produced viable configs, all configs are returned as a fallback.

**Exception handling for `check_model_fits_gpu()`**: Some model architectures (e.g., GGUF repos) cause `NotASafetensorsRepoError` when `check_model_fits_gpu()` calls `get_safetensors_metadata()`. These exceptions are caught and the model is skipped with a warning.

### 4.7 Modified: `BenchmarkData`

**Location**: `src/planner/knowledge_base/benchmarks.py`

Add `source` and `confidence_level` fields to `__init__()` and `to_dict()`. See section 8.3 for details.

### 4.8 Modified: API Route

**Location**: `src/planner/api/routes/recommendation.py`

Add to `RankedRecommendationFromSpecRequest`:
```python
preferred_models: list[str] | None = None
enable_estimated: bool = True
```

Pass through to `plan_all_capacities()`.

---

## 5. UI Changes

### 5.1 Define Use Case Tab — Model Selection

**Location**: `ui/app.py` and `ui/components/extraction.py`

Add model selection after LLM extraction produces the initial intent:
- `st.multiselect` populated from model catalog IDs.
- `st.text_input` for free-text HuggingFace model IDs (comma-separated).
- Store in `st.session_state.preferred_models`.

### 5.2 Define Use Case Tab — GPU Selection Fix

**Location**: `ui/components/extraction.py`, `render_extraction_edit_form()`, line 165

Replace hardcoded list with dynamic loading via `fetch_gpu_types()` API client. Change from `st.selectbox` (single) to `st.multiselect` (multi).

### 5.3 Technical Specification Tab — Display Constraints

**Location**: `ui/components/slo.py`

Add a section showing:
- "GPUs: Any" or "GPUs: H100, A100-80"
- "Models: Any" or "Models: meta-llama/Llama-3.3-70B-Instruct, Qwen/Qwen3-32B"

### 5.4 Modify Extracted Context — Multi-select for Both

**Location**: `ui/components/extraction.py`, `render_extraction_edit_form()`

Replace single GPU selectbox with multi-select. Add model multi-select + free-text input.

### 5.5 Recommendation Cards — Confidence Badges

**Location**: `ui/components/recommendations.py`

Check `rec.get("benchmark_metrics", {}).get("confidence_level", "benchmarked")`:
- `"benchmarked"`: Green **"Benchmarked"** badge — tooltip: "Based on real hardware benchmark data."
- `"estimated"`: Amber **"Estimated"** badge — tooltip: "Based on roofline model estimation. Actual performance may vary."

### 5.6 Configuration Tab — Estimated Toggle

**Location**: `ui/components/settings.py`

Add toggle:
```python
st.subheader("Estimated Performance")
enable_estimated = st.toggle(
    "Enable estimated performance for models without benchmarks",
    value=True,
    key="enable_estimated",
)
```

---

## 6. Error Handling

All estimation failures are collected and returned in the API response metadata so the UI can display warnings to the user. This applies consistently to all failure scenarios — the user always knows what was skipped and why.

| Scenario | Handling |
|----------|----------|
| HuggingFace API unreachable | Skip model, collect in response `warnings` list for UI display (e.g., "Could not estimate performance for model X: HuggingFace API unreachable"), continue with remaining models |
| Model ID doesn't exist on HF | Skip, collect in response `warnings` list for UI display (e.g., "Model 'meta-llama/llama-3.3-70b-instuct' not found on HuggingFace") |
| Roofline estimation fails for (model, GPU) | Already handled by `GPURecommender.get_gpu_results()` — stored in `failed_gpus` dict; propagate to response `warnings` |
| Model fits no GPUs | Skip model entirely, add to `warnings` (e.g., "Model X does not fit on any available GPU") |
| No benchmarks AND no estimated results | Return empty list (existing behavior); enhance error message to mention estimation was attempted |
| Latency budget exceeded | Configurable cap on models evaluated (default 5) and overall timeout (default 60s). Use `check_model_fits_gpu()` as fast pre-filter before slower roofline model. Consider `ThreadPoolExecutor` for parallelization. |

---

## 7. Session State and Configuration

### New Session State Keys

Add to `ui/state.py` `SESSION_DEFAULTS`:
```python
"preferred_models": [],
"enable_estimated": True,
```

### API Request Payload Changes

`fetch_ranked_recommendations()` in `ui/api_client.py` must include:
```python
"preferred_models": preferred_models or [],
"enable_estimated": enable_estimated,
```

### Backend Configuration

The `estimate_all_catalog` setting (FR-8) is a backend configuration (environment variable or config file), not exposed in the UI initially.

Configurable latency budget settings (environment variables):
```
PLANNER_ESTIMATED_MAX_MODELS=5        # Max models to evaluate in estimated flow
PLANNER_ESTIMATED_TIMEOUT_S=60        # Overall timeout for estimated flow
```

---

## 8. Database Schema Changes

All benchmark data lives in the single `exported_summaries` table. No separate cache table is needed — roofline estimates are written to the same table and returned by the existing `find_configurations_meeting_slo()` query on subsequent requests. They can be selectively cleared via `DELETE WHERE source = 'llm-optimizer'`.

### 8.1 Repurpose `source` Column

The `source` column already exists as `text NOT NULL DEFAULT 'local'`. Current values are `local` (CLI-loaded data) and `model_catalog` (catalog sync). Repurpose it to identify the benchmark tool/method:

| `source` value | Meaning | Loaded by |
|----------------|---------|-----------|
| `guidellm` | Real hardware benchmarks from GuideLLM | `make db-load-guidellm` |
| `blis` | BLIS physics-based simulator | `make db-load-blis` |
| `llm-optimizer` | BentoML llm-optimizer roofline estimates | Estimated flow (runtime) |
| `manual` | Manually produced data (estimated or interpolated) | `make db-load-estimated`, `make db-load-interpolated` |
| `model_catalog` | Model catalog sync | `model_catalog_sync.py` (unchanged) |
| `llmd-benchmark` | llm-d benchmark tool (future) | Future loader |
| `other` | Unclassified / legacy | Default for migration |

**Migration**: Update existing data:
```sql
-- Migrate existing source values to new semantics
-- Existing 'local' data needs manual classification or a default
UPDATE exported_summaries SET source = 'other' WHERE source = 'local';
```

Note: The `model_catalog_sync.py` deletion logic (`DELETE WHERE source = 'model_catalog'`) continues to work unchanged. The roofline cleanup (`DELETE WHERE source = 'llm-optimizer'`) follows the same pattern.

**Update `loader.py`**: Accept `source` as a parameter instead of hardcoding `"local"`. The Makefile targets pass the appropriate value:
```
make db-load-blis       -->  source='blis'
make db-load-estimated  -->  source='manual'
make db-load-guidellm   -->  source='guidellm'
```

### 8.2 Add `confidence_level` Column

```sql
ALTER TABLE exported_summaries
    ADD COLUMN IF NOT EXISTS confidence_level text NOT NULL DEFAULT 'estimated';

COMMENT ON COLUMN exported_summaries.confidence_level IS
    'Trust level: benchmarked (real or validated benchmarks), estimated (analytical/llm-optimizer)';
```

| `confidence_level` | Meaning | UI Badge | Sources |
|--------------------|---------|----------|---------|
| `benchmarked` | Real or validated benchmarks | Green "Benchmarked" | `guidellm`, `llmd-benchmark`, `blis` |
| `estimated` | Analytical / roofline model | Amber "Estimated" | `llm-optimizer`, `manual` |

The mapping from `source` to `confidence_level` is set at load time by the loader or the estimated flow. It is not derived automatically.

### 8.3 Update `BenchmarkData` Class

**Location**: `src/planner/knowledge_base/benchmarks.py`

Add fields to `BenchmarkData.__init__()`:
```python
self.source = data.get("source", "other")
self.confidence_level = data.get("confidence_level", "estimated")
```

Add to `to_dict()`:
```python
"estimated": self.estimated,
"source": self.source,
"confidence_level": self.confidence_level,
```

Update `find_configurations_meeting_slo()` to include `source` and `confidence_level` in the SELECT columns.

### 8.4 Clearing Estimated Data

- **API endpoint**: `DELETE /api/v1/benchmarks/llm-optimizer` — runs `DELETE FROM exported_summaries WHERE source = 'llm-optimizer'`
- **UI**: "Clear Estimated Data" button in Configuration tab
- **CLI**: `make db-clear-estimated`
- **Optional TTL-based cleanup** (future): `DELETE WHERE source = 'llm-optimizer' AND created_at < NOW() - INTERVAL '7 days'`

---

## 9. LLM Extraction — Model Name Extraction

### 9.1 Extraction Prompt Update

**Location**: `src/planner/llm/prompts.py`

Add `preferred_models` to the extraction schema (follows the same pattern as `preferred_gpu_types`):

```json
"preferred_models": ["<list of model IDs if mentioned, empty list if not specified>"]
```

Add extraction examples to the prompt:
```
Model extraction examples (use HuggingFace format):
- "deploy Llama 3.3 70B" → preferred_models: ["meta-llama/Llama-3.3-70B-Instruct"]
- "I want to use Qwen3-32B or Mistral Small" → preferred_models: ["Qwen/Qwen3-32B", "mistralai/Mistral-Small-24B-Instruct-2501"]
- "compare granite and llama for my use case" → preferred_models: ["ibm-granite/granite-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct"]
- No model mentioned → preferred_models: []
```

The LLM should map common model names to their canonical HuggingFace repo IDs. The model catalog (`data/configuration/model_catalog.json`) provides the mapping — include a representative subset in the prompt for reference.

### 9.2 User Modification Flow

After LLM extraction, the user reviews the extracted `preferred_models` in the approval view. They can:
1. Accept the extracted models as-is.
2. Click "Modify Extracted Context" to adjust:
   - Multi-select from model catalog (pre-populated with catalog `model_id` values).
   - Free-text input for arbitrary HuggingFace model IDs not in the catalog.

---

## 10. Test Requirements (FR-11)

### 11.1 New Unit Tests Required

| Function | Test File | Test Cases |
|----------|-----------|------------|
| `check_model_fits_gpu()` | `tests/unit/test_capacity_planner.py` | Model fits at TP=1; model needs TP=2+; model too large for any TP; edge cases (0 memory, invalid config) |
| `_generate_estimated_configs()` | `tests/unit/test_config_finder.py` | No models to estimate; models partially covered by benchmarks; all models uncovered; GPU filter applied; HF API failure (mocked); roofline failure (mocked) |
| `_convert_estimation_to_benchmark()` | `tests/unit/test_config_finder.py` | Correct field mapping; all percentiles set to same value; E2E unit conversion (s→ms); requests_per_second calculation |
| Validation CLI | `tests/unit/test_validate_estimation.py` | Comparison logic; MAPE calculation; CSV output format (future work) |

### 11.2 Existing Functions — Verify Coverage

| Function | Test File | Status |
|----------|-----------|--------|
| `BenchmarkData.to_dict()` | `tests/unit/test_benchmarks.py` | Verify `estimated`, `source`, `confidence_level` fields are included |
| `DeploymentIntent` with `preferred_models` | `tests/unit/test_schemas.py` | Verify new field serialization/deserialization |
| `plan_all_capacities()` | `tests/unit/test_config_finder.py` | Add cases for `enable_estimated=True/False`, `preferred_models` parameter |
| LLM extraction prompt | `tests/unit/test_extraction.py` | Add cases verifying `preferred_models` extraction from natural language |

### 11.3 Integration Tests

| Scenario | Description |
|----------|-------------|
| End-to-end estimated flow | User specifies a model not in DB → estimated results returned with `estimated=True` |
| Mixed benchmark + estimated | Some (model, GPU) combos have benchmarks, others estimated → both appear in results |
| Estimated toggle disabled | `enable_estimated=False` → only benchmark results, no estimation attempted |
| DB cache hit | Second request for same (model, GPU, traffic) → prior roofline estimate returned from `exported_summaries`, no re-computation |

---

## 11. Implementation Sequence

### Phase 1: Backend Core
1. DB schema: repurpose `source` column, add `confidence_level` column to `exported_summaries`
2. Add `check_model_fits_gpu()` to `capacity_planner.py`
3. Add `preferred_models` to `DeploymentIntent`
4. Update LLM extraction prompt for model name extraction
5. Add `_generate_estimated_configs()` and `_convert_estimation_to_benchmark()` to `ConfigFinder`
6. Modify `plan_all_capacities()` to call estimated flow
7. Add `enable_estimated` parameter through API route chain
8. Add `source`, `confidence_level` to `BenchmarkData`
9. Write roofline estimates to `exported_summaries`; update `loader.py` to accept `source`/`confidence_level`
10. Add response `warnings` field for estimation failures
11. Unit tests for all new/modified functions

### Phase 2: UI Changes
1. Replace hardcoded GPU list with catalog-driven multi-select
2. Add model selection widgets (multi-select + free text)
3. Add estimated toggle, "Clear Estimated Data" button to Configuration tab
4. Pass new parameters through API client
5. Add "Benchmarked" and "Estimated" badges to recommendation cards
6. Show GPU/model constraints in Technical Specification tab
7. Display estimation warnings to user

### Phase 3: Validation and Polish (future — not part of this PR)
1. Build estimation accuracy validation CLI tool (see section 13.1)
2. Run initial accuracy baseline against existing benchmark data
3. Add `estimate_all_catalog` configuration option
4. Performance optimization (parallelization, configurable timeouts)
5. Confidence scoring based on validation results (see section 13.2)
6. "Clear Estimated Data" button in Configuration tab

---

## 12. Risks

| Risk | Mitigation |
|------|------------|
| Roofline estimates diverge significantly from real benchmarks | Clear "Estimated"/"Benchmarked" labeling; accuracy validation tool (future); confidence discount in scoring based on measured MAPE |
| HuggingFace rate limiting on model config fetches | `@lru_cache` already in place; roofline estimates persisted in DB; add retry with backoff |
| Response time increase for large matrices | Configurable cap on models/timeout (NFR-1); fast pre-filter via memory check; DB-persisted estimates avoid re-computation; consider `ThreadPoolExecutor` |
| Free-text model IDs with typos | Validate model exists on HF before estimation; report all failures to UI via `warnings` in response |
| LLM extraction maps model names incorrectly | Provide reference model list in extraction prompt; user can correct via "Modify Extracted Context" |
| Roofline estimate staleness | Estimates persisted in DB by `source='llm-optimizer'`; clearable via API/UI/CLI; optional TTL-based cleanup for future |

---

## 13. Future Work

### 13.1 Estimation Accuracy Validation

Provide a CLI tool to measure how well roofline estimates match real benchmark data:
- Compare estimated vs actual for TTFT, ITL, E2E, throughput across all benchmarked (model, GPU) combinations.
- Report MAPE, median error, max error, and systematic bias per metric.
- Location: `scripts/validate_estimation_accuracy.py` (or Makefile target `make validate-estimates`).
- Also consider exposing validation results in the UI.

### 13.2 Confidence Scoring

Use validation MAPE data to discount estimated scores (e.g., if MAPE for latency is 20%, apply a 20% penalty to the latency score for estimated configs). This would be a multiplier in `src/planner/recommendation/scorer.py`.

### 13.3 Other

- "Clear Estimated Data" button in Configuration tab (FR-9/8.4)
- `estimate_all_catalog` configuration option
- Performance optimization (parallelization, configurable timeouts)
- Integration tests for the estimated performance flow
- Pluggable estimation backend: abstract the estimation engine behind a common interface so different backends (llm-optimizer, BLIS, etc.) can be swapped without modifying orchestration logic

---

## Appendix A: Key Files

| File | Role |
|------|------|
| `src/planner/recommendation/config_finder.py` | Main insertion point (line 255) |
| `src/planner/capacity_planner.py` | Memory feasibility check (new wrapper function) |
| `src/planner/gpu_recommender.py` | Roofline estimation (used as-is) |
| `src/planner/knowledge_base/benchmarks.py` | `BenchmarkData` class |
| `src/planner/shared/schemas/intent.py` | `DeploymentIntent` schema |
| `src/planner/api/routes/recommendation.py` | API route for recommendations |
| `ui/components/extraction.py` | GPU/model selection UI |
| `ui/components/settings.py` | Estimated toggle |
| `ui/components/recommendations.py` | Estimated badge |
| `ui/api_client.py` | API client functions |
| `data/configuration/gpu_catalog.json` | GPU catalog (12 GPU types) |
| `data/configuration/model_catalog.json` | Model catalog (45+ models) |
| `src/planner/llm/prompts.py` | LLM extraction prompt (add `preferred_models`) |
| `scripts/schema.sql` | DB schema (add `confidence_level` column, update `source` values) |
| `src/planner/knowledge_base/loader.py` | Benchmark loader (accept `source`/`confidence_level` params) |
| `scripts/validate_estimation_accuracy.py` | Accuracy validation CLI tool (future — not yet implemented) |

