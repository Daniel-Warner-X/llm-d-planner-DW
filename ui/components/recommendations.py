"""Recommendation display components for the NeuralNav UI.

Category cards, top-5 table, options list, and recommendation results.
"""

import pandas as pd
import streamlit as st
from api_client import deploy_and_generate_yaml
from helpers import format_display_name, format_gpu_config, get_scores

# Tab indices for app.py st.tabs (0-based)
_RECOMMENDATION_TAB_INDEX = 2
_DEPLOYMENT_TAB_INDEX = 3


def _pop_viable_configs_table_state() -> None:
    """Clear legacy session keys if present (table no longer uses selectable dataframe widget)."""
    st.session_state.pop("viable_configs_table", None)
    st.session_state.pop("_viable_configs_last_selection_rows", None)


def _category_key_matches_selection(sel_cat: str, table_cat_key: str) -> bool:
    """Cards use short keys (accuracy, latency, cost); table uses API keys (best_accuracy, …)."""
    if sel_cat == table_cat_key:
        return True
    return {
        "accuracy": "best_accuracy",
        "latency": "lowest_latency",
        "cost": "lowest_cost",
    }.get(sel_cat) == table_cat_key


def _row_matches_selected_config(cat_key: str, model_display: str, gpu_str: str) -> bool:
    """True if this table row is the active deployment selection."""
    sel = st.session_state.get("deployment_selected_config")
    sel_cat = st.session_state.get("deployment_selected_category")
    if not sel or sel_cat is None:
        return False
    if not _category_key_matches_selection(sel_cat, cat_key):
        return False
    f_model = format_display_name(sel.get("model_name", "Unknown"))
    f_gpu = format_gpu_config(sel.get("gpu_config", {}) or {})
    return f_model == model_display and f_gpu == gpu_str


def _clear_deployment_after_card_nav() -> None:
    st.session_state.deployment_selected_config = None
    st.session_state.deployment_selected_category = None
    st.session_state.deployment_yaml_generated = False
    st.session_state.deployment_yaml_files = {}
    st.session_state.deployment_id = None
    st.session_state.deployment_error = None
    st.session_state.deployed_to_cluster = False
    _pop_viable_configs_table_state()


def _build_viable_configs_table_data(ranked_response: dict) -> list[dict]:
    """Build table rows in stable order (matches DataFrame rows)."""
    categories = [
        ("balanced", "Balanced"),
        ("best_accuracy", "Best Accuracy"),
        ("lowest_cost", "Lowest Cost"),
        ("lowest_latency", "Lowest Latency"),
    ]
    table_data: list[dict] = []
    for cat_key, cat_name in categories:
        recs = ranked_response.get(cat_key, [])
        for rec in recs[:5]:
            model_name = format_display_name(rec.get("model_name", "Unknown"))
            gpu_str = format_gpu_config(rec.get("gpu_config", {}) or {})
            ttft = rec.get("predicted_ttft_p95_ms", 0)
            cost = rec.get("cost_per_month_usd", 0)
            scores = rec.get("scores", {}) or {}
            accuracy = scores.get("accuracy_score", 0)
            balanced = scores.get("balanced_score", 0)
            meets_slo = rec.get("meets_slo", False)
            table_data.append(
                {
                    "cat_key": cat_key,
                    "rec": rec,
                    "category": cat_name,
                    "model": model_name,
                    "gpu_config": gpu_str,
                    "ttft": ttft,
                    "cost": cost,
                    "accuracy": accuracy,
                    "balanced": balanced,
                    "slo": "Yes" if meets_slo else "No",
                }
            )
    return table_data


def _viable_configs_display_dataframe(table_data: list[dict]) -> pd.DataFrame:
    """Read-only table; ✓ marks the row matching the active deployment (from cards)."""
    return pd.DataFrame(
        {
            "✓": [
                "✓" if _row_matches_selected_config(r["cat_key"], r["model"], r["gpu_config"]) else ""
                for r in table_data
            ],
            "Category": [r["category"] for r in table_data],
            "Model": [r["model"] for r in table_data],
            "GPU Config": [r["gpu_config"] for r in table_data],
            "TTFT (ms)": [r["ttft"] for r in table_data],
            "Cost/mo": [r["cost"] for r in table_data],
            "Acc": [r["accuracy"] for r in table_data],
            "Score": [r["balanced"] for r in table_data],
            "SLO": [r["slo"] for r in table_data],
        }
    )


def _render_view_deployment_config_button() -> None:
    """Centered primary button below recommendation cards; opens the Deployment tab."""
    has_selection = bool(
        st.session_state.get("deployment_selected_config")
        and st.session_state.get("deployment_selected_config", {}).get("model_name")
    )
    st.markdown('<div style="margin-top: 0.75rem;"></div>', unsafe_allow_html=True)
    c_l, c_mid, c_r = st.columns([1, 2, 1])
    with c_mid:
        if st.button(
            "View Deployment Config",
            key="model_rec_goto_deployment",
            type="primary",
            disabled=not has_selection,
            use_container_width=True,
        ):
            st.session_state["_pending_tab"] = _DEPLOYMENT_TAB_INDEX
            st.rerun()


def _render_model_recommendation_header() -> None:
    """Model Recommendation title, SLO stats, and selected model summary."""
    st.markdown(
        '<h3 style="font-weight: 600; font-size: 1.375rem; margin: 0 0 0.35rem 0; padding: 0; '
        'line-height: 1.2; color: var(--text-color, #31333F);">Model Recommendation</h3>',
        unsafe_allow_html=True,
    )

    ranked_response = st.session_state.get("ranked_response", {})
    total_configs = ranked_response.get("total_configs_evaluated", 0)
    passed_configs = ranked_response.get("configs_after_filters", 0)
    selected_cfg = st.session_state.get("deployment_selected_config")

    selected_html = ""
    if selected_cfg and selected_cfg.get("model_name"):
        selected_model = format_display_name(selected_cfg["model_name"])
        selected_html = (
            '<span style="font-size: 0.85rem; white-space: nowrap;">'
            '<span style="color: #10B981;">✅</span> Selected: '
            f"<strong>{selected_model}</strong>"
            "</span>"
        )

    stats_left = ""
    bottom_line = ""
    if total_configs > 0:
        all_passed = []
        for cat in ["balanced", "best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
            all_passed.extend(ranked_response.get(cat, []))
        unique_models = len({r.get("model_name", "") for r in all_passed if r.get("model_name")})
        filter_pct = passed_configs / total_configs * 100
        stats_left = (
            f'<span style="font-size: 0.85rem;"><strong style="color: #10B981;">{passed_configs:,}</strong> '
            f"configs passed SLO filter from <strong>{total_configs:,}</strong> total "
            f"<span>({filter_pct:.0f}% match)</span></span>"
        )
        bottom_line = (
            f'<span style="font-size: 0.85rem;"><strong>{unique_models}</strong> unique models</span>'
        )

    col_stats, col_actions = st.columns([3, 1])
    with col_stats:
        # Keep stats + unique models in this column so a tall right column would not
        # push the second line down (full-width row below columns would sit under it).
        if stats_left or bottom_line:
            parts = []
            if stats_left:
                parts.append(f'<div style="margin: 0; padding: 0;">{stats_left}</div>')
            if bottom_line:
                parts.append(
                    f'<div style="margin: 0.35rem 0 0 0; padding: 0;">{bottom_line}</div>'
                )
            st.markdown(
                '<div class="nn-filter-summary" style="margin: 0 0 1rem 0; padding: 0;">'
                + "".join(parts)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="margin: 0 0 1rem 0;"></div>',
                unsafe_allow_html=True,
            )
    with col_actions:
        if selected_html:
            st.markdown(
                '<div style="display: flex; justify-content: flex-end; align-items: center; '
                'min-height: 2.25rem;">'
                f"{selected_html}</div>",
                unsafe_allow_html=True,
            )


def _render_category_card(
    title,
    recs_list,
    highlight_field,
    category_key,
    col,
    *,
    show_recommended_badge: bool = False,
):
    """Render a recommendation card for a category with prev/next navigation.

    If show_recommended_badge is True, the badge appears only while index 0 is shown.
    """
    if not recs_list:
        return

    idx_key = f"cat_idx_{category_key}"
    idx = st.session_state.get(idx_key, 0)
    idx = min(idx, len(recs_list) - 1)  # Clamp if list shrank

    rec = recs_list[idx]
    scores = get_scores(rec)
    model_name = format_display_name(rec.get("model_name", "Unknown"))
    gpu_cfg = rec.get("gpu_config", {}) or {}
    hw_type = gpu_cfg.get("gpu_type", rec.get("hardware", "H100"))
    hw_count = gpu_cfg.get("gpu_count", rec.get("hardware_count", 1))
    replicas = gpu_cfg.get("replicas", 1)
    cost = rec.get("cost_per_month_usd", 0)

    # Performance metrics (P95)
    ttft = rec.get("predicted_ttft_p95_ms") or 0
    itl = rec.get("predicted_itl_p95_ms") or 0
    e2e = rec.get("predicted_e2e_p95_ms") or 0
    throughput = rec.get("predicted_throughput_qps") or 0

    # Build scores line with highlight on the matching category
    score_items = [
        ("accuracy", "Accuracy", scores["accuracy"]),
        ("cost", "Cost", scores["cost"]),
        ("latency", "Latency", scores["latency"]),
        ("final", "Balance", scores["final"]),
    ]
    score_parts = []
    for field, label, value in score_items:
        if field == highlight_field:
            score_parts.append(
                f'<span style="color: #1f77b4; font-weight: 700;">{label}: {value:.0f}</span>'
            )
        else:
            score_parts.append(f"{label}: {value:.0f}")
    scores_line = " | ".join(score_parts)

    # Build metrics line
    metrics_line = (
        f"TTFT: {ttft:,}ms | ITL: {itl}ms | E2E: {e2e:,}ms | "
        f"Throughput: {throughput:.1f} rps | Cost: ${cost:,.0f}/mo"
    )

    # Recommended badge only for the first ranked model (index 0); hide after prev/next arrows.
    if show_recommended_badge and idx == 0:
        title_html = (
            '<div style="display: flex; align-items: center; justify-content: space-between; '
            'gap: 0.75rem; flex-wrap: wrap; line-height: 1.7;">'
            f'<strong style="font-size: 1.05rem;">{title}</strong>'
            '<span style="display: inline-block; padding: 0.18rem 0.55rem; font-size: 0.7rem; '
            "font-weight: 600; letter-spacing: 0.03em; color: #047857; "
            "background: rgba(16, 185, 129, 0.14); border: 1px solid rgba(16, 185, 129, 0.35); "
            'border-radius: 999px; white-space: nowrap;">Recommended</span>'
            "</div>"
        )
    else:
        title_html = (
            f'<div style="line-height: 1.7;"><strong style="font-size: 1.05rem;">{title}</strong></div>'
        )

    with col, st.container(border=True):
        # Single st.columns row (no nesting) — Streamlit allows at most one column level inside a column
        if len(recs_list) > 1:
            last = len(recs_list) - 1
            # One st.columns row only (no nesting). Buttons avoid <a href="?…"> tab breaks / reloads.
            c_title, c_sp, c_prev, c_lab, c_next = st.columns(
                [0.3, 0.28, 0.07, 0.12, 0.07], vertical_alignment="center"
            )
            with c_title:
                st.markdown(title_html, unsafe_allow_html=True)
            with c_sp:
                st.empty()
            with c_prev:
                if st.button("‹", key=f"prev_{category_key}"):
                    st.session_state[idx_key] = last if idx == 0 else idx - 1
                    _clear_deployment_after_card_nav()
                    st.session_state["_pending_tab"] = _RECOMMENDATION_TAB_INDEX
                    st.rerun()
            with c_lab:
                st.markdown(
                    f"<div style='text-align: center; font-size: 0.85rem; line-height: 1.25; padding: 0; margin: 0; white-space: nowrap; color: inherit;'>#{idx + 1} of {len(recs_list)}</div>",
                    unsafe_allow_html=True,
                )
            with c_next:
                if st.button("›", key=f"next_{category_key}"):
                    st.session_state[idx_key] = 0 if idx == last else idx + 1
                    _clear_deployment_after_card_nav()
                    st.session_state["_pending_tab"] = _RECOMMENDATION_TAB_INDEX
                    st.rerun()
        else:
            st.markdown(title_html, unsafe_allow_html=True)

        st.markdown(
            f'<div style="line-height: 1.7;">'
            f'<span style="font-size: 0.9rem;"><strong>Solution:</strong> Model: {model_name} | Hardware: {hw_count}x {hw_type} | Replicas: {replicas}</span><br>'
            f'<span style="font-size: 0.9rem;"><strong>Scores:</strong> {scores_line}</span><br>'
            f'<span style="font-size: 0.9rem;"><strong>Values:</strong> {metrics_line}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        selected_category = st.session_state.get("deployment_selected_category")
        is_selected = selected_category == category_key

        if is_selected:
            if st.button(
                "✅ Selected",
                key=f"selected_{category_key}",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.deployment_selected_config = None
                st.session_state.deployment_selected_category = None
                st.session_state.deployment_yaml_generated = False
                st.session_state.deployment_yaml_files = {}
                st.session_state.deployment_id = None
                st.session_state.deployment_error = None
                st.session_state.deployed_to_cluster = False
                _pop_viable_configs_table_state()
                st.rerun()
        else:
            if st.button("Select", key=f"select_{category_key}", use_container_width=True):
                st.session_state.deployment_selected_config = rec
                st.session_state.deployment_selected_category = category_key
                st.session_state.deployment_yaml_generated = False
                st.session_state.deployment_yaml_files = {}
                st.session_state.deployment_id = None
                st.session_state.deployed_to_cluster = False
                _pop_viable_configs_table_state()

                result = deploy_and_generate_yaml(rec)
                if result and result.get("success"):
                    st.session_state.deployment_id = result["deployment_id"]
                    st.session_state.deployment_yaml_files = result["files"]
                    st.session_state.deployment_yaml_generated = True
                else:
                    st.session_state.deployment_yaml_generated = False
                st.rerun()


def render_top5_table(recommendations: list, priority: str):
    """Render Top 5 recommendation cards with filtering summary.

    Uses the backend's pre-ranked lists (ACCURACY-FIRST strategy in analyzer.py).
    """
    _render_model_recommendation_header()

    use_case = st.session_state.get("detected_use_case", "chatbot_conversational")

    if not recommendations:
        st.info("No models available. Please check your requirements.")
        return

    ranked_response = st.session_state.get("ranked_response", {})
    top5_balanced = ranked_response.get("balanced", [])[:5]
    top5_accuracy = ranked_response.get("best_accuracy", [])[:5]
    top5_latency = ranked_response.get("lowest_latency", [])[:5]
    top5_cost = ranked_response.get("lowest_cost", [])[:5]

    st.session_state.top5_balanced = top5_balanced
    st.session_state.top5_accuracy = top5_accuracy
    st.session_state.top5_latency = top5_latency
    st.session_state.top5_cost = top5_cost
    st.session_state.top5_simplest = ranked_response.get("simplest", [])[:5]

    # Render 4 category cards in a 2x2 grid
    col1, col2 = st.columns(2)
    _render_category_card(
        "Balance", top5_balanced, "final", "balanced", col1, show_recommended_badge=True
    )
    _render_category_card("Accuracy", top5_accuracy, "accuracy", "accuracy", col2)

    col3, col4 = st.columns(2)
    _render_category_card("Latency", top5_latency, "latency", "latency", col3)
    _render_category_card("Cost", top5_cost, "cost", "cost", col4)

    _render_view_deployment_config_button()

    total_available = len(recommendations)
    if total_available <= 2:
        use_case_display = use_case.replace("_", " ").title() if use_case else "this task"
        st.info(f"Only {total_available} model(s) have benchmarks for {use_case_display}")


def render_options_list_inline():
    """Render sortable table of ranked deployment configurations."""
    ranked_response = st.session_state.get("ranked_response")

    if not ranked_response:
        st.warning("No recommendations available. Please run the recommendation process first.")
        return

    total_configs = ranked_response.get("total_configs_evaluated", 0)
    configs_after_filters = ranked_response.get("configs_after_filters", 0)

    st.markdown(
        f"""<div style="margin: 1rem 0 1rem 0; padding: 0; text-align: left;">
<h3 style="font-weight: 600; font-size: 1.375rem; margin: 0; padding: 0; line-height: 1.2; color: var(--text-color, #31333F);">All viable deployment configurations</h3>
<p style="margin: 0.35rem 0 0 0; padding: 0; font-size: 0.9rem; line-height: 1.7; color: rgba(49, 51, 63, 0.62); text-align: left;">Evaluated <span style="font-weight: 600;">{total_configs}</span> viable configurations, showing <span style="font-weight: 600;">{configs_after_filters}</span> unique options</p>
</div>""",
        unsafe_allow_html=True,
    )

    table_data = _build_viable_configs_table_data(ranked_response)

    if table_data:
        sig = (
            ranked_response.get("total_configs_evaluated"),
            ranked_response.get("configs_after_filters"),
            len(table_data),
        )
        if st.session_state.get("_viable_configs_table_sig") != sig:
            st.session_state._viable_configs_table_sig = sig
            _pop_viable_configs_table_state()

        df = _viable_configs_display_dataframe(table_data)
        st.caption(
            "✓ marks the active deployment. Select a configuration using the recommendation cards above."
        )
        # No on_select / selection_mode — those add Streamlit's row checkbox column, which cannot
        # stay in sync with card selection; the ✓ column reflects the current deployment instead.
        st.dataframe(
            df,
            column_config={
                "✓": st.column_config.TextColumn("✓", width=None),
                "TTFT (ms)": st.column_config.NumberColumn("TTFT (ms)", format="%.0f"),
                "Cost/mo": st.column_config.NumberColumn("Cost/mo", format="%.0f"),
                "Acc": st.column_config.NumberColumn("Acc", format="%.0f"),
                "Score": st.column_config.NumberColumn("Score", format="%.1f"),
            },
            hide_index=True,
            use_container_width=True,
            height=400,
        )
    else:
        st.warning("No configurations to display")


def render_recommendation_result(result: dict, priority: str, extraction: dict):
    """Render recommendation results with Top 5 table."""

    if extraction is None:
        extraction = {}

    ranked_response = result

    if ranked_response:
        st.session_state.ranked_response = ranked_response

        balanced_recs = ranked_response.get("balanced", [])
        if balanced_recs:
            winner = balanced_recs[0]
            recommendations = balanced_recs
        else:
            for cat in ["best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
                if ranked_response.get(cat):
                    winner = ranked_response[cat][0]
                    recommendations = ranked_response[cat]
                    break
            else:
                st.warning("No recommendations found.")
                return
    else:
        st.warning(
            "Could not fetch ranked recommendations from backend. Ensure the backend is running."
        )
        st.session_state.ranked_response = None
        recommendations = result.get("recommendations", [])
        if not recommendations:
            st.warning("No recommendations found. Try adjusting your requirements.")
            return
        winner = recommendations[0]

    st.session_state.winner_recommendation = winner
    st.session_state.winner_priority = priority
    st.session_state.winner_extraction = extraction

    # Get all recommendations for the cards
    all_recs = []
    for cat in ["balanced", "best_accuracy", "lowest_cost", "lowest_latency", "simplest"]:
        cat_recs = (
            st.session_state.ranked_response.get(cat, [])
            if st.session_state.ranked_response
            else []
        )
        all_recs.extend(cat_recs)

    # Remove duplicates by model+hardware
    seen = set()
    unique_recs = []
    for rec in all_recs:
        model = rec.get("model_name", "")
        gpu_cfg = rec.get("gpu_config", {}) or {}
        hw = f"{gpu_cfg.get('gpu_type', 'H100')}x{gpu_cfg.get('gpu_count', 1)}"
        key = f"{model}_{hw}"
        if key not in seen:
            seen.add(key)
            unique_recs.append(rec)

    if unique_recs:
        render_top5_table(unique_recs, priority)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    render_options_list_inline()
