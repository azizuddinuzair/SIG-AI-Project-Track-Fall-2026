"""Streamlit app for Pokemon team generation and analysis.

Run:
    py -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
import streamlit as st

# Ensure project imports resolve when launched via streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ga import PokemonGA, load_pokemon_data
from src.ga.config import get_config_a, get_config_b, get_config_c, get_config_random
from src.ga.fitness import TYPE_NAMES
from src.ga.job_queue import get_ga_job_queue
from src.ga.job_runner import build_ga_job_request, run_ga_job
from src.team_store import TeamStore

from legacy.scripts.cli import analyze_team_by_names


st.set_page_config(page_title="Pokemon Team Optimizer", layout="wide")

PIVOT_CANDIDATE_THRESHOLD = 0.62
ABILITY_DATA_PATH = PROJECT_ROOT / "data" / "pokemon_abilities.csv"
SESSION_ID_KEY = "team_session_id"
TEAM_STORE_KEY = "team_session_store"
ACTIVE_JOB_ID_KEY = "active_ga_job_id"
LAST_RESULT_KEY = "last_ga_result"


def _get_session_id() -> str:
    session_id = st.session_state.get(SESSION_ID_KEY)
    if not session_id:
        session_id = uuid4().hex
        st.session_state[SESSION_ID_KEY] = session_id
    return session_id


def _get_team_store() -> TeamStore:
    store = st.session_state.get(TEAM_STORE_KEY)
    if store is None:
        store = TeamStore()
        st.session_state[TEAM_STORE_KEY] = store
    return store


def _submit_ga_job(
    *,
    config_name: str,
    population: int,
    generations: int,
    seed: int,
    top_n: int,
    locked_names: list[str],
    composition_name: str = "balanced",
    composition_weight: float = 0.20,
    power_mode: str = "standard",
) -> tuple[str | None, str | None]:
    request = build_ga_job_request(
        config_name=config_name,
        population=population,
        generations=generations,
        seed=seed,
        top_n=top_n,
        locked_names=locked_names,
        composition_name=composition_name,
        composition_weight=composition_weight,
        power_mode=power_mode,
    )
    return get_ga_job_queue().submit(request)


def _get_active_job_record():
    job_id = st.session_state.get(ACTIVE_JOB_ID_KEY)
    if not job_id:
        return None
    return get_ga_job_queue().get_job(job_id)


def _is_generation_in_progress() -> bool:
    record = _get_active_job_record()
    if record is None:
        return False
    return record.status in {"queued", "running"}


def _poll_ga_job_status() -> None:
    job_id = st.session_state.get(ACTIVE_JOB_ID_KEY)
    if not job_id:
        return

    record = get_ga_job_queue().get_job(job_id)
    if record is None:
        st.session_state.pop(ACTIVE_JOB_ID_KEY, None)
        return

    if record.status in {"queued", "running"}:
        with st.status("Generating team...", state="running", expanded=True):
            st.write(f"Current job status: **{record.status}**")
            st.write(f"Job ID: `{record.job_id}`")
            st.caption("You can keep using other app sections while this runs.")
            if st.button("Refresh Job Status", key=f"refresh_job_{record.job_id}"):
                st.rerun()
        return

    if record.status == "completed" and record.result is not None:
        st.session_state[LAST_RESULT_KEY] = record.result
        st.session_state.pop(ACTIVE_JOB_ID_KEY, None)
        st.success("Your GA job completed.")
        st.caption(f"Job ID: {record.job_id}")
        return

    if record.status == "failed":
        st.session_state.pop(ACTIVE_JOB_ID_KEY, None)
        st.error("GA job failed.")
        if record.error:
            st.caption(record.error)
        return


def _render_latest_ga_result(data_df: pd.DataFrame) -> None:
    result = st.session_state.get(LAST_RESULT_KEY)
    if not result:
        return

    with st.expander("Latest GA Result", expanded=False):
        _render_ga_results(result, data_df, include_analysis=True)


def _render_job_output_section(data_df: pd.DataFrame) -> None:
    st.markdown("---")
    st.markdown("### Job Status & Latest Result")
    _poll_ga_job_status()
    _render_latest_ga_result(data_df)


def _inject_theme() -> None:
        st.markdown(
                """
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

                    :root {
                        --bg-1: #0F172A;
                        --bg-2: #111B31;
                        --card: #1E293B;
                        --ink: #E2E8F0;
                        --muted: #94A3B8;
                        --muted-dim: #64748B;
                        --accent: #3B82F6;
                        --accent-2: #14B8A6;
                        --highlight: #F59E0B;
                        --ok: #22C55E;
                        --warn: #F97316;
                        --err: #EF4444;
                        --line: #334155;
                    }

                    .stApp {
                        background: radial-gradient(circle at 12% 8%, #17233d, var(--bg-1) 48%, #0b1220 100%);
                        color: var(--ink);
                        font-family: 'IBM Plex Sans', sans-serif;
                    }

                    h1, h2, h3 {
                        font-family: 'Space Grotesk', sans-serif;
                        letter-spacing: -0.02em;
                        color: var(--ink);
                    }

                    p, li, label, .stMarkdown, .stCaption, .stText {
                        color: var(--ink);
                    }

                    [data-testid="stSidebar"] {
                        background: linear-gradient(180deg, #131d34 0%, #0f172a 100%);
                        border-right: 1px solid var(--line);
                    }

                    [data-testid="stSidebar"] * {
                        color: var(--ink);
                    }

                    .hero {
                        background: linear-gradient(125deg, #1E293B 0%, #1A355A 55%, #17456B 100%);
                        border-radius: 14px;
                        padding: 1.1rem 1.25rem;
                        color: var(--ink);
                        margin-bottom: 0.9rem;
                        border: 1px solid var(--line);
                        box-shadow: 0 10px 30px rgba(3, 8, 20, 0.45);
                        animation: fadeSlide 420ms ease-out;
                    }

                    .subtle-card {
                        background: var(--card);
                        border: 1px solid var(--line);
                        border-radius: 12px;
                        padding: 0.85rem 0.95rem;
                        box-shadow: 0 4px 14px rgba(9, 24, 20, 0.07);
                        animation: fadeSlide 480ms ease-out;
                    }

                    .stDataFrame, .stTable {
                        border: 1px solid var(--line);
                        border-radius: 10px;
                        overflow: hidden;
                        background: var(--card);
                    }

                    .stButton > button {
                        border-radius: 10px;
                        background: linear-gradient(120deg, var(--accent), #2563eb);
                        color: var(--ink);
                        border: 1px solid #60a5fa;
                    }

                    .stButton > button:hover {
                        background: linear-gradient(120deg, #2563eb, var(--accent-2));
                        border: 1px solid #67e8f9;
                    }

                    div[data-baseweb="select"] > div,
                    div[data-baseweb="input"] > div,
                    .stNumberInput input,
                    .stTextInput input,
                    .stTextArea textarea {
                        background: #0f1b33;
                        color: var(--ink);
                        border: 1px solid var(--line);
                    }

                    .stSlider [data-baseweb="slider"] * {
                        color: var(--ink);
                    }

                    .stAlert {
                        border-radius: 10px;
                    }

                    .stSuccess {
                        border: 1px solid var(--ok);
                    }

                    .stWarning {
                        border: 1px solid var(--warn);
                    }

                    .stError {
                        border: 1px solid var(--err);
                    }

                    @keyframes fadeSlide {
                        from {
                            opacity: 0;
                            transform: translateY(8px);
                        }
                        to {
                            opacity: 1;
                            transform: translateY(0);
                        }
                    }
                </style>
                """,
                unsafe_allow_html=True,
        )


def _build_error_log(operation: str, exc: Exception) -> tuple[str, bytes]:
    """Build error log content in memory; return (filename, bytes) for download."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    details = "\n".join([
        f"Operation: {operation}",
        f"Timestamp: {ts}",
        f"ExceptionType: {type(exc).__name__}",
        f"Message: {exc}",
        "",
        "Traceback:",
        traceback.format_exc(),
    ])
    filename = f"streamlit_{operation}_{ts}.log"
    return filename, details.encode("utf-8")


def _run_safe(operation: str, fn: Callable[[], Any]) -> tuple[bool, Any, str | None, tuple[str, bytes] | None]:
    try:
        result = fn()
        return True, result, None, None
    except Exception as exc:  # noqa: BLE001
        log_payload = _build_error_log(operation, exc)
        return False, None, str(exc), log_payload


def _friendly_failure(operation: str, message: str | None, log_payload: tuple[str, bytes] | None) -> None:
    st.error(f"{operation} failed.")
    if message:
        st.caption(f"Reason: {message}")
    if log_payload:
        filename, content = log_payload
        st.download_button(
            label="Download Error Log",
            data=content,
            file_name=filename,
            mime="text/plain",
            key=f"dl_errlog_{filename}",
        )


def _config_from_name(name: str) -> dict[str, Any]:
    options = {
        "A": get_config_a,
        "B": get_config_b,
        "C": get_config_c,
        "Random": get_config_random,
    }
    return copy.deepcopy(options[name]())


def _build_composition_presets() -> dict[str, dict[str, int]]:
    return {
        "balanced": {
            "Balanced All-Rounder": 2,
            "Generalist": 2,
            "Fast Attacker": 1,
            "Defensive Tank": 1,
        },
        "hyper_offense": {
            "Speed Sweeper": 2,
            "Fast Attacker": 2,
            "Physical Attacker": 1,
            "Generalist": 1,
        },
        "pivot_pressure": {
            "Generalist": 3,
            "Balanced All-Rounder": 1,
            "Fast Attacker": 1,
            "Defensive Tank": 1,
        },
        "bulky_offense": {
            "Defensive Tank": 2,
            "Balanced All-Rounder": 2,
            "Fast Attacker": 1,
            "Physical Attacker": 1,
        },
    }


def _normalize_composition_target(raw: dict[str, int], data_df: pd.DataFrame) -> dict[str, int]:
    available = set(data_df["archetype"].dropna().astype(str).unique().tolist())
    return {k: int(v) for k, v in raw.items() if int(v) > 0 and k in available}


def _power_mode_config(mode: str) -> dict[str, float]:
    mapping = {
        "standard": {"bst_cap": 3300, "bst_penalty_weight": 2.0},
        "competitive_strict": {"bst_cap": 3200, "bst_penalty_weight": 3.0},
        "open": {"bst_cap": 0, "bst_penalty_weight": 0.0},
    }
    return mapping[mode]


def _serialize_team(team_df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        "name",
        "archetype",
        "type1",
        "type2",
        "hp",
        "attack",
        "defense",
        "special-attack",
        "special-defense",
        "speed",
    ]
    safe_cols = [c for c in cols if c in team_df.columns]
    return team_df[safe_cols].to_dict("records")


def _team_table(team_records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in team_records:
        type2 = p.get("type2")
        type_label = p.get("type1", "") if not type2 else f"{p.get('type1', '')}/{type2}"
        bst = 0
        for col in ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]:
            bst += int(p.get(col, 0) or 0)
        rows.append(
            {
                "Pokemon": p.get("name", ""),
                "Type": type_label,
                "Archetype": p.get("archetype", "Unknown"),
                "BST": bst,
                "Speed": int(p.get("speed", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def _format_breakdown(breakdown: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, v in breakdown.items():
        if isinstance(v, (float, int)):
            rows.append({"Metric": k, "Value": float(v)})
    rows.sort(key=lambda x: x["Metric"])
    return pd.DataFrame(rows)


def _render_analysis_panel(team_names: list[str], data_df: pd.DataFrame, key_suffix: str = "") -> None:
    report = analyze_team_by_names(team_names, data_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Score", f"{report['overall_score']}/100")
    c2.metric("Type Score", f"{report['type']['score']}/100")
    c3.metric("Role Diversity", f"{report['roles']['diversity_score']}/100")

    st.markdown("### Fitness Snapshot")
    left, right = st.columns([1, 1])
    with left:
        weak_to = report["type"]["weak_to"]
        st.write("**Shared Weaknesses**")
        st.write(", ".join(weak_to) if weak_to else "No high-overlap weaknesses detected.")
        st.write("**Resistances**")
        resists = report["type"]["resistances"]
        st.write(", ".join(resists[:10]) if resists else "No standout resistance stack.")
    with right:
        st.write("**Role Distribution**")
        st.dataframe(
            pd.DataFrame(
                [{"Role": k, "Count": v} for k, v in report["roles"]["distribution"].items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.write("**Speed Tiers**")
        speed_tiers = report["advanced"]["speed_tiers"]
        st.caption(
            f"120+: {speed_tiers['120+']} | 100-119: {speed_tiers['100-119']} | "
            f"80-99: {speed_tiers['80-99']} | <80: {speed_tiers['<80']}"
        )

    st.markdown("### Team Issues")
    if report["issues"]:
        st.dataframe(pd.DataFrame(report["issues"]), use_container_width=True, hide_index=True)
    else:
        st.success("No major team issues were detected.")

    st.markdown("### Recommendations")
    for rec in report["recommendations"]:
        st.write(f"- {rec}")

    pivot_candidates = report["advanced"].get("pivot_candidates", [])
    if pivot_candidates:
        st.markdown("### Pivot Candidates")
        st.caption(
            f"Threshold {PIVOT_CANDIDATE_THRESHOLD:.2f} (bulk + speed + pressure + defensive utility)."
        )
        st.dataframe(pd.DataFrame(pivot_candidates), use_container_width=True, hide_index=True)

    report_json = json.dumps(report, indent=2, default=str).encode("utf-8")
    st.download_button(
        label="Download Analysis Report (JSON)",
        data=report_json,
        file_name="team_analysis.json",
        mime="application/json",
        key=f"dl_analysis{key_suffix}",
        use_container_width=True,
    )


def _run_ga_workflow(
    config_name: str,
    population: int,
    generations: int,
    seed: int,
    top_n: int,
    locked_names: list[str],
    composition_name: str = "balanced",
    composition_weight: float = 0.20,
    power_mode: str = "standard",
) -> dict[str, Any]:
    from src.ga.job_runner import run_ga_job

    request = build_ga_job_request(
        config_name=config_name,
        population=population,
        generations=generations,
        seed=seed,
        top_n=top_n,
        locked_names=locked_names,
        composition_name=composition_name,
        composition_weight=composition_weight,
        power_mode=power_mode,
    )
    return run_ga_job(request)


def _render_ga_results(results: dict[str, Any], data_df: pd.DataFrame, include_analysis: bool = True) -> None:
    st.success("GA run completed successfully.")
    if results["best_fitness"] is not None:
        st.metric("Best Fitness", f"{results['best_fitness']:.4f}")

    run_context = results.get("run_context", {})
    session_id = _get_session_id()
    team_store = _get_team_store()

    history_df = results["history"]
    if not history_df.empty:
        plot_cols = [c for c in ["max_fitness", "mean_fitness", "min_fitness"] if c in history_df.columns]
        if plot_cols:
            st.subheader("Fitness History")
            st.line_chart(history_df[plot_cols])

    st.subheader("Generated Team")
    for team in results["top_teams"]:
        save_key_prefix = f"{run_context.get('run_id', 'run')}_{team['rank']}"
        with st.expander(f"Rank {team['rank']} | Fitness {team['fitness']:.4f}", expanded=team["rank"] == 1):
            team_table_df = _team_table(team["pokemon"])
            st.dataframe(team_table_df, use_container_width=True, hide_index=True)
            breakdown_df = _format_breakdown(team["breakdown"])
            if not breakdown_df.empty:
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            team_csv = team_table_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Team as CSV",
                data=team_csv,
                file_name=f"team_rank_{team['rank']}.csv",
                mime="text/csv",
                key=f"dl_team_{team['rank']}",
                use_container_width=True,
            )

            nickname_key = f"team_nickname_{save_key_prefix}"
            default_nickname = st.session_state.get(
                nickname_key,
                team["pokemon"][0]["name"] if team["pokemon"] else f"Team {team['rank']}",
            )
            nickname = st.text_input(
                "Nickname",
                value=str(default_nickname),
                key=nickname_key,
                help="Give this team a name before saving it to the session database.",
            )
            if st.button("Save Team", key=f"save_team_{save_key_prefix}", use_container_width=True):
                nickname_clean = nickname.strip()
                if not nickname_clean:
                    st.warning("Please enter a nickname before saving.")
                else:
                    team_id = team_store.save_team(
                        session_id=session_id,
                        nickname=nickname_clean,
                        team_payload=team,
                        metadata={**run_context, "session_id": session_id},
                    )
                    st.success(f"Saved as '{nickname_clean}'.")
                    st.caption(f"Team ID: {team_id}")
                    st.caption("Open View Generated Teams from the sidebar to revisit saved teams in this session.")

            if include_analysis:
                st.markdown("### Team Analyzer")
                team_names = [p["name"] for p in team["pokemon"] if "name" in p]
                _render_analysis_panel(team_names, data_df, key_suffix=f"_rank{team['rank']}")

    with st.expander("Run Log"):
        st.code(results["run_log"] or "(no output)", language="text")


def _resolve_team_from_manual_input(input_names: list[str], data_df: pd.DataFrame) -> list[str]:
    available = set(data_df["name"].astype(str).tolist())
    resolved = []
    for name in input_names:
        if name in available:
            resolved.append(name)
    return resolved


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _get_stat_columns(df: pd.DataFrame) -> dict[str, str]:
    return {
        "hp": "hp",
        "attack": "attack",
        "defense": "defense",
        "sp_attack": "sp_attack" if "sp_attack" in df.columns else "special-attack",
        "sp_defense": "sp_defense" if "sp_defense" in df.columns else "special-defense",
        "speed": "speed",
    }


def _load_ability_lookup() -> dict[str, list[str]]:
    if not ABILITY_DATA_PATH.exists():
        return {}
    try:
        ability_df = pd.read_csv(ABILITY_DATA_PATH)
    except Exception:  # noqa: BLE001
        return {}

    if "name" not in ability_df.columns:
        return {}

    ability_cols = [col for col in ability_df.columns if "ability" in col.lower()]
    lookup: dict[str, list[str]] = {}
    for _, row in ability_df.iterrows():
        key = _normalize_name(row.get("name", ""))
        if not key:
            continue
        values: list[str] = []
        for col in ability_cols:
            raw = str(row.get(col, "")).strip()
            if not raw or raw.lower() == "nan":
                continue
            clean = raw.lower().replace("-", " ")
            if clean not in values:
                values.append(clean)
        lookup[key] = values
    return lookup


def _estimate_ability_bonus_from_names(abilities: list[str] | None) -> float:
    if not abilities:
        return 0.0
    ability_text = " ".join(abilities).lower()
    bonus = 0.0
    if "regenerator" in ability_text:
        bonus += 0.15
    if "intimidate" in ability_text:
        bonus += 0.12
    if "poison heal" in ability_text:
        bonus += 0.10
    if "natural cure" in ability_text:
        bonus += 0.08
    if "storm drain" in ability_text:
        bonus += 0.05
    if "water absorb" in ability_text:
        bonus += 0.04
    if "volt absorb" in ability_text:
        bonus += 0.04
    if "sap sipper" in ability_text:
        bonus += 0.04
    if "flash fire" in ability_text:
        bonus += 0.04
    if "levitate" in ability_text:
        bonus += 0.05
    if "magic guard" in ability_text:
        bonus += 0.05
    return min(0.18, bonus)


def _guide_section(title: str, text: str) -> None:
    st.markdown(
        f"<div style='color: var(--ink);'><strong>{title}</strong><br>{text}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)


def _render_pokemon_info_mode(data_df: pd.DataFrame) -> None:
    st.markdown("### Pokemon Info")
    st.caption("Inspect one Pokemon's typing, base stats, archetype, and pivot profile.")
    with st.expander("Quick Guide", expanded=False):
        _guide_section(
            "Pokemon Selector",
            "Pick a Pokemon by name to view its profile. This is useful for checking whether a single Pokemon is bulky, fast, or pivot-friendly.",
        )
        _guide_section(
            "Pivot Profile",
            "Pivot score combines bulk, speed, offensive pressure, defensive utility, and ability bonuses. Scores at or above the threshold are pivot candidates.",
        )

    names = sorted(data_df["name"].astype(str).tolist())
    selected_name = st.selectbox(
        "Select Pokemon",
        options=names,
        index=0,
        help="Choose one Pokemon to inspect detailed profile information.",
    )

    if not selected_name:
        return

    row = data_df[data_df["name"] == selected_name].iloc[0]
    stat_cols = _get_stat_columns(data_df)
    ability_lookup = _load_ability_lookup()
    selected_key = _normalize_name(selected_name)
    base_key = selected_key.split("-", 1)[0]
    abilities = ability_lookup.get(selected_key) or ability_lookup.get(base_key) or []

    hp = int(row.get(stat_cols["hp"], 0) or 0)
    attack = int(row.get(stat_cols["attack"], 0) or 0)
    defense = int(row.get(stat_cols["defense"], 0) or 0)
    sp_attack = int(row.get(stat_cols["sp_attack"], 0) or 0)
    sp_defense = int(row.get(stat_cols["sp_defense"], 0) or 0)
    speed = int(row.get(stat_cols["speed"], 0) or 0)
    bst = hp + attack + defense + sp_attack + sp_defense + speed

    type2_val = row.get("type2")
    type_label = row.get("type1", "Unknown")
    if pd.notna(type2_val) and str(type2_val).strip():
        type_label = f"{type_label}/{type2_val}"

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Profile")
        st.write(f"**Name:** {selected_name}")
        st.write(f"**Type:** {type_label}")
        st.write(f"**Archetype:** {row.get('archetype', 'Unknown')}")
        st.write(f"**Cluster:** {row.get('cluster', 'N/A')}")
        st.write(f"**Abilities:** {', '.join(abilities) if abilities else 'Unavailable'}")

    with right:
        st.markdown("#### Base Stats")
        stats_df = pd.DataFrame(
            [
                {"Stat": "HP", "Value": hp},
                {"Stat": "Attack", "Value": attack},
                {"Stat": "Defense", "Value": defense},
                {"Stat": "Sp. Attack", "Value": sp_attack},
                {"Stat": "Sp. Defense", "Value": sp_defense},
                {"Stat": "Speed", "Value": speed},
                {"Stat": "BST", "Value": bst},
            ]
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown("#### Pivot Profile")
    pivot_score = float(pd.to_numeric(pd.Series([row.get("pivot_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    pivot_candidate = "Yes" if pivot_score >= PIVOT_CANDIDATE_THRESHOLD else "No"
    c1, c2, c3 = st.columns(3)
    c1.metric("Pivot Score", f"{pivot_score:.2f}")
    c2.metric("Pivot Candidate", pivot_candidate)
    c3.metric("Style", str(row.get("pivot_style_hint", "hybrid")).title())

    pivot_details_df = pd.DataFrame(
        [
            {"Component": "Bulk", "Value": float(row.get("pivot_bulk_score", 0.0))},
            {"Component": "Speed", "Value": float(row.get("pivot_speed_score", 0.0))},
            {"Component": "Offense", "Value": float(row.get("pivot_offense_score", 0.0))},
            {"Component": "Type Utility", "Value": float(row.get("pivot_type_utility_score", 0.0))},
            {"Component": "Profile", "Value": float(row.get("pivot_profile_score", 0.0))},
            {"Component": "Ability Bonus (Dataset)", "Value": float(row.get("pivot_ability_bonus", 0.0))},
            {"Component": "Ability Bonus (Estimated)", "Value": _estimate_ability_bonus_from_names(abilities)},
        ]
    )
    st.dataframe(pivot_details_df, use_container_width=True, hide_index=True)

    profile_payload = {
        "name": selected_name,
        "type": type_label,
        "archetype": row.get("archetype", "Unknown"),
        "cluster": row.get("cluster", "N/A"),
        "abilities": abilities,
        "stats": {
            "hp": hp,
            "attack": attack,
            "defense": defense,
            "special_attack": sp_attack,
            "special_defense": sp_defense,
            "speed": speed,
            "bst": bst,
        },
        "pivot": {
            "threshold": PIVOT_CANDIDATE_THRESHOLD,
            "candidate": pivot_candidate.lower() == "yes",
            "score": pivot_score,
            "style_hint": row.get("pivot_style_hint", "hybrid"),
            "bulk_score": float(row.get("pivot_bulk_score", 0.0)),
            "speed_score": float(row.get("pivot_speed_score", 0.0)),
            "offense_score": float(row.get("pivot_offense_score", 0.0)),
            "type_utility_score": float(row.get("pivot_type_utility_score", 0.0)),
            "profile_score": float(row.get("pivot_profile_score", 0.0)),
            "ability_bonus_dataset": float(row.get("pivot_ability_bonus", 0.0)),
            "ability_bonus_estimated": _estimate_ability_bonus_from_names(abilities),
        },
    }
    st.download_button(
        label="Download Pokemon Profile (JSON)",
        data=json.dumps(profile_payload, indent=2, default=str).encode("utf-8"),
        file_name=f"pokemon_profile_{selected_name}.json",
        mime="application/json",
        use_container_width=True,
        key=f"dl_pokemon_profile_{selected_name}",
    )


def _render_team_analyzer_mode(data_df: pd.DataFrame) -> None:
    st.markdown("### Team Analyzer")
    st.caption("Select exactly six Pokemon and inspect strengths, issues, and recommendations.")
    with st.expander("Quick Guide", expanded=False):
        _guide_section(
            "Team Members (6)",
            "Choose the six Pokemon you currently use or want to test. In Pokemon battles, you always bring a team of six.",
        )
        _guide_section(
            "What The Analyzer Checks",
            "It checks whether many teammates share the same weaknesses (for example, all weak to Electric), whether your team jobs are balanced (tank, attacker, support), and whether your speed spread is healthy.",
        )
        _guide_section(
            "How To Use The Results",
            "Start with Team Issues to find the biggest problems, then use Recommendations to decide what kind of Pokemon to swap in or out.",
        )

    names = sorted(data_df["name"].astype(str).tolist())
    selected = st.multiselect(
        "Team Members (6)",
        options=names,
        default=[],
        max_selections=6,
        help="Choose six unique Pokemon to analyze as one team.",
    )

    if st.button("Analyze Team", type="primary", use_container_width=True):
        if len(selected) != 6:
            st.warning("Pick exactly six Pokemon before running analysis.")
            return

        resolved = _resolve_team_from_manual_input(selected, data_df)
        if len(resolved) != 6:
            st.warning("Some Pokemon names could not be resolved. Please reselect them.")
            return

        ok, payload, msg, log_path = _run_safe(
            "team_analyzer",
            lambda: _render_analysis_panel(resolved, data_df),
        )
        if not ok:
            _friendly_failure("Team Analyzer", msg, log_path)


def _render_team_generator_mode(data_df: pd.DataFrame) -> None:
    st.markdown("### Team Generator")
    st.caption("Build around anchors, choose a composition style, then optimize.")
    with st.expander("Quick Guide", expanded=False):
        _guide_section(
            "Anchor Pokemon",
            "These are guaranteed team members. If you always want Pikachu or Garchomp included, select them as anchors and the system builds around them.",
        )
        _guide_section(
            "Team Composition",
            "Composition means your team structure. Each Pokemon has an archetype (a team job), like tank (takes hits), sweeper (finishes weakened enemies), or generalist (flexible role).",
        )
        _guide_section(
            "Composition Strictness",
            "Higher strictness makes the model follow your chosen structure more exactly. Lower strictness allows creative picks if they improve score.",
        )
        _guide_section(
            "Power Mode",
            "Controls how hard the optimizer limits teams with very high combined base stats. Use Standard unless you specifically want looser or stricter power limits.",
        )
        _guide_section(
            "Optimization Profile",
            "Profiles change how the genetic algorithm explores teams. Full Balance is usually best for beginners because it balances strength, coverage, and diversity.",
        )
        _guide_section(
            "Population / Generations",
            "Population is how many team ideas are tested each round. Generations are how many rounds of improvement happen. Larger values usually improve quality but increase runtime.",
        )
        _guide_section(
            "Random Seed",
            "Keep the same seed to reproduce a similar run. Change the seed when you want different team outcomes.",
        )
        _guide_section(
            "Refresh Process",
            "After you press Generate Team, check Job Status & Latest Result at the bottom. Use Refresh Job Status there to poll progress. While one generation is queued/running, starting another is blocked.",
        )

    names = sorted(data_df["name"].astype(str).tolist())
    style_labels = {
        "balanced": "Balanced",
        "hyper_offense": "Hyper Offense",
        "pivot_pressure": "Pivot Pressure",
        "bulky_offense": "Bulky Offense",
    }

    col_a, col_b = st.columns([1, 1])
    with col_a:
        anchors = st.multiselect(
            "Anchor Pokemon (1-5)",
            options=names,
            default=[],
            max_selections=5,
            help="Anchors are locked into the generated team. The optimizer fills remaining slots.",
        )
        composition_name = st.selectbox(
            "Team Composition",
            options=list(style_labels.keys()),
            format_func=lambda x: style_labels[x],
            index=0,
            help="A target mix of archetypes (for example tank/sweeper/generalist) for the six-team lineup.",
        )
        composition_weight = st.slider(
            "Composition Strictness",
            min_value=0.05,
            max_value=0.40,
            value=0.20,
            step=0.01,
            help="Higher values force the optimizer to follow your composition target more strictly.",
        )
    with col_b:
        power_mode = st.selectbox(
            "Power Mode",
            options=["standard", "competitive_strict", "open"],
            format_func=lambda x: x.replace("_", " ").title(),
            index=0,
            help="Controls how strongly very high total stats are penalized.",
        )
        profile_labels = [
            "Baseline (Fast, Minimal Constraints)",
            "Weighted Start (Coverage-Aware Initialization)",
            "Full Balance (Recommended)",
        ]
        profile_to_config = {
            profile_labels[0]: "A",
            profile_labels[1]: "B",
            profile_labels[2]: "C",
        }
        profile_label = st.selectbox(
            "Optimization Profile",
            profile_labels,
            index=2,
            help="Named profiles replace A/B/C. 'Full Balance' is recommended for most users.",
        )
        top_n = st.slider("Top Teams to Show", min_value=1, max_value=5, value=1, step=1)
        

    st.markdown("#### GA Runtime")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        population = st.slider(
            "Population",
            min_value=20,
            max_value=500,
            value=150,
            step=10,
            help="How many team candidates are tested per generation.",
        )
    with c2:
        generations = st.slider(
            "Generations",
            min_value=10,
            max_value=300,
            value=80,
            step=10,
            help="Optimization rounds. More rounds can improve quality but take longer.",
        )
    with c3:
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            value=42,
            step=1,
            help="Reuse the same seed to reproduce results.",
        )
    # Informative note for user expectation
    st.caption(f"This process may take up to 1-2 minutes for large populations/generations. Please wait for results to appear. (Running {generations} generations.)")

    if st.button("Generate Team", type="primary", use_container_width=True):
        if not anchors:
            st.warning("Select at least one anchor Pokemon.")
            return

        if _is_generation_in_progress():
            active = _get_active_job_record()
            active_id = active.job_id if active is not None else "unknown"
            st.warning(
                f"A generation is already in progress (Job ID: {active_id}). Please use Refresh Job Status in the bottom section before starting another run."
            )
            return

        job_id, error = _submit_ga_job(
            config_name=profile_to_config[profile_label],
            population=int(population),
            generations=int(generations),
            seed=int(seed),
            top_n=int(top_n),
            locked_names=anchors,
            composition_name=composition_name,
            composition_weight=float(composition_weight),
            power_mode=power_mode,
        )
        if error:
            st.warning(error)
        else:
            st.session_state[ACTIVE_JOB_ID_KEY] = job_id
            st.rerun()


def _render_random_team_mode(data_df: pd.DataFrame) -> None:
    st.markdown("### Random Team")
    st.caption("Generate a creative team with optional single anchor.")
    with st.expander("Quick Guide", expanded=False):
        _guide_section(
            "Optional Anchor",
            "Pick one favorite Pokemon to force into the team, or leave it empty for a fully free random-style search.",
        )
        _guide_section(
            "Top Teams To Show",
            "Choose how many final candidates you want to compare. Showing more options helps if you want to pick based on playstyle preference.",
        )
        _guide_section(
            "Population / Generations",
            "Even in Random Team mode, the optimizer still evaluates many possible teams. Higher values mean deeper search and usually better random candidates.",
        )
        _guide_section(
            "Random Seed",
            "Use a fixed seed for repeatable experiments. Change it when you want fresh random variation.",
        )
        _guide_section(
            "Refresh Process",
            "After you press Generate Random Team, check Job Status & Latest Result at the bottom. Use Refresh Job Status there to poll progress. While one generation is queued/running, starting another is blocked.",
        )

    names = sorted(data_df["name"].astype(str).tolist())
    anchor_options = ["(None)"] + names

    col1, col2 = st.columns([1, 1])
    with col1:
        anchor = st.selectbox(
            "Optional Anchor",
            options=anchor_options,
            index=0,
            help="Include one chosen Pokemon, or leave empty for no forced pick.",
        )
        top_n = st.slider(
            "Top Teams to Show",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="Show multiple candidate teams to compare options.",
        )
    with col2:
        population = st.slider(
            "Population",
            min_value=20,
            max_value=500,
            value=150,
            step=10,
            key="rand_pop",
            help="More candidates can improve quality but increase runtime.",
        )
        generations = st.slider(
            "Generations",
            min_value=10,
            max_value=300,
            value=80,
            step=10,
            key="rand_gen",
            help="More rounds means more refinement of team candidates.",
        )

    c1, c2 = st.columns(2)
    with c1:
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            value=42,
            step=1,
            key="rand_seed",
            help="Lock seed for repeatable random results.",
        )


    if st.button("Generate Random Team", type="primary", use_container_width=True):
        if _is_generation_in_progress():
            active = _get_active_job_record()
            active_id = active.job_id if active is not None else "unknown"
            st.warning(
                f"A generation is already in progress (Job ID: {active_id}). Please use Refresh Job Status in the bottom section before starting another run."
            )
            return

        locked = [] if anchor == "(None)" else [anchor]
        job_id, error = _submit_ga_job(
            config_name="Random",
            population=int(population),
            generations=int(generations),
            seed=int(seed),
            top_n=int(top_n),
            locked_names=locked,
            composition_name="balanced",
            composition_weight=0.20,
            power_mode="standard",
        )
        if error:
            st.warning(error)
        else:
            st.session_state[ACTIVE_JOB_ID_KEY] = job_id
            st.rerun()


def _render_saved_teams_mode(data_df: pd.DataFrame) -> None:
    st.markdown("### View Generated Teams")
    st.caption("Saved teams are session-scoped and cleared when this session ends.")

    session_id = _get_session_id()
    team_store = _get_team_store()
    saved_teams = team_store.list_teams(session_id=session_id)

    if st.button("Clear Saved Teams", type="secondary"):
        team_store.clear_session(session_id)
        st.success("Cleared all saved teams for this session.")
        st.rerun()

    if not saved_teams:
        st.info("No teams have been saved in this session yet.")
        return

    st.write(f"{len(saved_teams)} saved team(s) in this session.")

    for record in saved_teams:
        team_payload = record["team_payload"]
        metadata = record.get("metadata", {})
        team_names = [p["name"] for p in team_payload.get("pokemon", []) if "name" in p]
        team_table_df = _team_table(team_payload.get("pokemon", []))
        breakdown_df = _format_breakdown(team_payload.get("breakdown", {}))
        record_key = record["id"]

        with st.expander(
            f"{record['nickname']} | Rank {record.get('rank', 0)} | Fitness {float(record.get('fitness', 0.0)):.4f}",
            expanded=False,
        ):
            col_meta, col_actions = st.columns([2, 1])
            with col_meta:
                st.write(f"**Saved:** {record.get('created_at', 'Unknown')}")
                st.write(f"**Config:** {metadata.get('config_name', 'Unknown')}")
                st.write(f"**Composition:** {metadata.get('composition_name', 'Unknown')}")
                st.write(f"**Power Mode:** {metadata.get('power_mode', 'Unknown')}")
                st.write(f"**Seed:** {metadata.get('seed', 'Unknown')}")
            with col_actions:
                renamed_value = st.text_input(
                    "Rename",
                    value=record["nickname"],
                    key=f"rename_{record_key}",
                )
                if st.button("Update Nickname", key=f"update_{record_key}", use_container_width=True):
                    if renamed_value.strip():
                        team_store.rename_team(record_key, renamed_value)
                        st.success("Nickname updated.")
                        st.rerun()
                    else:
                        st.warning("Nickname cannot be empty.")
                if st.button("Delete Team", key=f"delete_{record_key}", use_container_width=True):
                    team_store.delete_team(record_key)
                    st.success("Saved team deleted.")
                    st.rerun()

            st.dataframe(team_table_df, use_container_width=True, hide_index=True)
            if not breakdown_df.empty:
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            if team_names:
                with st.expander("Re-open Analysis", expanded=False):
                    _render_analysis_panel(team_names, data_df, key_suffix=f"_saved_{record_key}")

            st.download_button(
                label="Download Saved Team JSON",
                data=json.dumps(team_payload, indent=2, default=str).encode("utf-8"),
                file_name=f"saved_team_{record_key}.json",
                mime="application/json",
                key=f"download_saved_{record_key}",
                use_container_width=True,
            )


def main() -> None:
    _inject_theme()
    data_df = load_pokemon_data()

    st.markdown(
        """
        <div class="hero">
          <h2 style="margin:0; color:#eefcf6;">Pokemon Team Optimizer</h2>
          <p style="margin:.35rem 0 0 0; color:#d8f5e8;">
            Generate stronger squads, audit weaknesses, and iterate quickly with GA-backed recommendations.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Select",
            options=["Team Generator", "Team Analyzer", "Random Team", "Pokemon Info", "View Generated Teams"],
            index=0,
            help="Choose what you want to do: build a team, analyze a team, explore random lineups, inspect one Pokemon, or review saved teams.",
        )
        st.markdown("---")
        st.caption("If something goes wrong, a Download Error Log button will appear.")
        st.caption("Saved teams are session-only and will not persist after the session ends.")
        st.markdown("---")
        st.markdown(
            '<a href="https://github.com/acm-uic/SIG-AI-Project-Track-Fall-2026" target="_blank"><button style="width:100%;padding:0.5em 0.8em;font-size:1.1em;background:#3B82F6;color:white;border:none;border-radius:8px;cursor:pointer;">View Repo</button></a>',
            unsafe_allow_html=True,
        )

    if mode == "Team Generator":
        _render_team_generator_mode(data_df)
    elif mode == "Team Analyzer":
        _render_team_analyzer_mode(data_df)
    elif mode == "Pokemon Info":
        _render_pokemon_info_mode(data_df)
    elif mode == "View Generated Teams":
        _render_saved_teams_mode(data_df)
    else:
        _render_random_team_mode(data_df)

    _render_job_output_section(data_df)

    with st.expander("Run Metadata"):
        st.code(
            json.dumps(
                {
                    "project_root": str(PROJECT_ROOT),
                    "mode": mode,
                    "rows_loaded": int(len(data_df)),
                    "archetypes": int(data_df["archetype"].nunique() if "archetype" in data_df.columns else 0),
                    "types": int(
                        len(
                            set(data_df.get("type1", pd.Series(dtype=str)).dropna().astype(str))
                            .union(set(data_df.get("type2", pd.Series(dtype=str)).dropna().astype(str)))
                        )
                    ),
                    "type_chart_size": len(TYPE_NAMES),
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            ),
            language="json",
        )


if __name__ == "__main__":
    main()
