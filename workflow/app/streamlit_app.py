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

import pandas as pd
import streamlit as st

# Ensure project imports resolve when launched via streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core.paths import reports_dir
from src.ga import PokemonGA, load_pokemon_data
from src.ga.config import get_config_a, get_config_b, get_config_c, get_config_random
from src.ga.fitness import TYPE_NAMES

from legacy.scripts.cli import analyze_team_by_names


st.set_page_config(page_title="Pokemon Team Optimizer", layout="wide")

PIVOT_CANDIDATE_THRESHOLD = 0.62


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


def _error_log_dir() -> Path:
    path = reports_dir() / "error_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_error_log(operation: str, exc: Exception) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _error_log_dir() / f"streamlit_{operation}_{ts}.log"
    details = [
        f"Operation: {operation}",
        f"Timestamp: {ts}",
        f"ExceptionType: {type(exc).__name__}",
        f"Message: {exc}",
        "",
        "Traceback:",
        traceback.format_exc(),
    ]
    log_path.write_text("\n".join(details), encoding="utf-8")
    return log_path


def _run_safe(operation: str, fn: Callable[[], Any]) -> tuple[bool, Any, str | None, Path | None]:
    try:
        result = fn()
        return True, result, None, None
    except Exception as exc:  # noqa: BLE001
        log_path = _write_error_log(operation, exc)
        return False, None, str(exc), log_path


def _friendly_failure(operation: str, message: str | None, log_path: Path | None) -> None:
    st.error(f"{operation} failed.")
    if message:
        st.caption(f"Reason: {message}")
    if log_path:
        st.caption(f"Diagnostic log: {log_path}")


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


def _render_analysis_panel(team_names: list[str], data_df: pd.DataFrame) -> None:
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


def _run_ga_workflow(
    config_name: str,
    population: int,
    generations: int,
    seed: int,
    top_n: int,
    save_outputs: bool,
    locked_names: list[str],
    composition_name: str = "balanced",
    composition_weight: float = 0.20,
    power_mode: str = "standard",
) -> dict[str, Any]:
    config = _config_from_name(config_name)
    config["name"] = f"Streamlit_{config_name}"
    config["population"]["size"] = population
    config["population"]["generations"] = generations
    config["random_seed"] = seed

    data_df = load_pokemon_data()
    presets = _build_composition_presets()
    target_map = _normalize_composition_target(presets[composition_name], data_df)
    power_cfg = _power_mode_config(power_mode)

    config["fitness"]["composition_weight"] = float(composition_weight)
    config["fitness"]["target_archetype_counts"] = target_map
    config["fitness"]["bst_cap"] = int(power_cfg["bst_cap"])
    config["fitness"]["bst_penalty_weight"] = float(power_cfg["bst_penalty_weight"])
    config["fitness"]["pivot_weight"] = 0.0
    config["fitness"]["target_pivot_count"] = 0
    config["fitness"]["pivot_threshold"] = PIVOT_CANDIDATE_THRESHOLD
    if composition_name == "pivot_pressure":
        config["fitness"]["pivot_weight"] = 0.18
        config["fitness"]["target_pivot_count"] = 3

    output_dir = None
    if save_outputs:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = reports_dir() / "ga_results" / f"run_streamlit_{ts}"

    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        ga = PokemonGA(data_df, config, output_dir=output_dir, locked_pokemon=locked_names)
        history_df = ga.run()
        best_teams = ga.get_best_teams(top_n)
        if save_outputs and output_dir is not None:
            ga.export_results(output_dir)

    top_teams_payload = []
    for rank, (team_df, fitness, breakdown) in enumerate(best_teams, start=1):
        top_teams_payload.append(
            {
                "rank": rank,
                "fitness": float(fitness),
                "breakdown": breakdown,
                "pokemon": _serialize_team(team_df),
            }
        )

    return {
        "run_log": stdout_buffer.getvalue(),
        "history": history_df,
        "best_fitness": float(best_teams[0][1]) if best_teams else None,
        "output_dir": str(output_dir) if output_dir else None,
        "top_teams": top_teams_payload,
        "config_used": config,
    }


def _render_ga_results(results: dict[str, Any], data_df: pd.DataFrame, include_analysis: bool = True) -> None:
    st.success("GA run completed successfully.")
    if results["best_fitness"] is not None:
        st.metric("Best Fitness", f"{results['best_fitness']:.4f}")

    if results["output_dir"]:
        st.caption(f"Output directory: {results['output_dir']}")

    history_df = results["history"]
    if not history_df.empty:
        plot_cols = [c for c in ["max_fitness", "mean_fitness", "min_fitness"] if c in history_df.columns]
        if plot_cols:
            st.subheader("Fitness History")
            st.line_chart(history_df[plot_cols])

    st.subheader("Generated Team")
    for team in results["top_teams"]:
        with st.expander(f"Rank {team['rank']} | Fitness {team['fitness']:.4f}", expanded=team["rank"] == 1):
            st.dataframe(_team_table(team["pokemon"]), use_container_width=True, hide_index=True)
            breakdown_df = _format_breakdown(team["breakdown"])
            if not breakdown_df.empty:
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

            if include_analysis:
                st.markdown("### Team Analyzer")
                team_names = [p["name"] for p in team["pokemon"] if "name" in p]
                _render_analysis_panel(team_names, data_df)

    with st.expander("Run Log"):
        st.code(results["run_log"] or "(no output)", language="text")


def _resolve_team_from_manual_input(input_names: list[str], data_df: pd.DataFrame) -> list[str]:
    available = set(data_df["name"].astype(str).tolist())
    resolved = []
    for name in input_names:
        if name in available:
            resolved.append(name)
    return resolved


def _guide_section(title: str, text: str) -> None:
    st.markdown(
        f"<div style='color: var(--ink);'><strong>{title}</strong><br>{text}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)


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
    with c4:
        save_outputs = st.checkbox(
            "Save Artifacts",
            value=False,
            help="Store logs and run outputs for later review.",
        )

    if st.button("Generate Team", type="primary", use_container_width=True):
        if not anchors:
            st.warning("Select at least one anchor Pokemon.")
            return

        with st.spinner("Optimizing team..."):
            ok, payload, msg, log_path = _run_safe(
                "team_generator",
                lambda: _run_ga_workflow(
                    config_name=profile_to_config[profile_label],
                    population=int(population),
                    generations=int(generations),
                    seed=int(seed),
                    top_n=int(top_n),
                    save_outputs=save_outputs,
                    locked_names=anchors,
                    composition_name=composition_name,
                    composition_weight=float(composition_weight),
                    power_mode=power_mode,
                ),
            )

        if ok:
            _render_ga_results(payload, data_df, include_analysis=True)
        else:
            _friendly_failure("Team Generator", msg, log_path)


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
    with c2:
        save_outputs = st.checkbox(
            "Save Artifacts",
            value=False,
            key="rand_save",
            help="Store outputs and logs to inspect this run later.",
        )

    if st.button("Generate Random Team", type="primary", use_container_width=True):
        locked = [] if anchor == "(None)" else [anchor]
        with st.spinner("Generating random team..."):
            ok, payload, msg, log_path = _run_safe(
                "random_team",
                lambda: _run_ga_workflow(
                    config_name="Random",
                    population=int(population),
                    generations=int(generations),
                    seed=int(seed),
                    top_n=int(top_n),
                    save_outputs=save_outputs,
                    locked_names=locked,
                    composition_name="balanced",
                    composition_weight=0.20,
                    power_mode="standard",
                ),
            )

        if ok:
            _render_ga_results(payload, data_df, include_analysis=True)
        else:
            _friendly_failure("Random Team", msg, log_path)


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
            options=["Team Generator", "Team Analyzer", "Random Team"],
            index=0,
            help="Choose what you want to do: build a team, analyze a team, or explore random lineups.",
        )
        st.markdown("---")
        st.caption("Errors are logged to `reports/error_logs/` for debugging.")

    if mode == "Team Generator":
        _render_team_generator_mode(data_df)
    elif mode == "Team Analyzer":
        _render_team_analyzer_mode(data_df)
    else:
        _render_random_team_mode(data_df)

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
