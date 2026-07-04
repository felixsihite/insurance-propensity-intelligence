from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from insurance_propensity.config import DARK_THEME, LIGHT_THEME  # noqa: E402


st.set_page_config(
    page_title="Insurance Propensity Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "Executive Summary",
    "EDA Insights",
    "Model Performance",
    "Propensity Scoring",
    "Customer Segments",
    "Interpretability",
    "Campaign Targeting",
]

PAGE_SLUGS = {
    "executive-summary": "Executive Summary",
    "eda-insights": "EDA Insights",
    "model-performance": "Model Performance",
    "propensity-scoring": "Propensity Scoring",
    "customer-segments": "Customer Segments",
    "interpretability": "Interpretability",
    "campaign-targeting": "Campaign Targeting",
}


def resolve_theme(dark_mode: bool) -> dict[str, str]:
    theme = dict(DARK_THEME if dark_mode else LIGHT_THEME)
    theme.setdefault("border", theme.get("secondary_surface", "#B8C8D8"))
    theme.setdefault("control", theme.get("secondary_surface", "#EEF5FA"))
    theme.setdefault("plot", theme.get("card", "#F8FBFE"))
    theme.setdefault("grid", theme.get("secondary_surface", "#C6D3DF"))
    return theme


def apply_theme(dark_mode: bool) -> dict[str, str]:
    theme = resolve_theme(dark_mode)
    header_background = "#07111F" if dark_mode else "#0B1F33"
    selected_background = theme["blue"] if dark_mode else theme["teal"]
    selected_text = "#06111F" if dark_mode else "#FFFFFF"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {theme["background"]};
            color: {theme["text"]};
        }}
        [data-testid="stHeader"] {{
            background: {header_background};
            border-bottom: 1px solid {theme["border"]};
        }}
        [data-testid="stSidebar"] {{
            background: {theme["card"]};
            border-right: 1px solid {theme["border"]};
            min-width: 320px;
            max-width: 340px;
        }}
        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 100%;
            overflow-x: hidden;
        }}
        h1 {{
            font-size: 2.45rem !important;
            line-height: 1.12 !important;
            overflow-wrap: break-word;
        }}
        h2 {{
            font-size: 1.65rem !important;
        }}
        .metric-card {{
            background: {theme["card"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            padding: 16px 18px;
            min-height: 108px;
            min-width: 0;
            overflow-wrap: break-word;
            box-shadow: 0 12px 28px rgba(11, 31, 51, 0.08);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            width: 100%;
            margin: 10px 0 18px 0;
        }}
        .metric-card .label {{
            color: {theme["muted_text"]};
            font-size: 0.82rem;
            margin-bottom: 8px;
        }}
        .metric-card .value {{
            color: {theme["text"]};
            font-size: 1.3rem;
            font-weight: 700;
            line-height: 1.25;
        }}
        .section-panel {{
            background: {theme["card"]};
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            padding: 18px;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span,
        [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {{
            color: {theme["text"]};
        }}
        [data-testid="stCaptionContainer"] {{
            color: {theme["muted_text"]};
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            overflow: hidden;
        }}
        [data-baseweb="select"] > div {{
            background-color: {theme["control"]};
            border-color: {theme["border"]};
            color: {theme["text"]};
        }}
        [data-baseweb="tag"] {{
            background-color: {selected_background} !important;
            border: 1px solid {selected_background} !important;
            color: {selected_text} !important;
        }}
        [data-baseweb="tag"] span {{
            color: {selected_text} !important;
        }}
        input, textarea {{
            color: {theme["text"]} !important;
        }}
        .stButton button, .stDownloadButton button {{
            background: {selected_background};
            color: {selected_text};
            border: 1px solid {selected_background};
            border-radius: 8px;
            font-weight: 700;
        }}
        .stButton button:hover, .stDownloadButton button:hover {{
            filter: brightness(1.05);
            border-color: {theme["blue"]};
        }}
        div[role="radiogroup"] label {{
            color: {theme["text"]};
        }}
        hr {{
            border-color: {theme["border"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return theme


def chart_palette(dark_mode: bool) -> list[str]:
    if dark_mode:
        return ["#38BDF8", "#2DD4BF", "#22C55E", "#F59E0B", "#EF4444", "#A78BFA"]
    return ["#2563EB", "#1F7A8C", "#2E7D32", "#B7791F", "#C62828", "#7C3AED"]


def continuous_scale(dark_mode: bool) -> list[str]:
    if dark_mode:
        return ["#17385A", "#1F7A8C", "#38BDF8"]
    return ["#D6E4F0", "#1F7A8C", "#0B1F33"]


def style_plotly(fig, theme: dict[str, str]):
    fig.update_layout(
        paper_bgcolor=theme["card"],
        plot_bgcolor=theme["plot"],
        font={"color": theme["text"], "family": "Inter, Segoe UI, Arial"},
        title={"font": {"color": theme["text"], "size": 18}},
        legend={
            "font": {"color": theme["text"]},
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 20, "r": 20, "t": 56, "b": 40},
        hoverlabel={
            "bgcolor": theme["card"],
            "bordercolor": theme["border"],
            "font": {"color": theme["text"]},
        },
    )
    fig.update_xaxes(
        color=theme["text"],
        gridcolor=theme["grid"],
        linecolor=theme["border"],
        zerolinecolor=theme["border"],
        title_font={"color": theme["text"]},
        tickfont={"color": theme["muted_text"]},
    )
    fig.update_yaxes(
        color=theme["text"],
        gridcolor=theme["grid"],
        linecolor=theme["border"],
        zerolinecolor=theme["border"],
        title_font={"color": theme["text"]},
        tickfont={"color": theme["muted_text"]},
    )
    return fig


def default_theme_from_query() -> bool:
    return str(st.query_params.get("theme", "light")).lower() == "dark"


def default_page_from_query() -> str:
    requested = str(st.query_params.get("page", "executive-summary")).lower()
    return PAGE_SLUGS.get(requested, "Executive Summary")


@st.cache_data(show_spinner=False)
def load_tables():
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
    model_comparison = pd.read_csv(reports_dir / "model_comparison.csv")
    deciles = pd.read_csv(reports_dir / "validation_decile_report.csv")
    eda = pd.read_csv(reports_dir / "eda_highlights.csv")
    importance = pd.read_csv(reports_dir / "feature_importance.csv")
    scored = pd.read_csv(predictions_dir / "test_customer_scores.csv")
    metrics = json.loads((reports_dir / "metrics.json").read_text(encoding="utf-8"))
    return model_comparison, deciles, eda, importance, scored, metrics


@st.cache_resource(show_spinner=False)
def load_scorer():
    return joblib.load(PROJECT_ROOT / "models" / "final_propensity_model.joblib")


def metric_grid(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def filter_scored(scored: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.subheader("Customer Filters")
        gender = st.multiselect("Gender", sorted(scored["Gender"].dropna().unique()), default=sorted(scored["Gender"].dropna().unique()))
        vehicle_damage = st.multiselect("Vehicle Damage", sorted(scored["Vehicle_Damage"].dropna().unique()), default=sorted(scored["Vehicle_Damage"].dropna().unique()))
        decile = st.multiselect("Propensity Decile", sorted(scored["propensity_decile"].dropna().unique()), default=[1, 2, 3])
        age_group = st.multiselect("Age Group", sorted(scored["age_group"].dropna().unique()), default=sorted(scored["age_group"].dropna().unique()))

    mask = (
        scored["Gender"].isin(gender)
        & scored["Vehicle_Damage"].isin(vehicle_damage)
        & scored["propensity_decile"].isin(decile)
        & scored["age_group"].isin(age_group)
    )
    return scored.loc[mask].copy()


def executive_page(model_comparison, scored, metrics, theme: dict[str, str], dark_mode: bool):
    st.title("Propensity Intelligence")
    st.caption("Cross-sell ranking system for prioritizing health insurance customers most likely to consider vehicle insurance.")
    metric_grid(
        [
            ("Final Model", model_comparison.iloc[0]["model_name"]),
            ("Validation PR-AUC", f"{metrics['pr_auc']:.3f}"),
            ("Top Decile Lift", f"{metrics['top_decile_lift']:.2f}x"),
            ("Top 30% Capture", f"{metrics['capture_rate_top_30_pct']:.1%}"),
        ]
    )

    st.subheader("Ranked Test Customers")
    filtered = filter_scored(scored)
    metric_grid(
        [
            ("Filtered Customers", f"{len(filtered):,}"),
            ("Average Score", f"{filtered['propensity_score'].mean():.3f}" if len(filtered) else "0.000"),
            ("Priority Share", f"{(filtered['propensity_decile'].le(3).mean()):.1%}" if len(filtered) else "0.0%"),
        ]
    )

    fig = px.histogram(
        filtered,
        x="propensity_score",
        color="propensity_decile",
        nbins=40,
        color_discrete_sequence=chart_palette(dark_mode),
        title="Propensity Score Distribution",
    )
    st.plotly_chart(style_plotly(fig, theme), use_container_width=True)


def eda_page(eda, theme: dict[str, str], dark_mode: bool):
    st.title("EDA Insights")
    st.caption("Observed response patterns from training data. These are associative signals, not causal claims.")
    selected_dimension = st.selectbox("Dimension", sorted(eda["dimension"].unique()))
    view = eda.loc[eda["dimension"].eq(selected_dimension)].sort_values("response_rate", ascending=False)
    fig = px.bar(view, x="response_rate", y="segment_value", orientation="h", color="response_rate", color_continuous_scale=continuous_scale(dark_mode))
    fig.update_layout(yaxis_title="", xaxis_tickformat=".0%")
    st.plotly_chart(style_plotly(fig, theme), use_container_width=True)
    st.dataframe(view, use_container_width=True, hide_index=True)


def model_page(model_comparison, deciles, metrics, theme: dict[str, str], dark_mode: bool):
    st.title("Model Performance")
    metric_grid(
        [
            ("ROC-AUC", f"{metrics['roc_auc']:.3f}"),
            ("PR-AUC", f"{metrics['pr_auc']:.3f}"),
            ("Top Decile Lift", f"{metrics['top_decile_lift']:.2f}x"),
            ("Top 30% Capture", f"{metrics['capture_rate_top_30_pct']:.1%}"),
        ]
    )
    st.dataframe(model_comparison, use_container_width=True, hide_index=True)
    cols = st.columns(2)
    with cols[0]:
        fig = px.bar(deciles, x="propensity_decile", y="lift", title="Lift by Decile", color="lift", color_continuous_scale=continuous_scale(dark_mode))
        st.plotly_chart(style_plotly(fig, theme), use_container_width=True)
    with cols[1]:
        fig = px.line(deciles, x="population_share", y="cumulative_capture_rate", markers=True, title="Cumulative Gain")
        fig.update_traces(line={"color": theme["blue"], "width": 3}, marker={"color": theme["teal"], "size": 8})
        fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(style_plotly(fig, theme), use_container_width=True)


def scoring_page(scored):
    st.title("Propensity Scoring")
    st.caption("Use prepared scored customers or upload raw customer rows with the original Kaggle schema.")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is not None:
        scorer = load_scorer()
        raw = pd.read_csv(uploaded)
        scored_view = scorer.score(raw)
    else:
        scored_view = filter_scored(scored)

    priority_cols = [
        "id",
        "propensity_score",
        "propensity_decile",
        "targeting_recommendation",
        "cross_sell_opportunity_segment",
        "Gender",
        "Age",
        "Vehicle_Age",
        "Vehicle_Damage",
        "Annual_Premium",
    ]
    st.dataframe(scored_view[priority_cols].sort_values("propensity_score", ascending=False).head(1000), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Scored Customers",
        scored_view.to_csv(index=False).encode("utf-8"),
        file_name="scored_customers.csv",
        mime="text/csv",
    )


def segment_page(scored, theme: dict[str, str], dark_mode: bool):
    st.title("Customer Segments")
    filtered = filter_scored(scored)
    segment_summary = (
        filtered.groupby(["propensity_decile", "cross_sell_opportunity_segment"], observed=True)
        .agg(customers=("id", "size"), avg_score=("propensity_score", "mean"), avg_premium=("Annual_Premium", "mean"))
        .reset_index()
        .sort_values(["propensity_decile", "avg_score"], ascending=[True, False])
    )
    fig = px.treemap(
        segment_summary,
        path=["propensity_decile", "cross_sell_opportunity_segment"],
        values="customers",
        color="avg_score",
        color_continuous_scale=continuous_scale(dark_mode),
        title="Segment Size and Average Propensity",
    )
    st.plotly_chart(style_plotly(fig, theme), use_container_width=True)
    st.dataframe(segment_summary, use_container_width=True, hide_index=True)


def interpretability_page(importance, theme: dict[str, str], dark_mode: bool):
    st.title("Model Interpretability")
    st.caption("Global feature drivers generated from validation-set permutation importance. SHAP is supported when installed.")
    top = importance.head(15).sort_values("importance_mean")
    fig = px.bar(top, x="importance_mean", y="feature", orientation="h", color="importance_mean", color_continuous_scale=continuous_scale(dark_mode))
    fig.update_layout(yaxis_title="")
    st.plotly_chart(style_plotly(fig, theme), use_container_width=True)
    st.dataframe(importance.head(30), use_container_width=True, hide_index=True)


def campaign_page(deciles, theme: dict[str, str]):
    st.title("Campaign Targeting Recommendation")
    st.caption("Validation deciles quantify how much responder concentration is captured as campaign reach expands.")
    view = deciles.copy()
    fig = px.area(view, x="population_share", y="cumulative_capture_rate", markers=True, title="Reach vs. Captured Responders")
    fig.update_traces(line={"color": theme["blue"], "width": 3}, fillcolor="rgba(56, 189, 248, 0.20)")
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(style_plotly(fig, theme), use_container_width=True)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.markdown(
        """
        <div class="section-panel">
        <strong>Recommended rollout:</strong> start with decile 1 for highest-priority outreach, expand to deciles 1-3 for scaled campaigns, and keep lower deciles for nurture or holdout measurement.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    default_page = default_page_from_query()
    dark_mode = st.sidebar.toggle("Dark mode", value=default_theme_from_query())
    theme = apply_theme(dark_mode)

    required = [
        PROJECT_ROOT / "outputs" / "reports" / "model_comparison.csv",
        PROJECT_ROOT / "outputs" / "predictions" / "test_customer_scores.csv",
        PROJECT_ROOT / "models" / "final_propensity_model.joblib",
    ]
    if any(not path.exists() for path in required):
        st.error("Project artifacts are not generated yet. Run: python scripts/run_project_pipeline.py")
        st.stop()

    model_comparison, deciles, eda, importance, scored, metrics = load_tables()
    page = st.sidebar.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(default_page),
    )

    if page == "Executive Summary":
        executive_page(model_comparison, scored, metrics, theme, dark_mode)
    elif page == "EDA Insights":
        eda_page(eda, theme, dark_mode)
    elif page == "Model Performance":
        model_page(model_comparison, deciles, metrics, theme, dark_mode)
    elif page == "Propensity Scoring":
        scoring_page(scored)
    elif page == "Customer Segments":
        segment_page(scored, theme, dark_mode)
    elif page == "Interpretability":
        interpretability_page(importance, theme, dark_mode)
    else:
        campaign_page(deciles, theme)


if __name__ == "__main__":
    main()