import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

GOLD_PATH = Path("data/gold")
LANGUAGE_NAMES = {
    "de-en": "German - English",
    "fr-en": "French - English",
    "es-en": "Spanish - English",
    "bg-en": "Bulgarian - English",
}

st.set_page_config(page_title="EU Parliamentary NLP Pipeline", layout="wide", page_icon="🇪🇺")


@st.cache_data
def load_gold_data() -> dict:
    data = {}
    for f in GOLD_PATH.glob("europarl_*_gold_*.json"):
        parts = f.stem.split("_")
        lang_pair = parts[1] + "-" + parts[2]
        with open(f, "r", encoding="utf-8") as fp:
            data[lang_pair] = json.load(fp)
    return data


@st.cache_data
def load_summary() -> dict:
    summary_file = GOLD_PATH / "summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def render_header():
    st.title("🇪🇺 EU Parliamentary Speeches — NLP Pipeline Dashboard")
    st.markdown("Multilingual sentiment analysis and entity extraction from the Europarl corpus.")
    st.divider()


def render_summary_metrics(summary: dict, gold_data: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Language Pairs", len(gold_data))
    col2.metric("Total Records Processed", f"{summary.get('total_records_processed', 0):,}")
    col3.metric("Languages", "DE, FR, ES, BG → EN")
    col4.metric("Pipeline Layers", "Bronze → Silver → Gold")


def render_sentiment_overview(gold_data: dict):
    st.subheader("Sentiment Distribution by Language Pair")

    rows = []
    for lp, data in gold_data.items():
        dist = data["sentiment_distribution"]
        name = LANGUAGE_NAMES.get(lp, lp)
        rows.append({"Language Pair": name, "Sentiment": "Positive", "Percentage": dist["positive_pct"]})
        rows.append({"Language Pair": name, "Sentiment": "Neutral", "Percentage": dist["neutral_pct"]})
        rows.append({"Language Pair": name, "Sentiment": "Negative", "Percentage": dist["negative_pct"]})

    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Language Pair", y="Percentage", color="Sentiment",
        color_discrete_map={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"},
        barmode="stack", height=400,
        labels={"Percentage": "Percentage (%)"}
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_family="Arial")
    st.plotly_chart(fig, use_container_width=True)


def render_avg_sentiment(gold_data: dict):
    st.subheader("Average Sentiment Score by Language Pair")
    lang_pairs = [LANGUAGE_NAMES.get(lp, lp) for lp in gold_data]
    scores = [data["avg_sentiment_score"] for data in gold_data.values()]
    colors = ["#2ecc71" if s > 0 else "#e74c3c" if s < 0 else "#95a5a6" for s in scores]

    fig = go.Figure(go.Bar(
        x=lang_pairs, y=scores,
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside"
    ))
    fig.update_layout(
        yaxis_title="Avg Sentiment Score", plot_bgcolor="white",
        paper_bgcolor="white", font_family="Arial", height=350
    )
    st.plotly_chart(fig, use_container_width=True)


def render_lang_pair_detail(gold_data: dict):
    st.subheader("Language Pair Detail")
    selected = st.selectbox(
        "Select language pair",
        options=list(gold_data.keys()),
        format_func=lambda lp: LANGUAGE_NAMES.get(lp, lp)
    )

    data = gold_data[selected]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Named Entities**")
        entities = data.get("top_entities", [])
        if entities:
            df_ent = pd.DataFrame(entities[:15])
            fig = px.bar(
                df_ent, x="count", y="text", orientation="h",
                color="label", height=400,
                labels={"count": "Frequency", "text": "Entity", "label": "Type"}
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              font_family="Arial", yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No entities found.")

    with col2:
        st.markdown("**Top Keywords**")
        keywords = data.get("top_keywords", [])
        if keywords:
            df_kw = pd.DataFrame(keywords[:20])
            fig = px.bar(
                df_kw, x="count", y="keyword", orientation="h",
                height=400, color_discrete_sequence=["#3498db"],
                labels={"count": "Frequency", "keyword": "Keyword"}
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              font_family="Arial", yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No keywords found.")


def render_sentiment_pie(gold_data: dict):
    st.subheader("Sentiment Breakdown per Language Pair")
    cols = st.columns(len(gold_data))
    for col, (lp, data) in zip(cols, gold_data.items()):
        dist = data["sentiment_distribution"]
        fig = go.Figure(go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[dist["positive"], dist["neutral"], dist["negative"]],
            marker_colors=["#2ecc71", "#95a5a6", "#e74c3c"],
            hole=0.4
        ))
        fig.update_layout(
            title=LANGUAGE_NAMES.get(lp, lp), height=280,
            margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False, font_family="Arial"
        )
        col.plotly_chart(fig, use_container_width=True)


def main():
    gold_data = load_gold_data()
    summary = load_summary()

    if not gold_data:
        st.error("No gold data found. Please run the full pipeline first.")
        return

    render_header()
    render_summary_metrics(summary, gold_data)
    st.divider()
    render_sentiment_overview(gold_data)
    render_avg_sentiment(gold_data)
    st.divider()
    render_sentiment_pie(gold_data)
    st.divider()
    render_lang_pair_detail(gold_data)


if __name__ == "__main__":
    main()