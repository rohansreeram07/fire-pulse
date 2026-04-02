"""
╔══════════════════════════════════════════════════════════════════╗
║              FirePulse — Streamlit Dashboard v1.0               ║
║       NLP Sentiment Analysis for Fire Departments               ║
║                                                                  ║
║  Run:  streamlit run app.py                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import io
import datetime
import warnings
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

# FirePulse core (must be in the same directory)
from firepulse import SentimentAnalyzer, SentimentVisualizer, SentimentExporter

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.0)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FirePulse",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Arial', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a1a2e 0%, #2c3e50 100%);
}
section[data-testid="stSidebar"] * { color: #ecf0f1 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 15px; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #f8f9fa;
    border-left: 4px solid #e74c3c;
    border-radius: 6px;
    padding: 12px 16px;
}

/* ── Section headers ── */
.section-header {
    background: linear-gradient(90deg, #e74c3c, #c0392b);
    color: white;
    padding: 10px 18px;
    border-radius: 6px;
    font-size: 17px;
    font-weight: bold;
    margin-bottom: 16px;
}

/* ── Sentiment badges ── */
.badge-positive { background:#2ecc71; color:white; padding:3px 12px;
                  border-radius:12px; font-weight:bold; font-size:13px; }
.badge-negative { background:#e74c3c; color:white; padding:3px 12px;
                  border-radius:12px; font-weight:bold; font-size:13px; }
.badge-neutral  { background:#95a5a6; color:white; padding:3px 12px;
                  border-radius:12px; font-weight:bold; font-size:13px; }

/* ── Score bar ── */
.score-bar-wrap { background:#e0e0e0; border-radius:8px; height:14px;
                  width:100%; margin-top:6px; }
.score-bar      { height:14px; border-radius:8px; }

/* ── Result card ── */
.result-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Footer ── */
.footer {
    text-align:center; color:#aaa; font-size:12px;
    margin-top:40px; padding-top:16px;
    border-top: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FirePulse engine...")
def load_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def load_viz():
    return SentimentVisualizer(output_dir="charts")

@st.cache_resource
def load_exporter():
    return SentimentExporter(output_dir="reports")

analyzer = load_analyzer()
viz      = load_viz()
exporter = load_exporter()

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}

def badge(label: str) -> str:
    return f'<span class="badge-{label}">{label.upper()}</span>'

def score_bar(score: float) -> str:
    """Render a colored progress bar for a compound score (-1 to 1)."""
    pct   = int((score + 1) / 2 * 100)
    color = PALETTE["positive"] if score >= 0.05 else (PALETTE["negative"] if score <= -0.05 else PALETTE["neutral"])
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
    )

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 CSV bytes for download."""
    export = df.copy()
    for col in ["key_themes", "top_keywords"]:
        if col in export.columns:
            export[col] = export[col].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else x)
    return export.to_csv(index=False).encode("utf-8")

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def render_chart(fig) -> bytes:
    return fig_to_bytes(fig)

def make_distribution_fig(df):
    counts = df["label"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    colors = [PALETTE[l] for l in counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("Sentiment Distribution", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig

def make_theme_fig(df, top_n=7):
    all_themes = [t for themes in df["key_themes"] for t in themes]
    if not all_themes:
        return None
    counts = Counter(all_themes).most_common(top_n)
    labels, values = zip(*counts)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(reversed(labels)), list(reversed(values)), color="#3498db", edgecolor="white")
    for i, val in enumerate(reversed(values)):
        ax.text(val + 0.05, i, str(val), va="center", fontsize=9, fontweight="bold")
    ax.set_title("Top Domain Themes", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Frequency")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig

def make_scatter_fig(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in PALETTE.items():
        sub = df[df["label"] == label]
        ax.scatter(sub["compound_score"], sub["subjectivity"],
                   c=color, label=label.capitalize(), alpha=0.75,
                   edgecolors="white", s=60)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Sentiment vs. Subjectivity", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Compound Score")
    ax.set_ylabel("Subjectivity")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def make_trend_fig(df, date_col):
    try:
        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        trend = temp.groupby(temp[date_col].dt.to_period("M"))["compound_score"].mean()
        trend.index = trend.index.to_timestamp()
        if len(trend) < 2:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(trend.index, trend.values, color="#2980b9", linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(trend.index, trend.values, 0,
                        where=(trend.values >= 0), alpha=0.15, color="#2ecc71")
        ax.fill_between(trend.index, trend.values, 0,
                        where=(trend.values < 0),  alpha=0.15, color="#e74c3c")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title("Avg. Sentiment Over Time", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Compound Score")
        ax.set_ylim(-1.05, 1.05)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔥 FirePulse")
    st.markdown("*Sentiment Analysis for Fire Departments*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔍  Analyze Text", "📂  Analyze CSV", "📊  Dashboard", "ℹ️  About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("NLTK · TextBlob · pandas\nmatplotlib · seaborn · Streamlit")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px;color:#bdc3c7;'>github.com/rohansreeram07<br>MIT License</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown("# 🔥 FirePulse")
    st.markdown("### NLP-Powered Sentiment Analysis for Fire Departments")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="result-card">
            <h4>🔍 Analyze Text</h4>
            <p>Instantly analyze a single incident report, survey response, or social media post for sentiment, subjectivity, and domain themes.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="result-card">
            <h4>📂 Analyze CSV</h4>
            <p>Upload a CSV file of incident reports or community feedback and get batch sentiment scores, theme detection, and flagged records.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="result-card">
            <h4>📊 Dashboard</h4>
            <p>Visualize sentiment distribution, monthly trend lines, subjectivity scatter, and domain theme frequency across your dataset.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🎯 Eight Firefighter Domain Themes")

    themes = {
        "🪖 Equipment":     "scba, hose, ladder, pump, apparatus, gear",
        "⚠️ Safety":        "hazard, mayday, injury, protocol, ppe",
        "⏱️ Response Time": "dispatch, delay, arrival, minutes, late",
        "📻 Communication": "command, radio, debrief, briefing, update",
        "💙 Morale":        "burnout, fatigue, stress, motivated, culture",
        "🎓 Training":      "drill, certification, exercise, simulation",
        "🏘️ Community":    "resident, public, feedback, complaint, grateful",
        "🚒 Incident":      "fire, rescue, hazmat, ems, wildfire, collapse",
    }

    col_a, col_b = st.columns(2)
    items = list(themes.items())
    for i, (theme, kws) in enumerate(items):
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(f"**{theme}**")
            st.caption(kws)

    st.markdown("""
    <div class="footer">
        FirePulse v1.0 · Built by Rohan Sreeram · MIT License
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: ANALYZE TEXT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍  Analyze Text":
    st.markdown('<div class="section-header">🔍 Single Text Analyzer</div>', unsafe_allow_html=True)
    st.markdown("Enter any incident report narrative, survey response, or social media post.")

    # Sample texts
    samples = {
        "— choose a sample —": "",
        "Positive report":    "The crew performed exceptionally during the structure fire. Coordination was excellent and all members followed protocol precisely. Equipment functioned without issues.",
        "Negative report":    "The SCBA gear failed again during training. This is the third incident this month. Morale is critically low because leadership keeps ignoring our equipment concerns.",
        "Community feedback": "The firefighters arrived within minutes and were incredibly professional. They saved our home and we are so grateful for their outstanding service.",
        "Mixed report":       "Night shift structure fire. The crew was fatigued from a prior call but executed the suppression professionally. Two areas flagged for procedure improvement.",
    }

    sample_choice = st.selectbox("Load a sample text (optional)", list(samples.keys()))
    default_text  = samples[sample_choice]

    text_input = st.text_area(
        "Enter text to analyze",
        value=default_text,
        height=160,
        placeholder="Paste incident report, community feedback, or social media post here...",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🔥 Analyze", type="primary", use_container_width=True)

    if analyze_clicked:
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                result = analyzer.analyze(text_input)

            st.markdown("---")
            st.markdown("#### Results")

            # Top metric row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sentiment",    result["label"].upper())
            m2.metric("Compound Score", f"{result['compound_score']:+.4f}")
            m3.metric("Subjectivity",  f"{result['subjectivity']:.4f}")
            m4.metric("Polarity",      f"{result['polarity']:+.4f}")

            st.markdown("---")
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("**Sentiment Breakdown**")
                label   = result["label"]
                color   = PALETTE[label]
                score   = result["compound_score"]
                pct     = int((score + 1) / 2 * 100)
                st.markdown(f"Label: {badge(label)}", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="margin-top:10px;">
                    <small>Score: {score:+.4f}</small>
                    <div class="score-bar-wrap">
                        <div class="score-bar" style="width:{pct}%; background:{color};"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:11px; color:#888;">
                        <span>-1.0 (Negative)</span><span>0 (Neutral)</span><span>+1.0 (Positive)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Component Scores**")
                comp_df = pd.DataFrame({
                    "Component": ["Positive", "Neutral", "Negative"],
                    "Score":     [result["positive"], result["neutral"], result["negative"]],
                })
                fig_comp, ax_comp = plt.subplots(figsize=(4, 2.5))
                bars = ax_comp.bar(comp_df["Component"], comp_df["Score"],
                                   color=["#2ecc71", "#95a5a6", "#e74c3c"], edgecolor="white")
                for bar, val in zip(bars, comp_df["Score"]):
                    ax_comp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
                ax_comp.set_ylim(0, 1.1)
                ax_comp.set_ylabel("Score")
                fig_comp.tight_layout()
                st.pyplot(fig_comp)
                plt.close(fig_comp)

            with col_r:
                st.markdown("**Detected Domain Themes**")
                themes_found = result["key_themes"]
                if themes_found:
                    for t in themes_found:
                        st.markdown(
                            f"<span style='background:#fadbd8;color:#c0392b;padding:4px 12px;"
                            f"border-radius:12px;font-size:13px;margin:3px;display:inline-block;'>"
                            f"🏷️ {t}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No specific domain themes detected.")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Top Keywords**")
                kws = result["top_keywords"]
                if kws:
                    kw_html = " ".join(
                        f"<span style='background:#d5e8f0;color:#2980b9;padding:4px 10px;"
                        f"border-radius:10px;font-size:13px;margin:2px;display:inline-block;'>{k}</span>"
                        for k in kws
                    )
                    st.markdown(kw_html, unsafe_allow_html=True)
                else:
                    st.info("No keywords extracted.")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Subjectivity Gauge**")
                subj = result["subjectivity"]
                subj_pct = int(subj * 100)
                subj_color = "#e67e22" if subj > 0.5 else "#27ae60"
                st.markdown(f"""
                <div>
                    <small>Score: {subj:.4f}  (0 = Objective, 1 = Subjective)</small>
                    <div class="score-bar-wrap">
                        <div class="score-bar" style="width:{subj_pct}%; background:{subj_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: ANALYZE CSV
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📂  Analyze CSV":
    st.markdown('<div class="section-header">📂 Batch CSV Analyzer</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file containing incident reports or community feedback for batch analysis.")

    with st.expander("📋 Expected CSV format"):
        st.markdown("""
        Your CSV should have at least one **text column** containing the records to analyze.
        An optional **ID column** can be used to label each record. Example:

        | report_id | date       | text                                      |
        |-----------|------------|-------------------------------------------|
        | RPT-001   | 2024-01-05 | Crew performed excellently during the ... |
        | RPT-002   | 2024-01-12 | Equipment failed again. Morale is low ... |
        """)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    # Use sample data if no file uploaded
    use_sample = st.checkbox("Use built-in sample data (incident reports)", value=not bool(uploaded))

    if use_sample and not uploaded:
        from firepulse import SAMPLE_REPORTS, SAMPLE_FEEDBACK
        data_choice = st.radio("Sample dataset", ["Incident Reports", "Community Feedback"], horizontal=True)
        if data_choice == "Incident Reports":
            raw_df = pd.DataFrame(SAMPLE_REPORTS)
        else:
            raw_df = pd.DataFrame(SAMPLE_FEEDBACK)
        st.info(f"Using built-in sample data: {len(raw_df)} records loaded.")
    elif uploaded:
        raw_df = pd.read_csv(uploaded)
        st.success(f"File uploaded: {len(raw_df)} records, {len(raw_df.columns)} columns.")
    else:
        raw_df = None

    if raw_df is not None:
        st.markdown("**Preview (first 5 rows)**")
        st.dataframe(raw_df.head(), use_container_width=True)

        text_col  = st.selectbox("Select the text column",  raw_df.columns.tolist())
        label_col = st.selectbox("Select the ID/label column (optional)",
                                 ["— none —"] + raw_df.columns.tolist())
        label_col = None if label_col == "— none —" else label_col

        flag_threshold = st.slider(
            "Flagged record threshold (compound score ≤ this value)",
            min_value=-1.0, max_value=0.0, value=-0.3, step=0.05,
        )

        if st.button("🔥 Run Analysis", type="primary"):
            with st.spinner(f"Analyzing {len(raw_df)} records..."):
                labels = raw_df[label_col].tolist() if label_col else None
                results_df = analyzer.analyze_batch(raw_df[text_col].fillna("").tolist(), labels)

                # Attach extra columns
                for col in raw_df.columns:
                    if col not in [text_col, label_col] and col not in results_df.columns:
                        results_df[col] = raw_df[col].values

                stats = analyzer.summary_stats(results_df)

            st.session_state["results_df"] = results_df
            st.session_state["stats"]      = stats

            st.markdown("---")
            st.markdown("#### Summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Records",  stats["total_records"])
            m2.metric("Avg. Score",     f"{stats['avg_compound']:+.4f}")
            m3.metric("Positive",       f"{stats['positive_count']} ({stats['positive_pct']}%)")
            m4.metric("Neutral",        stats["neutral_count"])
            m5.metric("Negative",       f"{stats['negative_count']} ({stats['negative_pct']}%)")

            # Charts side by side
            st.markdown("---")
            st.markdown("#### Visual Overview")
            c1, c2 = st.columns(2)
            with c1:
                fig_dist = make_distribution_fig(results_df)
                st.pyplot(fig_dist)
                plt.close(fig_dist)
            with c2:
                fig_theme = make_theme_fig(results_df)
                if fig_theme:
                    st.pyplot(fig_theme)
                    plt.close(fig_theme)

            # Results table
            st.markdown("---")
            st.markdown("#### Full Results")
            display_df = results_df.copy()
            for col in ["key_themes", "top_keywords"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: ", ".join(x) if isinstance(x, list) else x)

            def highlight_label(row):
                colors = {"positive": "background-color:#d5f5e3",
                          "negative": "background-color:#fadbd8",
                          "neutral":  "background-color:#f2f3f4"}
                return [colors.get(row.get("label", ""), "") if col == "label" else ""
                        for col in row.index]

            st.dataframe(
                display_df.style.apply(highlight_label, axis=1),
                use_container_width=True, height=400,
            )

            # Flagged records
            flagged = results_df[results_df["compound_score"] <= flag_threshold]
            if not flagged.empty:
                st.markdown(f"---")
                st.warning(f"⚠️ {len(flagged)} flagged record(s) with score ≤ {flag_threshold}")
                flag_display = flagged.copy()
                for col in ["key_themes", "top_keywords"]:
                    if col in flag_display.columns:
                        flag_display[col] = flag_display[col].apply(
                            lambda x: ", ".join(x) if isinstance(x, list) else x)
                st.dataframe(flag_display[["text", "label", "compound_score", "key_themes"]],
                             use_container_width=True)
            else:
                st.success(f"✅ No records flagged below threshold {flag_threshold}.")

            # Downloads
            st.markdown("---")
            st.markdown("#### Download Results")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Download Full Results (CSV)",
                    data=df_to_csv_bytes(results_df),
                    file_name="firepulse_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                if not flagged.empty:
                    st.download_button(
                        "⬇️ Download Flagged Records (CSV)",
                        data=df_to_csv_bytes(flagged),
                        file_name="firepulse_flagged.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊  Dashboard":
    st.markdown('<div class="section-header">📊 Sentiment Dashboard</div>', unsafe_allow_html=True)

    # Load from session state or fall back to sample data
    if "results_df" in st.session_state:
        df    = st.session_state["results_df"]
        stats = st.session_state["stats"]
        st.info(f"Showing results from your last CSV analysis — {len(df)} records.")
    else:
        from firepulse import SAMPLE_REPORTS
        st.info("No CSV analysis run yet — showing built-in sample data (15 incident reports).")
        raw = pd.DataFrame(SAMPLE_REPORTS)
        df  = analyzer.analyze_batch(raw["text"].tolist(), raw["id"].tolist())
        df["date"] = raw["date"].values
        stats = analyzer.summary_stats(df)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown("#### Key Metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Records",  stats["total_records"])
    k2.metric("Avg. Compound",  f"{stats['avg_compound']:+.4f}",
              delta="Positive" if stats["avg_compound"] > 0.05 else ("Negative" if stats["avg_compound"] < -0.05 else "Neutral"))
    k3.metric("Positive",       f"{stats['positive_count']} ({stats['positive_pct']}%)")
    k4.metric("Neutral",        stats["neutral_count"])
    k5.metric("Negative",       f"{stats['negative_count']} ({stats['negative_pct']}%)")

    st.markdown("---")

    # ── Row 1: Distribution + Themes ─────────────────────────────────────────
    st.markdown("#### Sentiment Breakdown")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig_d = make_distribution_fig(df)
        st.pyplot(fig_d)
        st.download_button("⬇️ Save Chart", render_chart(fig_d), "distribution.png", "image/png")
        plt.close(fig_d)

    with r1c2:
        fig_t = make_theme_fig(df)
        if fig_t:
            st.pyplot(fig_t)
            st.download_button("⬇️ Save Chart", render_chart(fig_t), "themes.png", "image/png")
            plt.close(fig_t)
        else:
            st.info("No themes detected in dataset.")

    st.markdown("---")

    # ── Row 2: Scatter + Trend/Histogram ─────────────────────────────────────
    st.markdown("#### Detailed Analysis")
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig_s = make_scatter_fig(df)
        st.pyplot(fig_s)
        st.download_button("⬇️ Save Chart", render_chart(fig_s), "scatter.png", "image/png")
        plt.close(fig_s)

    with r2c2:
        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if date_cols:
            date_col  = st.selectbox("Date column for trend chart", date_cols)
            fig_trend = make_trend_fig(df, date_col)
            if fig_trend:
                st.pyplot(fig_trend)
                st.download_button("⬇️ Save Chart", render_chart(fig_trend), "trend.png", "image/png")
                plt.close(fig_trend)
            else:
                st.info("Not enough monthly data points to render a trend line.")
        else:
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            ax_hist.hist(df["compound_score"], bins=15, color="#9b59b6", edgecolor="white")
            ax_hist.axvline(0, color="gray", linestyle="--", alpha=0.5)
            ax_hist.set_title("Compound Score Distribution", fontsize=13, fontweight="bold")
            ax_hist.set_xlabel("Compound Score")
            ax_hist.set_ylabel("Count")
            fig_hist.tight_layout()
            st.pyplot(fig_hist)
            st.download_button("⬇️ Save Chart", render_chart(fig_hist), "histogram.png", "image/png")
            plt.close(fig_hist)

    st.markdown("---")

    # ── Top themes summary ────────────────────────────────────────────────────
    st.markdown("#### Top Domain Themes")
    if stats["top_themes"]:
        theme_cols = st.columns(len(stats["top_themes"]))
        for col, (theme, count) in zip(theme_cols, stats["top_themes"]):
            col.metric(theme.replace("_", " ").title(), count)

    # ── Download summary report ───────────────────────────────────────────────
    st.markdown("---")
    summary_lines = exporter._build_report(
        "FirePulse Dashboard Report",
        datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        stats,
    )
    st.download_button(
        "⬇️ Download Text Summary Report",
        data="\n".join(summary_lines).encode("utf-8"),
        file_name="firepulse_summary.txt",
        mime="text/plain",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️  About":
    st.markdown('<div class="section-header">ℹ️ About FirePulse</div>', unsafe_allow_html=True)

    st.markdown("""
    **FirePulse** is an open-source NLP-powered sentiment analysis tool built for fire departments.
    It analyzes unstructured text — incident reports, after-action reviews, community feedback surveys,
    and social media mentions — to surface morale trends and community satisfaction patterns for
    department leadership.
    """)

    st.markdown("---")
    st.markdown("#### How It Works")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **1. Sentiment Scoring (VADER)**
        Each text is scored using NLTK's VADER model — a rule-based NLP tool calibrated for
        real-world, informal text. It returns a compound score from -1.0 (most negative)
        to +1.0 (most positive).

        **2. Subjectivity (TextBlob)**
        TextBlob assigns a subjectivity score from 0 (completely objective) to 1
        (completely subjective), helping identify opinion-heavy records vs. factual reports.
        """)
    with col2:
        st.markdown("""
        **3. Theme Detection**
        A keyword taxonomy maps each record to one or more of eight firefighter-specific
        domain themes: equipment, safety, response time, communication, morale, training,
        community, and incident.

        **4. Visualization & Export**
        Results are visualized as distribution charts, trend lines, scatter plots, and
        full dashboards. All results are downloadable as CSV and plain-text reports.
        """)

    st.markdown("---")
    st.markdown("#### Developer")
    st.markdown("""
    Built by **Rohan Sreeram**, Computer Science student at the University of Maryland College Park
    (GPA: 3.79 | President Scholarship Recipient).

    Certified in Databricks Generative AI, Python (PCEP), and FEMA Emergency Management.
    Active volunteer firefighter at The Singerly Fire Company — bringing both technical
    rigor and genuine domain expertise to this project.

    🔗 [github.com/rohansreeram07](https://github.com/rohansreeram07)
    """)

    st.markdown("---")
    st.markdown("#### License")
    st.info("MIT License — open for use, modification, and contribution.")
