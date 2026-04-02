"""
╔══════════════════════════════════════════════════════════════════╗
║                         FirePulse v1.0                          ║
║       NLP Sentiment Analysis for Fire Departments               ║
║                                                                  ║
║  Author  : Rohan Sreeram                                         ║
║  GitHub  : github.com/rohansreeram07/firepulse                  ║
║  License : MIT                                                   ║
║                                                                  ║
║  Stack   : NLTK · TextBlob · pandas · matplotlib · seaborn      ║
╚══════════════════════════════════════════════════════════════════╝

Install dependencies:
    pip install nltk textblob pandas matplotlib seaborn

Run:
    python firepulse.py
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re
import os
import csv
import datetime
import warnings
from pathlib import Path
from collections import Counter
from typing import Optional

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)


# ─────────────────────────────────────────────────────────────────────────────
#  NLTK SETUP  (auto-downloads on first run)
# ─────────────────────────────────────────────────────────────────────────────
def _setup_nltk():
    resources = [
        ("vader_lexicon", "sentiment/vader_lexicon.zip"),
        ("stopwords",     "corpora/stopwords.zip"),
        ("punkt",         "tokenizers/punkt.zip"),
        ("punkt_tab",     "tokenizers/punkt_tab.zip"),
    ]
    for name, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"  [setup] Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)

_setup_nltk()


# ─────────────────────────────────────────────────────────────────────────────
#  DOMAIN THEME TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────
FIREFIGHTER_THEMES = {
    "equipment":     ["equipment", "gear", "scba", "hose", "truck", "apparatus",
                      "ladder", "pump", "radio", "mask", "turnout", "tools"],
    "safety":        ["safety", "hazard", "injury", "risk", "protocol", "procedure",
                      "mayday", "danger", "exposure", "nfpa", "ppe"],
    "response_time": ["response", "arrival", "dispatch", "delay", "time", "minutes",
                      "late", "fast", "quick", "slow"],
    "communication": ["communication", "command", "radio", "report", "debrief",
                      "briefing", "coordination", "information", "update"],
    "morale":        ["morale", "stress", "burnout", "fatigue", "exhausted",
                      "motivated", "support", "team", "culture", "mental"],
    "training":      ["training", "drill", "exercise", "certification", "practice",
                      "simulation", "qualified", "skills", "competency"],
    "community":     ["community", "public", "resident", "citizen", "neighborhood",
                      "family", "victim", "grateful", "complaint", "feedback"],
    "incident":      ["fire", "rescue", "medical", "ems", "structure", "wildfire",
                      "accident", "flood", "hazmat", "collapse"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS: SentimentAnalyzer
# ─────────────────────────────────────────────────────────────────────────────
class SentimentAnalyzer:
    """
    Core FirePulse engine.

    Analyzes incident reports and community feedback using VADER for sentiment
    scoring, TextBlob for subjectivity, and a keyword taxonomy for theme detection.

    Usage
    -----
    >>> analyzer = SentimentAnalyzer()
    >>> result = analyzer.analyze("The SCBA equipment failed mid-operation. Morale is low.")
    >>> print(result["label"], result["compound_score"])
    negative -0.6249
    """

    def __init__(self, theme_taxonomy: Optional[dict] = None):
        self._vader      = SentimentIntensityAnalyzer()
        self._stop_words = set(stopwords.words("english"))
        self._themes     = theme_taxonomy or FIREFIGHTER_THEMES

    # ── Single text ──────────────────────────────────────────────────────────

    def analyze(self, text: str) -> dict:
        """
        Analyze a single string.

        Returns
        -------
        dict:
            text, compound_score, label, positive, negative, neutral,
            subjectivity, polarity, key_themes, top_keywords
        """
        if not isinstance(text, str) or not text.strip():
            return self._empty_result(text)

        clean          = self._clean(text)
        vader          = self._vader.polarity_scores(clean)
        blob           = TextBlob(clean)

        return {
            "text":           text,
            "compound_score": round(vader["compound"], 4),
            "label":          self._label(vader["compound"]),
            "positive":       round(vader["pos"], 4),
            "negative":       round(vader["neg"], 4),
            "neutral":        round(vader["neu"], 4),
            "subjectivity":   round(blob.sentiment.subjectivity, 4),
            "polarity":       round(blob.sentiment.polarity, 4),
            "key_themes":     self._detect_themes(clean),
            "top_keywords":   self._top_keywords(clean),
        }

    # ── Batch list ───────────────────────────────────────────────────────────

    def analyze_batch(self, texts: list, source_labels: Optional[list] = None) -> pd.DataFrame:
        """
        Analyze a list of strings. Returns a DataFrame.

        Parameters
        ----------
        texts         : list of str
        source_labels : optional list of identifiers (e.g. report IDs)
        """
        results = [self.analyze(t) for t in texts]
        df = pd.DataFrame(results)
        if source_labels:
            if len(source_labels) != len(df):
                raise ValueError("source_labels length must match texts length.")
            df.insert(0, "source", source_labels)
        return df

    # ── CSV file ─────────────────────────────────────────────────────────────

    def analyze_csv(
        self,
        filepath: str,
        text_column: str,
        label_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a CSV and analyze a text column.

        Parameters
        ----------
        filepath     : path to CSV
        text_column  : column containing the text to analyze
        label_column : optional column to use as source identifier
        """
        df = pd.read_csv(filepath)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")

        labels     = df[label_column].tolist() if label_column and label_column in df.columns else None
        results_df = self.analyze_batch(df[text_column].fillna("").tolist(), labels)

        # Attach extra original columns
        extra = [c for c in df.columns if c not in [text_column, label_column]]
        for col in extra:
            results_df[col] = df[col].values

        return results_df

    # ── Summary stats ────────────────────────────────────────────────────────

    def summary_stats(self, df: pd.DataFrame) -> dict:
        """
        Compute summary statistics from an analysis DataFrame.

        Returns counts, averages, percentages, and top themes.
        """
        if "compound_score" not in df.columns:
            raise ValueError("DataFrame must contain 'compound_score'.")

        label_counts = df["label"].value_counts().to_dict()
        all_themes   = [t for themes in df["key_themes"] for t in themes]
        top_themes   = Counter(all_themes).most_common(5)
        total        = len(df)

        return {
            "total_records":    total,
            "avg_compound":     round(df["compound_score"].mean(), 4),
            "avg_subjectivity": round(df["subjectivity"].mean(), 4),
            "positive_count":   label_counts.get("positive", 0),
            "neutral_count":    label_counts.get("neutral", 0),
            "negative_count":   label_counts.get("negative", 0),
            "positive_pct":     round(label_counts.get("positive", 0) / total * 100, 1),
            "negative_pct":     round(label_counts.get("negative", 0) / total * 100, 1),
            "top_themes":       top_themes,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _label(compound: float) -> str:
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        return "neutral"

    def _detect_themes(self, text: str) -> list:
        tokens = set(text.split())
        return [theme for theme, kws in self._themes.items() if tokens.intersection(kws)]

    def _top_keywords(self, text: str, n: int = 5) -> list:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        filtered = [t for t in tokens if t.isalpha() and t not in self._stop_words and len(t) > 2]
        return [w for w, _ in Counter(filtered).most_common(n)]

    @staticmethod
    def _empty_result(text) -> dict:
        return {
            "text": text, "compound_score": 0.0, "label": "neutral",
            "positive": 0.0, "negative": 0.0, "neutral": 1.0,
            "subjectivity": 0.0, "polarity": 0.0,
            "key_themes": [], "top_keywords": [],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS: SentimentVisualizer
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}


class SentimentVisualizer:
    """
    Generate charts from FirePulse analysis DataFrames.

    Usage
    -----
    >>> viz = SentimentVisualizer(output_dir="charts/")
    >>> viz.plot_dashboard(df, date_column="date")
    """

    def __init__(self, output_dir: str = "charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Distribution bar ─────────────────────────────────────────────────────

    def plot_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Sentiment Distribution",
        save_as: Optional[str] = "distribution.png",
    ) -> plt.Figure:
        """Bar chart: count of positive / neutral / negative records."""
        self._check(df, "label")
        counts = df["label"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
        colors = [PALETTE[l] for l in counts.index]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(val), ha="center", va="bottom", fontweight="bold")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        return self._save(fig, save_as)

    # ── Trend line ───────────────────────────────────────────────────────────

    def plot_trends(
        self,
        df: pd.DataFrame,
        date_column: str,
        title: str = "Sentiment Trend Over Time",
        save_as: Optional[str] = "trends.png",
    ) -> plt.Figure:
        """Line chart: average compound score per month."""
        self._check(df, "compound_score")
        self._check(df, date_column)

        temp = df.copy()
        temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
        trend = temp.groupby(temp[date_column].dt.to_period("M"))["compound_score"].mean()
        trend.index = trend.index.to_timestamp()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(trend.index, trend.values, color="#2980b9", linewidth=2.5, marker="o", markersize=5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Neutral baseline")
        ax.fill_between(trend.index, trend.values, 0,
                        where=(trend.values >= 0), alpha=0.15, color="#2ecc71", label="Positive zone")
        ax.fill_between(trend.index, trend.values, 0,
                        where=(trend.values < 0),  alpha=0.15, color="#e74c3c", label="Negative zone")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Avg. Compound Score")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        return self._save(fig, save_as)

    # ── Scatter ──────────────────────────────────────────────────────────────

    def plot_subjectivity_scatter(
        self,
        df: pd.DataFrame,
        title: str = "Sentiment vs. Subjectivity",
        save_as: Optional[str] = "scatter.png",
    ) -> plt.Figure:
        """Scatter: compound score (x) vs. subjectivity (y), colored by label."""
        self._check(df, "compound_score")
        self._check(df, "subjectivity")
        self._check(df, "label")

        fig, ax = plt.subplots(figsize=(8, 6))
        for label, color in PALETTE.items():
            sub = df[df["label"] == label]
            ax.scatter(sub["compound_score"], sub["subjectivity"],
                       c=color, label=label.capitalize(), alpha=0.75,
                       edgecolors="white", s=65)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Compound Sentiment Score")
        ax.set_ylabel("Subjectivity  (0 = Objective,  1 = Subjective)")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="Label")
        fig.tight_layout()
        return self._save(fig, save_as)

    # ── Theme frequency ──────────────────────────────────────────────────────

    def plot_theme_frequency(
        self,
        df: pd.DataFrame,
        title: str = "Top Domain Themes",
        top_n: int = 8,
        save_as: Optional[str] = "themes.png",
    ) -> Optional[plt.Figure]:
        """Horizontal bar: most frequently detected domain themes."""
        self._check(df, "key_themes")
        all_themes = [t for themes in df["key_themes"] for t in themes]
        if not all_themes:
            print("  [viz] No themes detected — skipping chart.")
            return None

        counts = Counter(all_themes).most_common(top_n)
        labels, values = zip(*counts)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(list(reversed(labels)), list(reversed(values)),
                       color="#3498db", edgecolor="white")
        for bar, val in zip(bars, reversed(values)):
            ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Frequency")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        return self._save(fig, save_as)

    # ── 2×2 Dashboard ────────────────────────────────────────────────────────

    def plot_dashboard(
        self,
        df: pd.DataFrame,
        date_column: Optional[str] = None,
        title: str = "FirePulse — Sentiment Dashboard",
        save_as: Optional[str] = "dashboard.png",
    ) -> plt.Figure:
        """2×2 summary dashboard: distribution, scatter, themes, trend/histogram."""
        has_date = date_column and date_column in df.columns
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

        # Panel 1: Sentiment distribution
        ax1 = fig.add_subplot(2, 2, 1)
        counts = df["label"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
        bars = ax1.bar(counts.index, counts.values,
                       color=[PALETTE[l] for l in counts.index], edgecolor="white")
        for bar, val in zip(bars, counts.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax1.set_title("Sentiment Distribution")
        ax1.set_ylabel("Count")
        ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Panel 2: Subjectivity scatter
        ax2 = fig.add_subplot(2, 2, 2)
        for label, color in PALETTE.items():
            sub = df[df["label"] == label]
            ax2.scatter(sub["compound_score"], sub["subjectivity"],
                        c=color, label=label.capitalize(), alpha=0.7, s=40, edgecolors="white")
        ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_title("Sentiment vs. Subjectivity")
        ax2.set_xlabel("Compound Score")
        ax2.set_ylabel("Subjectivity")
        ax2.legend(fontsize=8)

        # Panel 3: Theme frequency
        ax3 = fig.add_subplot(2, 2, 3)
        all_themes = [t for themes in df["key_themes"] for t in themes]
        if all_themes:
            top = Counter(all_themes).most_common(6)
            tlabels, tvals = zip(*top)
            ax3.barh(list(reversed(tlabels)), list(reversed(tvals)),
                     color="#3498db", edgecolor="white")
            ax3.set_title("Top Domain Themes")
            ax3.set_xlabel("Frequency")
            ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        else:
            ax3.text(0.5, 0.5, "No themes detected", ha="center", va="center",
                     transform=ax3.transAxes, color="gray")
            ax3.set_title("Top Domain Themes")

        # Panel 4: Trend line or compound score histogram
        ax4 = fig.add_subplot(2, 2, 4)
        if has_date:
            temp = df.copy()
            temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
            trend = temp.groupby(temp[date_column].dt.to_period("M"))["compound_score"].mean()
            trend.index = trend.index.to_timestamp()
            ax4.plot(trend.index, trend.values, color="#2980b9", linewidth=2, marker="o", markersize=4)
            ax4.fill_between(trend.index, trend.values, 0,
                             where=(trend.values >= 0), alpha=0.12, color="#2ecc71")
            ax4.fill_between(trend.index, trend.values, 0,
                             where=(trend.values < 0),  alpha=0.12, color="#e74c3c")
            ax4.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax4.set_title("Avg. Sentiment Over Time")
            ax4.set_ylabel("Compound Score")
            ax4.set_ylim(-1.05, 1.05)
            fig.autofmt_xdate()
        else:
            ax4.hist(df["compound_score"], bins=20, color="#9b59b6", edgecolor="white")
            ax4.axvline(0, color="gray", linestyle="--", alpha=0.5)
            ax4.set_title("Compound Score Distribution")
            ax4.set_xlabel("Compound Score")
            ax4.set_ylabel("Count")

        fig.tight_layout()
        return self._save(fig, save_as)

    # ── Private ──────────────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, save_as: Optional[str]) -> plt.Figure:
        if save_as:
            path = self.output_dir / save_as
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  [chart] Saved → {path}")
        else:
            plt.show()
        plt.close(fig)
        return fig

    @staticmethod
    def _check(df: pd.DataFrame, col: str):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS: SentimentExporter
# ─────────────────────────────────────────────────────────────────────────────
class SentimentExporter:
    """
    Export FirePulse results to CSV and plain-text summary reports.

    Usage
    -----
    >>> exporter = SentimentExporter(output_dir="reports/")
    >>> exporter.to_csv(df, "results.csv")
    >>> exporter.to_summary(df, stats, "summary.txt")
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_csv(self, df: pd.DataFrame, filename: str = "firepulse_results.csv",
               include_text: bool = True) -> Path:
        """Export full results to CSV."""
        export = df.copy()
        for col in ["key_themes", "top_keywords"]:
            if col in export.columns:
                export[col] = export[col].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else x)
        if not include_text and "text" in export.columns:
            export = export.drop(columns=["text"])
        path = self.output_dir / filename
        export.to_csv(path, index=False)
        print(f"  [export] Results saved → {path}")
        return path

    def to_flagged_csv(self, df: pd.DataFrame, threshold: float = -0.4,
                       filename: str = "flagged_records.csv") -> Path:
        """Export only records with compound_score <= threshold."""
        if "compound_score" not in df.columns:
            raise ValueError("DataFrame must contain 'compound_score'.")
        flagged = df[df["compound_score"] <= threshold].sort_values("compound_score")
        count = len(flagged)
        if count == 0:
            print(f"  [export] No records below threshold {threshold}.")
        else:
            print(f"  [export] {count} flagged records (score ≤ {threshold}) found.")
        return self.to_csv(flagged, filename)

    def to_summary(self, df: pd.DataFrame, stats: dict, filename: str = "summary.txt",
                   report_title: str = "FirePulse Sentiment Report") -> Path:
        """Generate a plain-text narrative summary report."""
        now   = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
        lines = self._build_report(report_title, now, stats)
        path  = self.output_dir / filename
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [export] Summary saved → {path}")
        return path

    @staticmethod
    def _build_report(title: str, timestamp: str, stats: dict) -> list:
        total = stats.get("total_records", 0)
        sep   = "=" * 60
        lines = [
            sep,
            f"  {title}",
            f"  Generated: {timestamp}",
            sep, "",
            "OVERVIEW",      "-" * 40,
            f"  Total records    : {total}",
            f"  Avg. compound    : {stats.get('avg_compound', 0):+.4f}",
            f"  Avg. subjectivity: {stats.get('avg_subjectivity', 0):.4f}",
            "",
            "SENTIMENT BREAKDOWN", "-" * 40,
            f"  Positive : {stats.get('positive_count', 0):>4}  ({stats.get('positive_pct', 0):.1f}%)",
            f"  Neutral  : {stats.get('neutral_count', 0):>4}",
            f"  Negative : {stats.get('negative_count', 0):>4}  ({stats.get('negative_pct', 0):.1f}%)",
            "",
            "TOP THEMES", "-" * 40,
        ]
        for theme, count in stats.get("top_themes", []):
            lines.append(f"  {theme:<20} {count} mention(s)")

        avg_c = stats.get("avg_compound", 0)
        lines += ["", "INTERPRETATION", "-" * 40]
        if avg_c > 0.2:
            lines.append("  Overall sentiment is POSITIVE. Morale and community feedback")
            lines.append("  appear favorable across this reporting period.")
        elif avg_c < -0.2:
            lines.append("  Overall sentiment is NEGATIVE. Review flagged records for")
            lines.append("  recurring concerns that require leadership attention.")
        else:
            lines.append("  Overall sentiment is NEUTRAL. No strong signal detected.")
            lines.append("  Monitor trends over subsequent reporting periods.")

        lines += ["", sep, "  End of Report — FirePulse v1.0", sep]
        return lines


# ─────────────────────────────────────────────────────────────────────────────
#  SAMPLE DATA  (embedded so the script is fully self-contained)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_REPORTS = [
    {"id": "RPT-001", "date": "2024-01-05", "text": "Crew performed exceptionally during the structure fire. Coordination was excellent and all members followed protocol precisely."},
    {"id": "RPT-002", "date": "2024-01-12", "text": "The SCBA gear failed again during training today. This is the third incident this month. Morale is low because leadership keeps ignoring our equipment concerns."},
    {"id": "RPT-003", "date": "2024-01-19", "text": "Response to the vehicle rescue went well. The new extraction tools made a significant difference. Team was focused and professional under pressure."},
    {"id": "RPT-004", "date": "2024-02-03", "text": "Dispatch was extremely slow to respond to our mayday call. By the time we got backup the situation had escalated. Communication protocols need urgent review."},
    {"id": "RPT-005", "date": "2024-02-10", "text": "After-action review for the warehouse fire was productive. The team handled a complex scenario with composure and good teamwork."},
    {"id": "RPT-006", "date": "2024-02-17", "text": "Exhausted after back-to-back 24-hour shifts. The overtime is unsustainable and I am worried about fatigue affecting our safety on scene."},
    {"id": "RPT-007", "date": "2024-03-01", "text": "Hazmat incident on Route 40 was managed professionally. The crew followed NFPA 1072 protocols perfectly and there were no injuries."},
    {"id": "RPT-008", "date": "2024-03-15", "text": "The pump on Engine 3 failed mid-operation. We had to retreat and call for mutual aid. We reported this mechanical issue two weeks ago."},
    {"id": "RPT-009", "date": "2024-03-22", "text": "Quarterly training drill exceeded expectations. New volunteers showed strong skill development and veteran members provided excellent mentorship."},
    {"id": "RPT-010", "date": "2024-04-02", "text": "Medical call on Oak Street was routine. Patient was stabilized and transported without complications. EMS coordination was smooth."},
    {"id": "RPT-011", "date": "2024-04-14", "text": "Flood response was overwhelming. We were understaffed and resources were stretched thin. Residents were frustrated by the delayed response time."},
    {"id": "RPT-012", "date": "2024-04-28", "text": "Wildfire containment efforts were well-coordinated with neighboring departments. The mutual aid agreement worked exactly as designed."},
    {"id": "RPT-013", "date": "2024-05-06", "text": "Another radio failure in the field. If we cannot communicate reliably we are putting lives at risk. Three maintenance requests have been ignored."},
    {"id": "RPT-014", "date": "2024-05-19", "text": "Community open house was a success. Residents asked thoughtful questions and our public education team did a fantastic job."},
    {"id": "RPT-015", "date": "2024-06-03", "text": "Night shift structure fire. The crew was fatigued from a prior call but executed the suppression professionally. Two areas flagged for procedure improvement."},
]

SAMPLE_FEEDBACK = [
    {"id": "FB-001", "date": "2024-01-08", "channel": "survey",       "text": "The firefighters arrived within minutes and were incredibly professional. They saved our home and we are so grateful for their service."},
    {"id": "FB-002", "date": "2024-01-15", "channel": "social_media", "text": "It took over 20 minutes for a fire truck to show up. By the time they arrived the car was completely destroyed. Very disappointed."},
    {"id": "FB-003", "date": "2024-01-22", "channel": "survey",       "text": "The crew was respectful and kept us informed the whole time. Excellent communication and they made a scary situation feel manageable."},
    {"id": "FB-004", "date": "2024-02-05", "channel": "social_media", "text": "Firefighters blocked the entire road for 3 hours without any updates to neighbors. Poor community communication."},
    {"id": "FB-005", "date": "2024-02-14", "channel": "survey",       "text": "Outstanding response to the apartment fire. The team was fast, organized, and compassionate with the displaced families. Real heroes."},
    {"id": "FB-006", "date": "2024-02-20", "channel": "web_chat",     "text": "I called the non-emergency line and no one answered for 15 minutes. When someone finally picked up they were unhelpful and rude."},
    {"id": "FB-007", "date": "2024-03-04", "channel": "survey",       "text": "The fire station open day was wonderful. My kids loved meeting the firefighters and learning about fire safety. Great community event."},
    {"id": "FB-008", "date": "2024-03-18", "channel": "social_media", "text": "Response time has gotten worse every year. The department needs more funding and more staff. We pay taxes and deserve better service."},
    {"id": "FB-009", "date": "2024-03-25", "channel": "survey",       "text": "Quick response, professional conduct, and they even followed up the next day to check on us. Above and beyond service."},
    {"id": "FB-010", "date": "2024-04-07", "channel": "web_chat",     "text": "The firefighters who responded to our medical emergency were kind and skilled. They kept my father calm and got him to the hospital safely."},
    {"id": "FB-011", "date": "2024-04-20", "channel": "social_media", "text": "Three different trucks drove past the wrong address before finding us. The dispatch system clearly has problems. Almost cost us everything."},
    {"id": "FB-012", "date": "2024-05-01", "channel": "survey",       "text": "I was impressed by how well-equipped and trained the crew seemed. They handled a difficult chimney fire without any damage to the rest of our house."},
    {"id": "FB-013", "date": "2024-05-12", "channel": "web_chat",     "text": "Firefighters were rude to my elderly mother during the incident. She was already scared and their attitude made things worse."},
    {"id": "FB-014", "date": "2024-05-27", "channel": "survey",       "text": "Annual inspection was thorough and the firefighter who visited was educational and friendly. Really appreciate the proactive outreach."},
    {"id": "FB-015", "date": "2024-06-10", "channel": "social_media", "text": "Incredible work during last week's flooding. Crews worked through the night to help residents. Our community is lucky to have them."},
]


def _write_sample_csvs(data_dir: str = "data") -> tuple:
    """Write embedded sample data to CSV files and return their paths."""
    Path(data_dir).mkdir(exist_ok=True)
    reports_path  = os.path.join(data_dir, "sample_reports.csv")
    feedback_path = os.path.join(data_dir, "sample_feedback.csv")

    with open(reports_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "date", "text"])
        writer.writeheader()
        writer.writerows(SAMPLE_REPORTS)

    with open(feedback_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "date", "channel", "text"])
        writer.writeheader()
        writer.writerows(SAMPLE_FEEDBACK)

    return reports_path, feedback_path


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────
def main():
    SEP = "=" * 60

    print(f"\n{SEP}")
    print("  🔥  FirePulse v1.0  —  Sentiment Analysis Demo")
    print(f"{SEP}\n")

    analyzer = SentimentAnalyzer()
    viz      = SentimentVisualizer(output_dir="charts")
    exporter = SentimentExporter(output_dir="reports")

    # ── 1. Single text analysis ───────────────────────────────────────────────
    print("1. SINGLE TEXT ANALYSIS")
    print("-" * 40)
    sample = (
        "The SCBA equipment failed again mid-operation. "
        "Morale is critically low and the team feels leadership "
        "is ignoring repeated safety concerns."
    )
    result = analyzer.analyze(sample)
    print(f"  Text     : \"{sample[:65]}...\"")
    print(f"  Label    : {result['label'].upper()}")
    print(f"  Score    : {result['compound_score']:+.4f}")
    print(f"  Subj.    : {result['subjectivity']:.4f}")
    print(f"  Themes   : {', '.join(result['key_themes']) or 'none'}")
    print(f"  Keywords : {', '.join(result['top_keywords'])}")

    # ── 2. Batch analysis ─────────────────────────────────────────────────────
    print(f"\n2. BATCH ANALYSIS")
    print("-" * 40)
    texts = [
        "Outstanding response. Team coordination was flawless and equipment worked perfectly.",
        "Another radio failure in the field. This is a serious safety hazard.",
        "Routine medical call. Patient stabilized and transported without complications.",
        "Community feedback was overwhelmingly positive after last week's open house.",
        "Dispatch delays put lives at risk. We need immediate improvements.",
    ]
    ids = [f"RPT-{str(i+1).zfill(3)}" for i in range(len(texts))]
    batch_df = analyzer.analyze_batch(texts, source_labels=ids)
    print(batch_df[["source", "label", "compound_score", "key_themes"]].to_string(index=False))

    # ── 3. CSV analysis — incident reports ────────────────────────────────────
    print(f"\n3. INCIDENT REPORT ANALYSIS  (15 records)")
    print("-" * 40)
    reports_path, feedback_path = _write_sample_csvs()
    reports_df = analyzer.analyze_csv(reports_path, text_column="text", label_column="id")
    print(f"  Loaded {len(reports_df)} incident reports.")
    print(reports_df[["source", "label", "compound_score", "key_themes"]].to_string(index=False))

    # ── 4. CSV analysis — community feedback ──────────────────────────────────
    print(f"\n4. COMMUNITY FEEDBACK ANALYSIS  (15 records)")
    print("-" * 40)
    feedback_df = analyzer.analyze_csv(feedback_path, text_column="text", label_column="id")
    print(f"  Loaded {len(feedback_df)} feedback entries.")
    print(feedback_df[["source", "label", "compound_score", "channel"]].to_string(index=False))

    # ── 5. Summary statistics ─────────────────────────────────────────────────
    print(f"\n5. SUMMARY STATISTICS  —  Incident Reports")
    print("-" * 40)
    stats = analyzer.summary_stats(reports_df)
    print(f"  Total records     : {stats['total_records']}")
    print(f"  Avg. compound     : {stats['avg_compound']:+.4f}")
    print(f"  Avg. subjectivity : {stats['avg_subjectivity']:.4f}")
    print(f"  Positive          : {stats['positive_count']}  ({stats['positive_pct']}%)")
    print(f"  Neutral           : {stats['neutral_count']}")
    print(f"  Negative          : {stats['negative_count']}  ({stats['negative_pct']}%)")
    print(f"  Top themes        : {[t for t, _ in stats['top_themes']]}")

    # ── 6. Charts ─────────────────────────────────────────────────────────────
    print(f"\n6. GENERATING CHARTS  →  charts/")
    print("-" * 40)
    # Attach date column for trend chart
    reports_df["date"] = [r["date"] for r in SAMPLE_REPORTS]
    viz.plot_distribution(reports_df,            title="Incident Report Sentiment")
    viz.plot_trends(reports_df, "date",          title="Monthly Sentiment Trend — Incident Reports")
    viz.plot_subjectivity_scatter(reports_df,    title="Sentiment vs. Subjectivity")
    viz.plot_theme_frequency(reports_df,         title="Top Firefighter Domain Themes")
    viz.plot_dashboard(reports_df, "date",       title="FirePulse — Incident Reports Dashboard")

    # Community feedback dashboard
    feedback_df["date"] = [r["date"] for r in SAMPLE_FEEDBACK]
    viz.plot_dashboard(feedback_df, "date",
                       title="FirePulse — Community Feedback Dashboard",
                       save_as="dashboard_feedback.png")

    # ── 7. Export ─────────────────────────────────────────────────────────────
    print(f"\n7. EXPORTING RESULTS  →  reports/")
    print("-" * 40)
    exporter.to_csv(reports_df,   filename="incident_reports_analysis.csv")
    exporter.to_csv(feedback_df,  filename="community_feedback_analysis.csv")
    exporter.to_flagged_csv(reports_df,  threshold=-0.3, filename="flagged_reports.csv")
    exporter.to_flagged_csv(feedback_df, threshold=-0.3, filename="flagged_feedback.csv")
    exporter.to_summary(reports_df, stats, filename="incident_report_summary.txt",
                        report_title="Q1-Q2 2024 Incident Report Sentiment Analysis")
    feedback_stats = analyzer.summary_stats(feedback_df)
    exporter.to_summary(feedback_df, feedback_stats, filename="community_feedback_summary.txt",
                        report_title="Q1-Q2 2024 Community Feedback Sentiment Analysis")

    print(f"\n{SEP}")
    print("  FirePulse demo complete.")
    print("  Charts  → charts/")
    print("  Reports → reports/")
    print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
