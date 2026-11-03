"""
core/visual_aid_generator.py
Parses a trailing [CHART: ...] tag emitted by the LLM (see config.SYSTEM_PROMPT)
and renders it as a PNG with matplotlib — no extra model needed, fast enough
to stay inside the latency budget.

Covers the "visual aids (charts, diagrams)" requirement from the CMTI
problem statement: bar/line/pie for numeric comparisons, flow for simple
step-by-step process diagrams.
"""
import base64
import io
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CHART_TAG_RE = re.compile(
    r'\[CHART:\s*type=(?P<type>bar|line|pie|flow)\s*;\s*'
    r'title="(?P<title>[^"]*)"\s*;\s*'
    r'labels=(?P<labels>[^;\]]+)'
    r'(?:;\s*values=(?P<values>[^\]]+))?\s*\]',
    re.IGNORECASE,
)


def extract_chart_tag(answer_text: str) -> Tuple[str, Optional[dict]]:
    """
    Splits `answer_text` into (clean_text, chart_spec_or_None).
    clean_text has the [CHART: ...] tag stripped out.
    """
    match = CHART_TAG_RE.search(answer_text)
    if not match:
        return answer_text, None

    clean_text = (answer_text[:match.start()] + answer_text[match.end():]).strip()

    chart_type = match.group("type").lower()
    title = match.group("title").strip()
    labels = [l.strip() for l in match.group("labels").split(",") if l.strip()]
    values_raw = match.group("values")

    spec = {"type": chart_type, "title": title, "labels": labels}

    if chart_type != "flow":
        if not values_raw:
            return clean_text, None   # malformed — skip rendering rather than crash
        try:
            spec["values"] = [float(v.strip()) for v in values_raw.split(",")]
        except ValueError:
            return clean_text, None
        if len(spec["values"]) != len(labels):
            return clean_text, None

    return clean_text, spec


def render_chart(spec: dict) -> Optional[str]:
    """Returns a base64 PNG data URL, or None on failure."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless — no display on Jetson server
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=110)
        fig.patch.set_facecolor("#0b1220")
        ax.set_facecolor("#0b1220")
        accent = "#22d3ee"   # matches the app's cyan holographic theme

        ctype = spec["type"]
        labels = spec["labels"]

        if ctype == "bar":
            ax.bar(labels, spec["values"], color=accent)
        elif ctype == "line":
            ax.plot(labels, spec["values"], color=accent, marker="o", linewidth=2)
        elif ctype == "pie":
            ax.pie(spec["values"], labels=labels, autopct="%1.0f%%",
                   textprops={"color": "white"},
                   colors=plt.cm.cool(__import__("numpy").linspace(0.2, 0.9, len(labels))))
        elif ctype == "flow":
            _render_flow(ax, labels, accent)

        if ctype != "pie":
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#334155")

        ax.set_title(spec["title"], color="white", fontsize=12, fontweight="bold")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)

        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        logger.error(f"Chart render error: {e}")
        return None


def _render_flow(ax, labels, accent):
    """Simple left-to-right boxes-and-arrows process diagram."""
    n = len(labels)
    if n == 0:
        return
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, label in enumerate(labels):
        x = i + 0.5
        ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
            (x - 0.42, 0.35), 0.84, 0.3,
            boxstyle="round,pad=0.02", linewidth=1.5,
            edgecolor=accent, facecolor="#111827",
        ))
        ax.text(x, 0.5, label, ha="center", va="center", color="white",
                 fontsize=9, wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(i + 1.08, 0.5), xytext=(i + 0.92, 0.5),
                        arrowprops=dict(arrowstyle="->", color=accent, lw=2))
