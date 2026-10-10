"""Plotly figure builders. All inputs are already privacy-protected aggregates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as C

FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
SERIF = "Georgia, 'Iowan Old Style', 'Palatino Linotype', serif"
SCORE_COLORS = {1: "#C8472B", 2: "#E3A06C", 3: "#7FB5AE", 4: "#1B7F79"}
SCORE_NAMES = {1: "1 (low)", 2: "2", 3: "3", 4: "4 (high)"}


def _layout(fig: go.Figure, height: int = 340, legend: bool = True, motion: bool = True,
            margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=margin or dict(l=8, r=12, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=C.INK),
        hoverlabel=dict(bgcolor=C.CARD, bordercolor=C.RULE, font=dict(family=FONT, size=12, color=C.INK)),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, title=None,
                    font=dict(size=11)) if legend else dict(),
        showlegend=legend,
        transition=dict(duration=350 if motion else 0, easing="cubic-in-out"),
        colorway=[C.INDIGO, C.TEAL, C.AMBER, C.VERMILION, C.SLATE],
    )
    fig.update_xaxes(gridcolor=C.RULE, zeroline=False, linecolor=C.RULE, tickfont=dict(size=11), automargin=True)
    fig.update_yaxes(gridcolor=C.RULE, zeroline=False, linecolor=C.RULE, tickfont=dict(size=11), automargin=True)
    return fig


def empty_figure(message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color=C.SLATE))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, height, legend=False)


def _pct(x) -> str:
    return "withheld (<5 in cell)" if x is None or pd.isna(x) else f"{x:.1%}"


# --------------------------------------------------------------------------- #
def cei_histogram(hist: pd.DataFrame, org_mean: float | None, motion: bool = True) -> go.Figure:
    if hist.empty:
        return empty_figure("No records to display.")
    fig = go.Figure(go.Bar(
        x=hist["bin"], y=hist["share"], marker_color=C.INDIGO, marker_line_width=0,
        customdata=np.c_[hist["count"]],
        hovertemplate="<b>CEI %{x}</b><br>%{y:.1%} of records in view (n=%{customdata[0]})<extra></extra>",
        name="Share of records",
    ))
    fig.update_yaxes(tickformat=".0%", title="Share of records in view")
    fig.update_xaxes(title="CEI (0-100; 25 = every item rated 1)")
    return _layout(fig, 300, legend=False, motion=motion)


def satisfaction_stack(dists: dict[str, pd.DataFrame], motion: bool = True) -> go.Figure:
    """100% stacked horizontal bars of 1-4 ratings per item; withheld cells appear as one grey segment."""
    fig = go.Figure()
    labels = [C.SCALE_LABELS[c] for c in dists]
    for score in (1, 2, 3, 4):
        shares = [d.loc[d["score"] == score, "share"].iloc[0] for d in dists.values()]
        counts = [d.loc[d["score"] == score, "count"].iloc[0] for d in dists.values()]
        fig.add_trace(go.Bar(
            y=labels, x=[0 if pd.isna(s) else s for s in shares], orientation="h",
            name=SCORE_NAMES[score], marker_color=SCORE_COLORS[score],
            customdata=np.c_[[("withheld" if pd.isna(c) else f"{int(c)}") for c in counts]],
            text=[("" if pd.isna(s) or s < 0.07 else f"{s:.0%}") for s in shares], textposition="inside",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{y}</b><br>Rating " + str(score) + ": %{x:.1%} (n=%{customdata[0]})<extra></extra>",
        ))
    pooled = [d.attrs.get("pooled_share", 0) or 0 for d in dists.values()]
    if any(p > 0 for p in pooled):
        fig.add_trace(go.Bar(
            y=labels, x=pooled, orientation="h", name=C.WITHHELD_LABEL, marker_color=C.GREY,
            marker_pattern_shape="/", hovertemplate="<b>%{y}</b><br>Combined share withheld: %{x:.1%}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Share of records in view")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 300, motion=motion, margin=dict(l=8, r=12, t=12, b=8))


def group_bars(summ: pd.DataFrame, metric: str = "cei_mean", comparator: float | None = None,
               horizontal: bool = True, selected: list[str] | None = None, title_axis: str = "Mean CEI",
               height: int | None = None, motion: bool = True, color: str = C.INDIGO,
               axis_range: tuple[float, float] | None = None) -> go.Figure:
    """Dot-and-interval comparison of mean CEI by group (zoomed axis, so NOT a bar chart on purpose)."""
    if summ.empty:
        return empty_figure("Not enough records per group to display (minimum 5).")
    s = summ.copy()
    custom = np.c_[s["n"], s["cei_lo"], s["cei_hi"], s["pct_below"].map(_pct), s["precision"]]
    pos = "y" if horizontal else "x"
    val = "x" if horizontal else "y"
    ht = ("<b>%{" + pos + "}</b><br>" + title_axis + ": %{" + val + ":.1f}<br>n=%{customdata[0]:,}"
          "<br>95% CI: %{customdata[1]:.1f} to %{customdata[2]:.1f}<br>Below threshold: %{customdata[3]}"
          "<br>Precision: %{customdata[4]}<extra>click to cross-filter</extra>")
    err = dict(type="data", symmetric=False, array=(s["cei_hi"] - s[metric]).fillna(0),
               arrayminus=(s[metric] - s["cei_lo"]).fillna(0), color=C.SLATE, thickness=1.6, width=5)
    kw = dict(mode="markers", marker=dict(size=12, color=color, line=dict(width=1.5, color="white")),
              customdata=custom, hovertemplate=ht, selected=dict(marker=dict(opacity=1, size=14)),
              unselected=dict(marker=dict(opacity=0.35)))
    if selected:
        kw["selectedpoints"] = [i for i, g in enumerate(s["group"]) if g in selected]
    lo = np.nanmin(s["cei_lo"].fillna(s[metric])); hi = np.nanmax(s["cei_hi"].fillna(s[metric]))
    rng = list(axis_range) if axis_range else [max(0, lo - 2), min(100, hi + 2)]
    if horizontal:
        fig = go.Figure(go.Scatter(x=s[metric], y=s["group"], error_x=err, **kw))
        fig.update_yaxes(autorange="reversed", automargin=True)
        fig.update_xaxes(title=title_axis + " (zoomed axis)", range=rng)
        if comparator is not None:
            fig.add_vline(x=comparator, line_dash="dot", line_color=C.VERMILION,
                          annotation_text=f"view mean {comparator:.1f}", annotation_position="top",
                          annotation_font=dict(size=10, color=C.VERMILION))
        h = height or max(250, 42 * len(s) + 80)
    else:
        fig = go.Figure(go.Scatter(x=s["group"], y=s[metric], error_y=err, **kw))
        fig.update_yaxes(title=title_axis + " (zoomed)", range=rng)
        fig.update_xaxes(type="category")
        if comparator is not None:
            fig.add_hline(y=comparator, line_dash="dot", line_color=C.VERMILION,
                          annotation_text=f"view mean {comparator:.1f}", annotation_position="top left",
                          annotation_font=dict(size=10, color=C.VERMILION))
        h = height or 320
    fig.update_layout(clickmode="event+select", dragmode=False)
    return _layout(fig, h, legend=False, motion=motion)


def risk_bars(mix: pd.DataFrame, selected: list[str] | None = None, motion: bool = True) -> go.Figure:
    if mix.empty or mix["share"].isna().all():
        return empty_figure("Not enough records to display risk mix.")
    m = mix.copy()
    shown = m.dropna(subset=["count"])
    fig = go.Figure(go.Bar(
        x=shown["level"], y=shown["share"],
        marker_color=[C.RISK_COLORS.get(l, C.GREY) for l in shown["level"]],
        text=[f"{s:.0%}" for s in shown["share"]], textposition="outside", cliponaxis=False,
        customdata=np.c_[shown["count"]],
        hovertemplate="<b>%{x} risk (screening proxy)</b><br>%{y:.1%} of records in view (n=%{customdata[0]:,})<extra>click to cross-filter</extra>",
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=0.3)),
        selectedpoints=[i for i, l in enumerate(shown["level"]) if l in (selected or [])] if selected else None,
    ))
    pooled = mix.attrs.get("pooled", 0)
    if pooled:
        fig.add_annotation(text=f"{pooled} records in cells under 5 are not drawn",
                           xref="paper", yref="paper", x=1, y=1.12, showarrow=False, font=dict(size=10, color=C.SLATE))
    fig.update_yaxes(tickformat=".0%", title="Share of records in view", range=[0, max(0.1, shown["share"].max() * 1.2)])
    fig.update_layout(clickmode="event+select", dragmode=False)
    return _layout(fig, 300, legend=False, motion=motion)


def risk_stack(long: pd.DataFrame, motion: bool = True, height: int | None = None) -> go.Figure:
    if long.empty:
        return empty_figure("Not enough records per group to display (minimum 5).")
    groups = list(dict.fromkeys(long["group"]))
    fig = go.Figure()
    for lev in C.BURNOUT_LEVELS + [C.WITHHELD_LABEL]:
        d = long[long["level"] == lev].set_index("group").reindex(groups)
        if d["share"].isna().all():
            continue
        color = C.RISK_COLORS.get(lev, C.GREY)
        fig.add_trace(go.Bar(
            y=groups, x=d["share"].fillna(0), orientation="h", name=lev, marker_color=color,
            marker_pattern_shape="/" if lev == C.WITHHELD_LABEL else "",
            customdata=np.c_[d["n"].fillna(0), d["count"].fillna(0)],
            text=[("" if pd.isna(v) or v < 0.06 else f"{v:.0%}") for v in d["share"]], textposition="inside",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{y}</b> (n=%{customdata[0]:,})<br>" + lev + ": %{x:.1%} (n=%{customdata[1]:,})<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Share of records in group")
    fig.update_yaxes(autorange="reversed", automargin=True)
    return _layout(fig, height or max(230, 46 * len(groups) + 90), motion=motion)


def heatmap(n: pd.DataFrame, mean: pd.DataFrame, xtitle: str, ytitle: str, motion: bool = True,
            zmin: float = 45, zmax: float = 90, value_name: str = "Mean CEI") -> go.Figure:
    if n.empty or n.isna().all().all():
        return empty_figure("Not enough records per cell to display (minimum 5).")
    text = mean.copy().astype(object)
    for r in mean.index:
        for c in mean.columns:
            nv, mv = n.loc[r, c], mean.loc[r, c]
            text.loc[r, c] = "" if pd.isna(nv) or pd.isna(mv) else f"<b>{mv:.0f}</b><br><span style='font-size:10px'>n={int(nv)}</span>"
    custom = np.dstack([n.fillna(0).values, mean.fillna(np.nan).values])
    fig = go.Figure(go.Heatmap(
        z=mean.values, x=[str(c) for c in mean.columns], y=[str(i) for i in mean.index],
        colorscale=[[0, C.VERMILION], [0.5, "#F2E9CF"], [1, C.TEAL]], zmin=zmin, zmax=zmax,
        text=text.values, texttemplate="%{text}", textfont=dict(size=12), customdata=custom,
        hovertemplate="<b>%{y}</b> / <b>%{x}</b><br>" + value_name + ": %{z:.1f}<br>n=%{customdata[0]:,.0f}<extra></extra>",
        hoverongaps=False, xgap=3, ygap=3, colorbar=dict(title=value_name, thickness=10, len=0.8)))
    fig.update_xaxes(title=xtitle, side="bottom")
    fig.update_yaxes(title=ytitle, autorange="reversed")
    fig.add_annotation(text="Blank cell = fewer than 5 records (withheld) or no records", xref="paper", yref="paper",
                       x=0, y=-0.28, showarrow=False, font=dict(size=10, color=C.SLATE))
    return _layout(fig, 300, legend=False, motion=motion, margin=dict(l=8, r=12, t=12, b=40))


def tenure_lines(series: dict[str, pd.DataFrame], motion: bool = True) -> go.Figure:
    fig = go.Figure()
    palette = [C.INDIGO, C.TEAL, C.AMBER]
    drawn = False
    for (name, d), col in zip(series.items(), palette):
        if d is None or d.empty:
            continue
        drawn = True
        fig.add_trace(go.Scatter(
            x=list(d["group"]) + list(d["group"])[::-1],
            y=list(d["cei_hi"].fillna(d["cei_mean"])) + list(d["cei_lo"].fillna(d["cei_mean"]))[::-1],
            fill="toself", fillcolor=_rgba(col, 0.12), line=dict(width=0), hoverinfo="skip",
            showlegend=False, legendgroup=name))
        fig.add_trace(go.Scatter(
            x=d["group"], y=d["cei_mean"], mode="lines+markers", name=name, legendgroup=name,
            line=dict(color=col, width=2.5), marker=dict(size=8),
            customdata=np.c_[d["n"], d["cei_lo"], d["cei_hi"]],
            hovertemplate="<b>" + name + "</b>: %{x}<br>Mean CEI %{y:.1f}<br>n=%{customdata[0]:,}<br>95% CI %{customdata[1]:.1f} to %{customdata[2]:.1f}<extra></extra>"))
    if not drawn:
        return empty_figure("Not enough records per tenure band to display (minimum 5).")
    fig.update_yaxes(title="Mean CEI (95% CI band)")
    fig.update_xaxes(title="Tenure band (groups under 5 records are merged with a neighbour)")
    return _layout(fig, 320, motion=motion)


def attrition_pair(comp, motion: bool = True) -> go.Figure:
    if not comp.reportable:
        return empty_figure("Stayed/left groups too small in this view (minimum 5 each).", 220)
    fig = go.Figure(go.Bar(
        x=["Stayed (Attrition = 0)", "Left (Attrition = 1)"], y=[comp.mean_stayed, comp.mean_left],
        marker_color=[C.TEAL, C.SLATE], customdata=np.c_[[comp.n_stayed, comp.n_left]],
        text=[f"{comp.mean_stayed:.1f}", f"{comp.mean_left:.1f}"], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Mean CEI %{y:.1f}<br>n=%{customdata[0]:,}<extra></extra>"))
    fig.update_yaxes(title="Mean CEI", range=[0, 100])
    return _layout(fig, 260, legend=False, motion=motion)


def gap_bars(pr: pd.DataFrame, motion: bool = True) -> go.Figure:
    """Mean-CEI gap vs comparator by area (aggregate), coloured by number of flags."""
    if pr.empty:
        return empty_figure("Not enough records per area to display (minimum 5).")
    d = pr.sort_values("cei_gap")
    colors = {0: C.GREY, 1: C.AMBER, 2: "#E07A2E", 3: C.VERMILION, 4: C.VERMILION}
    fig = go.Figure(go.Bar(
        x=d["cei_gap"], y=d["group"], orientation="h", marker_color=[colors[n] for n in d["n_flags"]],
        customdata=np.c_[d["n"], d["baseline"], d["baseline_basis"], d["n_flags"]],
        hovertemplate="<b>%{y}</b> (n=%{customdata[0]:,})<br>Gap vs %{customdata[2]} mean (%{customdata[1]:.1f}): %{x:+.1f} CEI pts<br>Flags: %{customdata[3]}<extra></extra>"))
    fig.add_vline(x=0, line_color=C.INK, line_width=1)
    fig.update_xaxes(title="Mean CEI minus comparator (points)")
    fig.update_yaxes(automargin=True)
    return _layout(fig, max(240, 40 * len(d) + 70), legend=False, motion=motion)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
