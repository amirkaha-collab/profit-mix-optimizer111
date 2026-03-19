# -*- coding: utf-8 -*-
"""
client_portfolio/charts.py
───────────────────────────
9 analysis layers as professional Plotly charts.
All charts return go.Figure and are self-contained.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Design ─────────────────────────────────────────────────────────────────────

_NAVY    = "#1F3A5F"
_BLUE    = "#3A7AFE"
_GREEN   = "#10B981"
_AMBER   = "#F59E0B"
_RED     = "#EF4444"
_PURPLE  = "#8B5CF6"
_TEAL    = "#06B6D4"
_PINK    = "#EC4899"
_SLATE   = "#64748B"

# Richer palette for multi-series
_PAL = [_BLUE, _GREEN, _AMBER, _RED, _PURPLE, _TEAL, _PINK,
        "#F97316", "#84CC16", "#6366F1", "#14B8A6", "#FB7185"]

_FONT = dict(family="Segoe UI, -apple-system, sans-serif", color="#374151")
_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,249,252,1)",
    font=_FONT,
    margin=dict(l=80, r=36, t=60, b=70),
    legend=dict(orientation="h", yanchor="bottom", y=-0.28,
                xanchor="center", x=0.5, font=dict(size=12)),
)

def _title(text: str) -> dict:
    return dict(text=text, font=dict(size=15, color=_NAVY, family=_FONT["family"]), x=0.5)

def _axis_common(**kwargs):
    base = dict(automargin=True, gridcolor="#E5E7EB", zeroline=False)
    base.update(kwargs)
    return base

def _truncate_label(value: str, limit: int = 28) -> str:
    s = str(value or "").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"

def _product_label(row: pd.Series, multiline: bool = False) -> str:
    provider = _truncate_label(row.get("provider", ""), 22)
    product = _truncate_label(row.get("product_name", row.get("provider", "")), 26)
    return f"{provider}<br>{product}" if multiline else f"{provider} | {product}"

def _weighted_metric(active: pd.DataFrame, col: str) -> tuple[float, float]:
    if col not in active.columns or active.empty:
        return float("nan"), 0.0
    sub = active[active[col].notna()].copy()
    if sub.empty:
        return float("nan"), 0.0
    covered_amount = float(sub["amount"].sum())
    total = float(active["amount"].sum())
    if covered_amount <= 0 or total <= 0:
        return float("nan"), 0.0
    weighted = float((sub[col] * sub["amount"]).sum() / covered_amount)
    coverage = covered_amount / total * 100
    return weighted, coverage

def _fmt_ils(v: float) -> str:
    if v >= 1_000_000:
        return f"₪{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"₪{v/1_000:.0f}K"
    return f"₪{v:.0f}"

def _nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except Exception:
        return True

# ── 1. Total assets KPI cards (returns dict, not figure) ──────────────────────

def compute_totals(df: pd.DataFrame) -> dict:
    """Return high-level portfolio numbers plus coverage by layer."""
    active = df[~df.get("excluded", pd.Series([False] * len(df))).astype(bool)].copy()
    total = float(active["amount"].sum()) if "amount" in active.columns else 0.0
    n_prod = len(active)
    n_mgr = active["provider"].nunique() if "provider" in active.columns else 0

    equity, equity_cov = _weighted_metric(active, "equity_pct")
    foreign, foreign_cov = _weighted_metric(active, "foreign_pct")
    fx, fx_cov = _weighted_metric(active, "fx_pct")
    illiquid, illiquid_cov = _weighted_metric(active, "illiquid_pct")
    cost, cost_cov = _weighted_metric(active, "annual_cost_pct")

    return {
        "total": total,
        "n_products": n_prod,
        "n_managers": n_mgr,
        "equity": equity,
        "foreign": foreign,
        "fx": fx,
        "illiquid": illiquid,
        "cost": cost,
        "equity_coverage": equity_cov,
        "foreign_coverage": foreign_cov,
        "fx_coverage": fx_cov,
        "illiquid_coverage": illiquid_cov,
        "cost_coverage": cost_cov,
        "data_coverage": min(equity_cov, foreign_cov, fx_cov, illiquid_cov) if total > 0 else 0.0,
    }


# ── 2. By-manager pie ─────────────────────────────────────────────────────────

def chart_by_manager(df: pd.DataFrame) -> go.Figure:
    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)]
    grp = active.groupby("provider")["amount"].sum().reset_index()
    grp = grp[grp["amount"] > 0].sort_values("amount", ascending=False)

    fig = go.Figure(go.Pie(
        labels=grp["provider"],
        values=grp["amount"],
        hole=0.45,
        marker=dict(colors=_PAL[:len(grp)], line=dict(color="#fff", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} ₪<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT_BASE, title=_title("2. פיזור בין מנהלים"), height=380)
    fig.add_annotation(text=f"<b>{_fmt_ils(grp['amount'].sum())}</b>",
                       x=0.5, y=0.5, font=dict(size=16, color=_NAVY), showarrow=False)
    return fig


# ── 3. Stocks vs Bonds ────────────────────────────────────────────────────────

def chart_stocks_bonds(df: pd.DataFrame) -> go.Figure:
    active = df[~df.get("excluded", pd.Series([False] * len(df))).astype(bool)].copy()
    equity, coverage = _weighted_metric(active, "equity_pct")
    equity = 0.0 if _nan(equity) else equity
    residual = max(0.0, 100 - equity)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["תמהיל ידוע"],
        y=[equity],
        name=f"מניות ({equity:.1f}%)",
        marker_color=_BLUE,
        text=[f"{equity:.1f}%"],
        textposition="inside",
        hovertemplate="מניות: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=["תמהיל ידוע"],
        y=[residual],
        name=f"יתר הרכיבים ({residual:.1f}%)",
        marker_color=_GREEN,
        text=[f"{residual:.1f}%"],
        textposition="inside",
        hovertemplate="יתר הרכיבים: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=_title(f"3. מניות מול יתר הרכיבים · כיסוי {coverage:.0f}%"),
        barmode="stack",
        height=340,
        yaxis=_axis_common(ticksuffix="%", range=[0, 105]),
        xaxis=_axis_common(showgrid=False),
        showlegend=True,
    )
    return fig


# ── 4. Foreign vs Israel ──────────────────────────────────────────────────────

def chart_foreign_domestic(df: pd.DataFrame) -> go.Figure:
    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)]

    def _wsum(col):
        sub = active[active[col].notna()].copy() if col in active.columns else pd.DataFrame()
        if sub.empty:
            return 0.0
        t = sub["amount"].sum()
        total = active["amount"].sum()
        return float((sub[col] * sub["amount"]).sum() / t * (t / total)) if t > 0 else 0.0

    foreign = _wsum("foreign_pct")
    domestic = max(0.0, 100 - foreign)
    _, coverage = _weighted_metric(active, "foreign_pct")

    fig = go.Figure(go.Pie(
        labels=["חו\"ל", "ישראל"],
        values=[foreign, domestic],
        hole=0.5,
        marker=dict(colors=[_BLUE, _GREEN], line=dict(color="#fff", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT_BASE, title=_title(f"4. חו\"ל מול ישראל · כיסוי {coverage:.0f}%"), height=340)
    return fig


# ── 5. FX vs ILS ─────────────────────────────────────────────────────────────

def chart_fx_ils(df: pd.DataFrame) -> go.Figure:
    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)]

    def _wsum(col):
        sub = active[active[col].notna()].copy() if col in active.columns else pd.DataFrame()
        if sub.empty:
            return 0.0
        t = sub["amount"].sum()
        total = active["amount"].sum()
        return float((sub[col] * sub["amount"]).sum() / t * (t / total)) if t > 0 else 0.0

    fx = _wsum("fx_pct")
    ils = max(0.0, 100 - fx)
    _, coverage = _weighted_metric(active, "fx_pct")

    fig = go.Figure(go.Pie(
        labels=['מט"ח', "שקל"],
        values=[fx, ils],
        hole=0.5,
        marker=dict(colors=[_AMBER, _SLATE], line=dict(color="#fff", width=2)),
        textinfo="label+percent",
    ))
    fig.update_layout(**_LAYOUT_BASE, title=_title(f'5. מט"ח מול שקל · כיסוי {coverage:.0f}%'), height=340)
    return fig


# ── 6. Asset-type breakdown per product ──────────────────────────────────────

def chart_asset_breakdown(df: pd.DataFrame) -> go.Figure:
    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)]
    if active.empty:
        return go.Figure()

    cols_show = {
        "equity_pct":   ("מניות", _BLUE),
        "foreign_pct":  ('חו"ל', _GREEN),
        "fx_pct":       ('מט"ח', _AMBER),
        "illiquid_pct": ("לא סחיר", _RED),
    }

    labels = active.apply(lambda row: _product_label(row, multiline=True), axis=1).tolist()

    fig = go.Figure()
    for col, (name, color) in cols_show.items():
        vals = active[col].fillna(0).tolist() if col in active.columns else [0]*len(active)
        fig.add_trace(go.Bar(
            name=name, x=labels, y=vals,
            marker_color=color, opacity=0.88,
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=_title("6. פיזור סוגי נכסים לפי מוצר"),
        barmode="group", height=max(420, 120 + 24 * len(active)),
        xaxis=_axis_common(tickangle=-20, tickfont=dict(size=10)),
        yaxis=_axis_common(ticksuffix="%"),
    )
    return fig


# ── 7. Annuity vs Capital ─────────────────────────────────────────────────────

_ANNUITY_TYPES = {"קרנות פנסיה"}
_CAPITAL_TYPES = {"קרנות השתלמות", "פוליסות חיסכון", "קופות גמל", "גמל להשקעה"}

def chart_annuity_capital(df: pd.DataFrame) -> go.Figure:
    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)]
    if "product_type" not in active.columns:
        return go.Figure().update_layout(**_LAYOUT_BASE, title=_title("7. קצבה vs הון"))

    annuity = active[active["product_type"].isin(_ANNUITY_TYPES)]["amount"].sum()
    capital = active[active["product_type"].isin(_CAPITAL_TYPES)]["amount"].sum()
    other   = active["amount"].sum() - annuity - capital

    labels = ["קצבה (פנסיה)", "הון (קרנות/גמל)"]
    values = [annuity, capital]
    colors = [_PURPLE, _TEAL]
    if other > 0:
        labels.append("אחר")
        values.append(other)
        colors.append(_SLATE)

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textinfo="label+percent+value",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} ₪<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT_BASE, title=_title("7. קצבה vs הון"), height=360)
    return fig


# ── 8. Cost analysis ──────────────────────────────────────────────────────────

def chart_costs(df: pd.DataFrame) -> go.Figure:
    if "annual_cost_pct" not in df.columns:
        return go.Figure()

    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)]
    active = active[active["annual_cost_pct"].notna()]
    if active.empty:
        return go.Figure()

    active = active.copy()
    active["annual_cost_ils"] = active["amount"] * active["annual_cost_pct"] / 100
    active["label"] = active["provider"] + "\n" + active.get("product_name", active["provider"])

    total_cost = active["annual_cost_ils"].sum()
    total_amt  = active["amount"].sum()
    weighted_cost_pct = total_cost / total_amt * 100 if total_amt > 0 else 0

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("עלות שנתית לפי מוצר (₪)", "דמי ניהול % לפי מוצר"))

    fig.add_trace(go.Bar(
        x=active["label"], y=active["annual_cost_ils"],
        marker_color=_RED, name="עלות שנתית",
        text=active["annual_cost_ils"].map(lambda v: f"₪{v:,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>₪%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=active["label"], y=active["annual_cost_pct"],
        marker_color=_AMBER, name='דמי ניהול %',
        text=active["annual_cost_pct"].map(lambda v: f"{v:.2f}%"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        **_LAYOUT_BASE,
        title=_title(f"8. עלויות — עלות משוקללת: {weighted_cost_pct:.2f}% | סה\"כ שנתי: {_fmt_ils(total_cost)}"),
        height=max(420, 140 + 18 * len(active)), showlegend=False,
    )
    fig.update_xaxes(**_axis_common(tickangle=-18, tickfont=dict(size=10)))
    fig.update_yaxes(**_axis_common())
    return fig


# ── 9a. Concentration risk ────────────────────────────────────────────────────

def chart_concentration(df: pd.DataFrame) -> go.Figure:
    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)]
    total  = active["amount"].sum()
    if total == 0:
        return go.Figure()

    # Herfindahl-Hirschman Index by manager
    mgr = active.groupby("provider")["amount"].sum().reset_index()
    mgr["weight"] = mgr["amount"] / total * 100
    mgr = mgr.sort_values("weight", ascending=False)

    hhi = float(((mgr["weight"] / 100) ** 2).sum() * 10000)
    risk_label = "ריכוז גבוה" if hhi > 2500 else "ריכוז בינוני" if hhi > 1500 else "פיזור טוב"
    risk_color = _RED if hhi > 2500 else _AMBER if hhi > 1500 else _GREEN

    fig = go.Figure(go.Bar(
        x=mgr["provider"].map(_truncate_label), y=mgr["weight"],
        marker_color=[risk_color if w > 30 else _BLUE for w in mgr["weight"]],
        text=mgr["weight"].map(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.1f}% מהתיק<extra></extra>",
    ))
    fig.add_hline(y=30, line_dash="dash", line_color=_AMBER,
                  annotation_text="30% — סף ריכוז", annotation_position="top right")
    fig.update_layout(
        **_LAYOUT_BASE,
        title=_title(f"9א. ריכוז מנהלים — HHI: {hhi:.0f} ({risk_label})"),
        height=360,
        yaxis=_axis_common(ticksuffix="%"),
        xaxis=_axis_common(tickangle=-18),
    )
    return fig


# ── 9b. Sharpe comparison per product ────────────────────────────────────────

def chart_sharpe_comparison(df: pd.DataFrame) -> go.Figure:
    active = df[~df.get("excluded", pd.Series([False]*len(df))).astype(bool)]
    if "sharpe" not in active.columns:
        return go.Figure()

    sub = active[active["sharpe"].notna()].copy()
    if sub.empty:
        return go.Figure()

    sub["label"] = sub.apply(lambda row: _product_label(row, multiline=False), axis=1)
    sub = sub.sort_values("sharpe", ascending=True)

    fig = go.Figure(go.Bar(
        x=sub["sharpe"], y=sub["label"],
        orientation="h",
        marker_color=[_GREEN if v > 0.6 else _AMBER if v > 0.3 else _RED for v in sub["sharpe"]],
        text=sub["sharpe"].map(lambda v: f"{v:.2f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>שארפ: %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color=_SLATE,
                  annotation_text="0.5 — יעד", annotation_position="top right")
    fig.update_layout(
        **_LAYOUT_BASE,
        title=_title("9ב. השוואת שארפ בין מוצרים"),
        height=max(300, 60 + 35 * len(sub)),
        xaxis=_axis_common(),
        yaxis=_axis_common(tickfont=dict(size=11)),
        showlegend=False,
    )
    return fig


# ── 9c. Weighted allocation radar ────────────────────────────────────────────

def chart_radar(df: pd.DataFrame) -> go.Figure:
    totals = compute_totals(df)
    cats   = ["מניות", 'חו"ל', 'מט"ח', "לא סחיר"]
    vals   = [
        totals.get("equity", 0) or 0,
        totals.get("foreign", 0) or 0,
        totals.get("fx", 0) or 0,
        totals.get("illiquid", 0) or 0,
    ]
    # close the polygon
    cats_c = cats + [cats[0]]
    vals_c = vals + [vals[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_c, theta=cats_c,
        fill="toself", fillcolor=f"rgba(58,122,254,0.15)",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(size=7, color=_BLUE),
        name="תמהיל נוכחי",
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT_BASE,
        title=_title("9ג. מפת נכסים — Radar"),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            ticksuffix="%", gridcolor="#E5E7EB"),
            angularaxis=dict(direction="clockwise"),
        ),
        showlegend=False, height=380,
    )
    return fig
