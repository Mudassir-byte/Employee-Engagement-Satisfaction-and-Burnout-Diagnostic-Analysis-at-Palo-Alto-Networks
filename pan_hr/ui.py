"""Styling and small HTML components for the Streamlit workspace."""
from __future__ import annotations

import html

from . import config as C

BASE_CSS = f"""
<style>
:root {{
  --ink:{C.INK}; --paper:{C.PAPER}; --paper-deep:{C.PAPER_DEEP}; --card:{C.CARD}; --slate:{C.SLATE};
  --rule:{C.RULE}; --teal:{C.TEAL}; --amber:{C.AMBER}; --vermilion:{C.VERMILION}; --indigo:{C.INDIGO};
  --serif: Georgia, 'Iowan Old Style', 'Palatino Linotype', serif;
}}
.stApp {{
  background-color: var(--paper);
  background-image:
    linear-gradient(rgba(20,32,43,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(20,32,43,.035) 1px, transparent 1px),
    radial-gradient(1200px 500px at 85% -10%, rgba(27,127,121,.10), transparent 60%);
  background-size: 28px 28px, 28px 28px, auto;
  color: var(--ink);
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
h1,h2,h3,h4 {{ font-family: var(--serif); letter-spacing:-.01em; color: var(--ink); }}
section[data-testid="stSidebar"] {{ background: var(--paper-deep); border-right: 1px solid var(--rule); }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}

.hero {{ border-bottom: 2px solid var(--ink); padding-bottom: .6rem; margin-bottom: .9rem; }}
.hero .eyebrow {{ font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color: var(--slate); font-weight:600; }}
.hero h1 {{ font-size: clamp(1.5rem, 3.2vw, 2.3rem); margin:.15rem 0 .25rem; line-height:1.1; }}
.hero .sub {{ color: var(--slate); font-size:.95rem; max-width: 70ch; }}
.pill {{ display:inline-block; font-size:.72rem; padding:.18rem .55rem; border:1px solid var(--ink); border-radius:999px;
  margin: .45rem .35rem 0 0; background: var(--card); }}
.pill.warn {{ border-color: var(--vermilion); color: var(--vermilion); }}

.kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(138px, 1fr)); gap:.75rem; margin: .4rem 0 1rem; }}
.kpi {{ background: var(--card); border:1px solid var(--rule); border-top:3px solid var(--ink); border-radius:4px;
  padding:.7rem .85rem .75rem; box-shadow: 0 1px 0 rgba(20,32,43,.04), 0 6px 14px -10px rgba(20,32,43,.35); }}
.kpi .label {{ font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color: var(--slate); font-weight:600; }}
.kpi .value {{ font-family: var(--serif); font-size:1.85rem; line-height:1.15; margin-top:.15rem; }}
.kpi .foot {{ font-size:.76rem; color: var(--slate); margin-top:.1rem; }}
.kpi.risk-high {{ border-top-color: var(--vermilion); }} .kpi.risk-med {{ border-top-color: var(--amber); }}
.kpi.risk-low {{ border-top-color: var(--teal); }}

.section-title {{ font-family: var(--serif); font-size:1.25rem; margin: 1.1rem 0 .1rem; }}
.section-cap {{ color: var(--slate); font-size:.86rem; margin-bottom:.4rem; max-width: 90ch; }}
.note {{ border-left:4px solid var(--indigo); background: rgba(46,75,115,.07); padding:.55rem .8rem; border-radius:0 4px 4px 0;
  font-size:.86rem; margin:.5rem 0 .8rem; }}
.note.caution {{ border-left-color: var(--vermilion); background: rgba(210,68,28,.07); }}
.note.good {{ border-left-color: var(--teal); background: rgba(27,127,121,.08); }}
.area-card {{ background: var(--card); border:1px solid var(--rule); border-left:5px solid var(--grey,#B4AEA0); border-radius:4px;
  padding:.75rem .95rem; margin:.55rem 0; }}
.area-card.t3 {{ border-left-color: var(--vermilion); }} .area-card.t2 {{ border-left-color:#E07A2E; }}
.area-card.t1 {{ border-left-color: var(--amber); }}
.area-card h4 {{ margin:0 0 .15rem; font-size:1.02rem; }}
.area-card .meta {{ font-size:.8rem; color: var(--slate); }}
.area-card ul {{ margin:.35rem 0 .1rem 1.1rem; padding:0; font-size:.88rem; }}
.chip {{ display:inline-block; font-size:.72rem; padding:.1rem .5rem; border-radius:3px; background: var(--paper-deep);
  border:1px solid var(--rule); margin-right:.3rem; }}
.chip.active {{ background: var(--ink); color: var(--paper); border-color: var(--ink); }}
.footer {{ color: var(--slate); font-size:.75rem; border-top:1px solid var(--rule); margin-top:1.6rem; padding-top:.6rem; }}

/* Navigation radio styled as a tab strip */
div[data-testid="stRadio"] > div[role="radiogroup"] {{ gap:.35rem; flex-wrap:wrap; }}
div[data-testid="stRadio"] label[data-baseweb="radio"] {{ background: var(--card); border:1px solid var(--rule);
  padding:.35rem .8rem; border-radius:999px; }}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{ background: var(--ink); color: var(--paper); border-color: var(--ink); }}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {{ color: var(--paper); }}
div[data-testid="stRadio"] label > div:first-child {{ display:none; }}

@media (prefers-reduced-motion: no-preference) {{
  @keyframes rise {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform:none; }} }}
  .hero, .kpi, .section-title, .area-card, .note {{ animation: rise .32s ease-out both; }}
  .kpi:nth-child(2) {{ animation-delay:.04s; }} .kpi:nth-child(3) {{ animation-delay:.08s; }}
  .kpi:nth-child(4) {{ animation-delay:.12s; }} .kpi:nth-child(5) {{ animation-delay:.16s; }}
  .kpi:nth-child(6) {{ animation-delay:.2s; }}
}}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation:none !important; transition:none !important; }} }}
@media (max-width: 640px) {{
  .block-container {{ padding-left:.8rem; padding-right:.8rem; }}
  .kpi .value {{ font-size:1.5rem; }}
}}
</style>
"""

NO_MOTION_CSS = "<style>*, *::before, *::after { animation:none !important; transition:none !important; }</style>"


def esc(x) -> str:
    return html.escape(str(x))


def kpi(label: str, value: str, foot: str = "", tone: str = "") -> str:
    cls = f"kpi {tone}".strip()
    return (f'<div class="{cls}"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
            f'<div class="foot">{esc(foot)}</div></div>')


def kpi_row(items: list[str]) -> str:
    return '<div class="kpis">' + "".join(items) + "</div>"


def section(title: str, caption: str = "") -> str:
    cap = f'<div class="section-cap">{esc(caption)}</div>' if caption else ""
    return f'<div class="section-title">{esc(title)}</div>{cap}'


def note(text: str, kind: str = "") -> str:
    return f'<div class="note {kind}">{esc(text)}</div>'


def hero(n_view: str, n_total: int) -> str:
    return (
        '<div class="hero"><div class="eyebrow">HR analytics workspace - aggregate view</div>'
        '<h1>Engagement, Satisfaction &amp; Burnout Diagnostic</h1>'
        '<div class="sub">Early-warning view of aggregate work conditions in the provided dataset, '
        'so leaders can investigate workload and career conditions before attrition occurs.</div>'
        f'<span class="pill">Records in view: {esc(n_view)} of {n_total:,}</span>'
        '<span class="pill">Snapshot data - no trend or persistence data</span>'
        '<span class="pill warn">Not confirmed as actual Palo Alto Networks employee data</span>'
        '<span class="pill">Aggregates only - groups under 5 withheld</span></div>'
    )
