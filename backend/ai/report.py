"""
Interactive HTML report for the AI-vs-AI tournament results.

Produces a restrained, high-end editorial light mode HTML dashboard (white background,
clean data tables, vibrant blue histograms matching user design, and animated green/red
win split bars for matchups).

Usage:
    from ai.report import write_report
    write_report(results, games_per_pair, "ai_vs_ai_results.html")
"""

from __future__ import annotations

import html
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Palette constants
_HISTOGRAM_BLUE = "#3B82F6"  # Exact vivid blue requested by user
_GREEN_WINNER = "#10B981"    # Green for larger win portion
_RED_LOSER = "#EF4444"       # Red for smaller win portion
_BORDER_COLOR = "#E2E8F0"


# ----------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------

def _per_ai_totals(
    results: Dict[str, Dict], keys: Sequence[str],
) -> Dict[str, Tuple[int, int, int, int, float, float]]:
    """Return {ai: (games, wins, losses, draws, win_rate, avg_shots)}."""
    totals = {}
    for ai in keys:
        wins = losses = draws = 0
        all_shots = []
        for opp in keys:
            if ai == opp:
                continue
            s = results[ai][opp]
            wins += s["a_wins"]
            losses += s["b_wins"]
            draws += s["draws"]
            all_shots.extend(s["shots_a"])
        games = wins + losses + draws
        pct = wins / max(wins + losses, 1)
        avg_shots = float(np.mean(all_shots)) if all_shots else 0.0
        totals[ai] = (games, wins, losses, draws, pct, avg_shots)
    return totals


def _ai_shot_stats(ai_key: str, results: Dict[str, Dict], keys: Sequence[str]) -> Tuple[List[int], int, float, float]:
    """Return (winning_shots_list, min_shots, median_shots, mean_shots)."""
    winning_shots = []
    all_shots = []
    for opp in keys:
        if opp == ai_key:
            continue
        s = results[ai_key][opp]
        winners = s.get("winners_a", [])
        shots = s.get("shots_a", [])
        for w, sh in zip(winners, shots):
            all_shots.append(sh)
            if w == 1:
                winning_shots.append(sh)
    
    data = winning_shots if winning_shots else all_shots
    if not data:
        return [], 0, 0.0, 0.0

    min_s = int(np.min(data))
    med_s = float(np.median(data))
    mean_s = float(np.mean(data))
    return data, min_s, med_s, mean_s





# ----------------------------------------------------------------------
# Figure Builders
# ----------------------------------------------------------------------

def _build_shot_efficiency_histogram(
    ai_key: str,
    shot_data: List[int],
    display_name: str,
) -> str:
    """Build a clean light-mode histogram using the exact vivid blue requested for all 4 AI charts."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=shot_data,
        marker_color=_HISTOGRAM_BLUE,
        marker_line_color="#FFFFFF",
        marker_line_width=1,
        opacity=0.9,
        xbins=dict(start=15, end=100, size=4),
        hovertemplate=f"<b>{display_name}</b><br>Shot Range: %{{x}}<br>Victories: <b>%{{y}}</b><extra></extra>",
    ))

    fig.update_layout(
        autosize=True,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFA",
        font=dict(color="#1E293B", family="Inter, sans-serif"),
        height=220,
        margin=dict(l=35, r=25, t=20, b=35),
        xaxis=dict(
            title=dict(text="Shots Fired to Win Game", font=dict(size=11, color="#64748B")),
            gridcolor="#E2E8F0",
            zerolinecolor="#E2E8F0",
            color="#475569",
            range=[15, 100],
            autorange=False,
        ),
        yaxis=dict(
            title=dict(text="Games Count", font=dict(size=11, color="#64748B")),
            gridcolor="#E2E8F0",
            zerolinecolor="#E2E8F0",
            color="#475569",
        ),
        showlegend=False,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


# ----------------------------------------------------------------------
# HTML Assembly
# ----------------------------------------------------------------------

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,500;0,700;1,400&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script>{plotly_js}</script>
<style>
  :root {{
    --bg-page: #FFFFFF;
    --bg-alt: #F8FAFC;
    --border-color: #E2E8F0;
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --text-muted: #94A3B8;
    --green-win: #10B981;
    --red-loss: #EF4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-page);
    color: var(--text-primary);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    padding: 48px 24px 64px;
    line-height: 1.6;
  }}
  .container {{ max-width: 1060px; margin: 0 auto; }}

  header {{
    margin-bottom: 40px;
    border-bottom: 2px solid var(--text-primary);
    padding-bottom: 20px;
  }}
  header h1 {{
    font-family: 'Newsreader', Georgia, serif;
    font-size: 36px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.15;
  }}
  header p {{
    color: var(--text-secondary);
    font-size: 14px;
    margin-top: 8px;
  }}

  .section-header {{
    margin-top: 40px;
    margin-bottom: 16px;
  }}
  .section-header h2 {{
    font-family: 'Newsreader', Georgia, serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
  }}
  .section-header p {{
    font-size: 13px;
    color: var(--text-secondary);
  }}

  /* Tables */
  .table-wrapper {{
    width: 100%;
    overflow-x: auto;
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    margin-bottom: 36px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    text-align: left;
    font-size: 13px;
  }}
  th {{
    background-color: var(--bg-alt);
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
  }}
  td {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    color: var(--text-primary);
  }}
  tr:last-child td {{
    border-bottom: none;
  }}
  tr:hover {{
    background-color: #F1F5F9;
  }}

  .rank-badge {{
    font-weight: 700;
    font-size: 12px;
    color: var(--text-secondary);
  }}

  /* Animated Win Split Bar in Table */
  .bar-container {{
    display: flex;
    height: 8px;
    width: 140px;
    background: #E2E8F0;
    border-radius: 4px;
    overflow: hidden;
    margin-right: 10px;
  }}
  .bar-fill-a, .bar-fill-b {{
    height: 100%;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    animation: growSplit 1s ease-out forwards;
  }}
  @keyframes growSplit {{
    from {{ opacity: 0; transform: scaleX(0); }}
    to {{ opacity: 1; transform: scaleX(1); }}
  }}
  .flex-align {{
    display: flex;
    align-items: center;
  }}

  /* Histograms Grid */
  .histograms-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 24px;
    margin-top: 16px;
    margin-bottom: 40px;
  }}
  .histogram-card {{
    background: #FFFFFF;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 20px;
    min-width: 0;
    overflow: hidden;
  }}
  .histogram-card .js-plotly-plot,
  .histogram-card .plot-container {{
    width: 100% !important;
  }}
  .histogram-card h3 {{
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 2px;
  }}
  .histogram-subhead {{
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 12px;
  }}

  /* Insight Pills */
  .insight-row {{
    display: flex;
    gap: 12px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color);
  }}
  .insight-pill {{
    background: var(--bg-alt);
    border: 1px solid var(--border-color);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    flex: 1;
  }}
  .insight-pill label {{
    display: block;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
  }}
  .insight-pill span {{
    font-weight: 700;
    color: var(--text-primary);
  }}
  .insight-note {{
    margin-top: 10px;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.4;
    font-style: italic;
  }}

  footer {{
    text-align: center;
    margin-top: 48px;
    padding-top: 24px;
    color: var(--text-muted);
    font-size: 12px;
    border-top: 1px solid var(--border-color);
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Battleship Performance Analysis</h1>
  </header>

  <div class="section-header">
    <h2>Overall Rankings</h2>
  </div>
  <div class="table-wrapper">
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>AI Strategy</th>
          <th>Matches</th>
          <th>Wins</th>
          <th>Losses</th>
          <th>Draws</th>
          <th>Win Rate</th>
        </tr>
      </thead>
      <tbody>
        {leaderboard_rows}
      </tbody>
    </table>
  </div>

  <div class="section-header">
    <h2>Computer vs Computer Records</h2>
  </div>
  <div class="table-wrapper">
    <table>
      <thead>
        <tr>
          <th>Matchup</th>
          <th>Agent A Wins</th>
          <th>Agent B Wins</th>
          <th>Draws</th>
          <th>Win Split</th>
          <th>Win Rate</th>
        </tr>
      </thead>
      <tbody>
        {matchup_rows}
      </tbody>
    </table>
  </div>

  <div class="section-header">
    <h2>Distributions</h2>
  </div>
  <div class="histograms-grid">
    {histogram_cards}
  </div>
</div>
</body>
</html>
"""


def build_report_html(
    results: Dict[str, Dict],
    games_per_pair: int,
    keys: Optional[Sequence[str]] = None,
    display_names: Optional[Dict[str, str]] = None,
    title: str = "Battleship AI Performance Benchmark",
) -> str:
    """Assemble the restrained light-mode HTML dashboard with insights."""
    from plotly.offline import get_plotlyjs

    if keys is None:
        from ai.tournament import DISPLAY_NAMES, STRATEGIES
        keys = list(STRATEGIES.keys())
        display_names = DISPLAY_NAMES
    if display_names is None:
        display_names = {k: k for k in keys}

    totals = _per_ai_totals(results, keys)
    sorted_keys = sorted(keys, key=lambda k: totals[k][4], reverse=True)

    # 1. Leaderboard Table Rows
    leaderboard_rows = ""
    for rank, k in enumerate(sorted_keys, 1):
        games, w, l, d, pct, avg_shots = totals[k]
        leaderboard_rows += f"""
        <tr>
          <td><span class="rank-badge">#{rank}</span></td>
          <td><strong>{html.escape(display_names[k])}</strong></td>
          <td>{games}</td>
          <td><strong>{w}</strong></td>
          <td>{l}</td>
          <td>{d}</td>
          <td><strong>{pct:.1%}</strong></td>
        </tr>
        """

    # 2. Matchup Table Rows (Green for larger portion, Red for smaller portion)
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    matchup_rows = ""
    for a, b in pairs:
        s = results[a][b]
        aw, bw, d = s["a_wins"], s["b_wins"], s["draws"]
        decided = aw + bw
        pct_a = aw / max(decided, 1)
        pct_b = bw / max(decided, 1)

        # Color green for larger portion (winner), red for smaller portion (loser)
        if aw > bw:
            color_a = _GREEN_WINNER
            color_b = _RED_LOSER
        elif bw > aw:
            color_a = _RED_LOSER
            color_b = _GREEN_WINNER
        else:
            color_a = _GREEN_WINNER
            color_b = _GREEN_WINNER

        matchup_rows += f"""
        <tr>
          <td><strong>{html.escape(display_names[a])}</strong> <span style="color: #94A3B8;">vs</span> <strong>{html.escape(display_names[b])}</strong></td>
          <td><strong style="color: {'#10B981' if aw >= bw else '#EF4444'};">{aw}</strong></td>
          <td><strong style="color: {'#10B981' if bw >= aw else '#EF4444'};">{bw}</strong></td>
          <td>{d}</td>
          <td>
            <div class="flex-align">
              <div class="bar-container">
                <div class="bar-fill-a" style="width: {pct_a*100:.1f}%; background: {color_a};"></div>
                <div class="bar-fill-b" style="width: {pct_b*100:.1f}%; background: {color_b};"></div>
              </div>
              <span style="font-size: 11px; color: #64748B; font-weight: 600;">{pct_a:.0%} / {pct_b:.0%}</span>
            </div>
          </td>
          <td><strong>{pct_a:.1%}</strong></td>
        </tr>
        """

    # 3. 4 Shot Efficiency Histograms (Uniform Vivid Blue) + Metrics
    histogram_cards = ""
    for idx, k in enumerate(keys):
        shot_data, min_shots, median_shots, mean_shots = _ai_shot_stats(k, results, keys)
        hist_html = _build_shot_efficiency_histogram(k, shot_data, display_names[k])

        histogram_cards += f"""
        <div class="histogram-card">
          <h3>{html.escape(display_names[k])} AI</h3>
          <div class="histogram-subhead"></div>
          {hist_html}
          <div class="insight-row">
            <div class="insight-pill">
              <label>Fastest Win</label>
              <span>{min_shots if min_shots > 0 else 'N/A'} shots</span>
            </div>
            <div class="insight-pill">
              <label>Median Shots</label>
              <span>{median_shots:.1f} shots</span>
            </div>
            <div class="insight-pill">
              <label>Average Shots</label>
              <span>{mean_shots:.1f} shots</span>
            </div>
          </div>
        </div>
        """

    return _PAGE_TEMPLATE.format(
        title=title,
        plotly_js=get_plotlyjs(),
        games_per_pair=games_per_pair,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        leaderboard_rows=leaderboard_rows,
        matchup_rows=matchup_rows,
        histogram_cards=histogram_cards,
    )


import webbrowser


def write_report(
    results: Dict[str, Dict],
    games_per_pair: int,
    path: str,
    keys: Optional[Sequence[str]] = None,
    display_names: Optional[Dict[str, str]] = None,
    auto_open: bool = True,
) -> str:
    """Build the restrained light-mode HTML report and write it to ``path``."""
    html_doc = build_report_html(
        results, games_per_pair, keys=keys, display_names=display_names,
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    abs_path = os.path.abspath(path)
    print(f"\n  Interactive HTML report written to {abs_path}")
    if auto_open:
        try:
            webbrowser.open(abs_path)
            print("  Opened HTML report in default browser.")
        except Exception as err:
            print(f"  [!] Note: Could not auto-open browser ({err})")
    return abs_path
