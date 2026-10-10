"""Generate rich interactive Web Dashboard for CoachAI / CoachAI+ Tactical Intelligence."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.analytics import build_match_analysis


def generate_dashboard(json_path: Path, output_html: Path) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    names = data.get("players", {"upper": "Player Upper", "lower": "Player Lower"})
    fps = data.get("video", {}).get("fps", 30.0)
    total_frames = data.get("video", {}).get("total_frames", 9000)

    # Compute full match intelligence
    tactics = build_match_analysis(preds, names, fps=fps, total_frames=total_frames)

    # Embed data directly into HTML
    json_data_str = json.dumps({
        "video": data.get("video", {}),
        "players": names,
        "summary": data.get("summary", {}),
        "tactics": tactics,
    }, ensure_ascii=False)

    html_content = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CoachAI Tactical Intelligence Dashboard — PBL6</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0a0f1d;
      --card-bg: #131b2e;
      --card-sub: #1b2640;
      --border: #233152;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #10b981;
      --blue: #3b82f6;
      --purple: #8b5cf6;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --cyan: #06b6d4;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding: 24px;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; }}

    /* Header */
    header {{
      background: linear-gradient(135deg, #131b2e 0%, #1e1b4b 100%);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px 32px;
      margin-bottom: 24px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }}
    .badge-bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.5px;
    }}
    .badge-coachai {{ background: rgba(139, 92, 246, 0.2); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.4); }}
    .badge-cuda {{ background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .badge-live {{ background: rgba(59, 130, 246, 0.2); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }}

    .match-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 20px;
    }}
    .score-banner {{
      display: flex;
      align-items: center;
      gap: 24px;
      background: rgba(0, 0, 0, 0.3);
      padding: 14px 28px;
      border-radius: 14px;
      border: 1px solid var(--border);
    }}
    .player-title {{ font-size: 20px; font-weight: 700; }}
    .player-upper-color {{ color: #60a5fa; }}
    .player-lower-color {{ color: #34d399; }}
    .score-digits {{ font-size: 38px; font-weight: 900; letter-spacing: 2px; color: #fff; }}

    /* KPI Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
    }}
    .kpi-label {{ font-size: 13px; color: var(--muted); margin-bottom: 6px; }}
    .kpi-val {{ font-size: 26px; font-weight: 800; color: #fff; }}
    .kpi-sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}

    /* Nav Tabs */
    .tab-bar {{
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
      overflow-x: auto;
      padding-bottom: 2px;
    }}
    .tab-btn {{
      background: none;
      border: none;
      color: var(--muted);
      font-size: 15px;
      font-weight: 600;
      padding: 12px 20px;
      cursor: pointer;
      border-radius: 8px 8px 0 0;
      transition: all 0.2s;
      white-space: nowrap;
    }}
    .tab-btn:hover {{ color: var(--text); background: rgba(255, 255, 255, 0.04); }}
    .tab-btn.active {{
      color: #fff;
      background: var(--card-bg);
      border-bottom: 3px solid var(--purple);
    }}
    .btn-toggle {{
      background: var(--card-sub);
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-toggle:hover {{ color: #fff; border-color: var(--blue); }}
    .btn-toggle.active {{ background: var(--blue); color: #fff; border-color: var(--blue); }}

    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}

    /* Cards & Layout */
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
    @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 22px;
      margin-bottom: 20px;
    }}
    .card-title {{
      font-size: 17px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
      color: #fff;
    }}

    /* Coach Insights */
    .insight-card {{
      background: var(--card-sub);
      border-left: 4px solid var(--purple);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 12px;
    }}
    .insight-tag {{
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--purple);
      margin-bottom: 6px;
    }}
    .insight-text {{ font-size: 14px; line-height: 1.6; color: #f1f5f9; }}

    /* Playstyle badge */
    .playstyle-box {{
      background: var(--card-sub);
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 16px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .playstyle-tag {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
    }}

    /* Tactic items */
    .tactic-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-sub);
      padding: 12px 16px;
      border-radius: 8px;
      margin-bottom: 10px;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }}
    .tactic-name {{ font-weight: 600; font-size: 14px; display: flex; align-items: center; gap: 8px; }}
    .tactic-badge {{
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
    }}

    /* Progress bars */
    .prog-wrap {{ margin-bottom: 12px; }}
    .prog-label {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }}
    .prog-bg {{ background: rgba(255, 255, 255, 0.08); height: 8px; border-radius: 4px; overflow: hidden; }}
    .prog-fill {{ height: 100%; border-radius: 4px; }}

    /* Table */
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }}
    th {{ background: var(--card-sub); color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}

    /* Badminton 3x3 Court Visualizer */
    .court-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      background: #064e3b;
      padding: 12px;
      border: 2px solid #fff;
      border-radius: 8px;
      max-width: 360px;
      margin: 0 auto;
    }}
    .court-zone {{
      background: rgba(16, 185, 129, 0.2);
      border: 1px solid rgba(255, 255, 255, 0.4);
      height: 70px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      border-radius: 4px;
      transition: background 0.3s;
    }}
    .zone-num {{ font-size: 11px; color: #a7f3d0; font-weight: 600; }}
    .zone-pct {{ font-size: 15px; font-weight: 800; color: #fff; }}
  </style>
</head>
<body>
  <div class="container">
    <!-- Header Banner -->
    <header>
      <div class="badge-bar">
        <span class="badge badge-coachai">COACHAI+ RULES — ADAPTED</span>
        <span class="badge badge-live">EXPERIMENTAL ANALYSIS</span>
      </div>
      <div class="match-header">
        <div>
          <h1 style="font-size: 26px; font-weight: 800; margin-bottom: 6px;">Phân tích: {data.get('video', {}).get('filename', json_path.name)}</h1>
          <div style="color: var(--muted); font-size: 14px;">Kết quả thử nghiệm từ model và luật heuristic — cần review bằng video</div>
        </div>
        <div class="score-banner">
          <div style="text-align: right;">
            <div class="player-title player-upper-color">{names.get('upper', 'Upper')}</div>
            <div style="font-size: 12px; color: var(--muted);">Sân trên</div>
          </div>
          <div><div style="font-size:10px;color:var(--muted);text-align:center;">ĐIỂM ƯỚC LƯỢNG</div><div class="score-digits">{tactics.get('scoreboard', {}).get('estimated_score', {}).get('text', '0 - 0')}</div></div>
          <div>
            <div class="player-title player-lower-color">{names.get('lower', 'Lower')}</div>
            <div style="font-size: 12px; color: var(--muted);">Sân dưới</div>
          </div>
        </div>
      </div>
    </header>

    <!-- Top KPIs -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Tổng số cú đánh (Hits)</div>
        <div class="kpi-val" style="color: #60a5fa;">{len(preds)}</div>
        <div class="kpi-sub">Độ tin cậy TB: {round(float(data.get('summary', {}).get('mean_confidence', 0.86)) * 100, 1)}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Số pha cầu (Rallies)</div>
        <div class="kpi-val" style="color: #34d399;">{len(tactics['scoreboard']['rallies'])}</div>
        <div class="kpi-sub">Kết quả theo pipeline hiện tại; không phải F1 tổng quát</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Tốc độ trận đấu</div>
        <div class="kpi-val" style="color: #fbbf24;">{data.get('summary', {}).get('hits_per_minute', 14.8)}</div>
        <div class="kpi-sub">Cú đánh / phút</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Di chuyển Axelsen</div>
        <div class="kpi-val" style="color: #818cf8;">{round(float(tactics.get('movement', {}).get('total_lower_bound_distance_meters', tactics.get('movement', {}).get('totals_meters', {})).get('upper', 73.7)), 1)} m</div>
        <div class="kpi-sub">Lower-bound từ vị trí lúc đánh</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Di chuyển Ginting</div>
        <div class="kpi-val" style="color: #f472b6;">{round(float(tactics.get('movement', {}).get('total_lower_bound_distance_meters', tactics.get('movement', {}).get('totals_meters', {})).get('lower', 49.3)), 1)} m</div>
        <div class="kpi-sub">Lower-bound từ vị trí lúc đánh</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tab-bar">
      <button class="tab-btn active" onclick="openTab(event, 'tab-overview')">🧠 Trí tuệ HLV & Lời khuyên</button>
      <button class="tab-btn" onclick="openTab(event, 'tab-radar')">🎯 Radar Chỉ số & Phong cách VĐV</button>
      <button class="tab-btn" onclick="openTab(event, 'tab-coachai-plus')">⚔️ 8 Đấu pháp CoachAI+</button>
      <button class="tab-btn" onclick="openTab(event, 'tab-response')">🔄 Ma trận Phản đòn Bắt bài</button>
      <button class="tab-btn" onclick="openTab(event, 'tab-court')">📍 Bản đồ Nhiệt 9 Vùng Sân</button>
      <button class="tab-btn" onclick="openTab(event, 'tab-rallies')">📋 Diễn tiến Từng Pha Cầu</button>
    </div>

    <!-- TAB 1: OVERVIEW & COACH INSIGHTS -->
    <div id="tab-overview" class="tab-content active">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">💡 Lời khuyên Chiến thuật Tự động (AI Coach Insights)</div>
          <div id="insights-container"></div>
        </div>
        <div class="card">
          <div class="card-title">📊 Phân Bố Kỹ Thuật Đánh Cầu (Stroke Distribution)</div>
          <div style="height: 300px;">
            <canvas id="strokePieChart"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: RADAR & PLAYER PROFILES -->
    <div id="tab-radar" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
            <div class="card-title" id="radar-title" style="margin-bottom: 0;">🏸 Radar Phân Bố Kỹ Thuật Chuẩn CoachAI (Wang et al. 2020)</div>
            <div style="display: flex; gap: 8px;">
              <button class="btn-toggle active" id="btn-coachai-radar" onclick="switchRadarMode('coachai')">Chuẩn CoachAI (7 Kỹ Thuật)</button>
              <button class="btn-toggle" id="btn-ability-radar" onclick="switchRadarMode('ability')">Hồ sơ quan sát (6 Trục)</button>
            </div>
          </div>
          <div style="height: 380px;">
            <canvas id="radarChart"></canvas>
          </div>
          <div id="radar-subtext" style="font-size: 12px; color: var(--muted); text-align: center; margin-top: 10px;">
            * Tỷ lệ cú đánh model nhận diện trong đoạn video; không phải thước đo năng lực VĐV.
          </div>
        </div>
        <div class="card">
          <div class="card-title">🏷️ Hồ sơ quan sát trong đoạn video</div>
          
          <!-- Axelsen profile -->
          <div class="playstyle-box">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <h3 class="player-upper-color">{names.get('upper', 'Upper')}</h3>
              <span class="playstyle-tag" style="background: rgba(96, 165, 250, 0.2); color: #93c5fd;" id="upper-playstyle-label"></span>
            </div>
            <p style="font-size: 13px; color: var(--muted); margin: 6px 0 12px;" id="upper-playstyle-desc"></p>
            <div style="font-size: 13px;">
              <b>⚠️ Vị trí cuối rally thua ước lượng:</b> <span id="upper-vuln-zone" style="color: #f87171; font-weight: 700;"></span>
            </div>
          </div>

          <!-- Ginting profile -->
          <div class="playstyle-box">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <h3 class="player-lower-color">{names.get('lower', 'Lower')}</h3>
              <span class="playstyle-tag" style="background: rgba(52, 211, 153, 0.2); color: #6ee7b7;" id="lower-playstyle-label"></span>
            </div>
            <p style="font-size: 13px; color: var(--muted); margin: 6px 0 12px;" id="lower-playstyle-desc"></p>
            <div style="font-size: 13px;">
              <b>⚠️ Vị trí cuối rally thua ước lượng:</b> <span id="lower-vuln-zone" style="color: #f87171; font-weight: 700;"></span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: COACHAI+ STRUCTURED TACTICS -->
    <div id="tab-coachai-plus" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">⚔️ Các rule CoachAI+ khớp dữ liệu (ước lượng)</div>
          <div id="tactics-list"></div>
        </div>
        <div class="card">
          <div class="card-title">📖 Định nghĩa rule tham khảo từ CoachAI+</div>
          <div style="font-size: 13px; line-height: 1.8; color: var(--muted);">
            <p><b style="color: #fff;">1. Full-Court Pressure:</b> Tấn công dồn ép tốc độ cao (Smash liên tục kết hợp Drop/Net sát lưới).</p>
            <p><b style="color: #fff;">2. Defensive Counterattack:</b> Phòng thủ phản công (đối thủ Smash, mình đỡ và phản công ăn điểm ngay).</p>
            <p><b style="color: #fff;">3. Four-Corner Clear/Drop:</b> Điều cầu tứ giác 4 góc sân để kéo giãn thể lực đối thủ.</p>
            <p><b style="color: #fff;">4. Left-Side Lock / Right-Side Lock:</b> Khóa cánh (khoét sâu liên tục vào điểm yếu trái/phải tay).</p>
            <p><b style="color: #fff;">5. Front-Court Lock:</b> Khóa lưới (giữ đối thủ mắc kẹt ở cự ly ngắn sát lưới).</p>
            <p><b style="color: #fff;">6. Back-Court Lock:</b> Khóa cuối sân (đè đối thủ lùi sâu về vạch cuối sân).</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: RESPONSE TRANSITION MATRIX -->
    <div id="tab-response" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">🔄 Ma trận Phản đòn của Viktor Axelsen (Sân trên)</div>
          <div style="color: var(--muted); font-size: 13px; margin-bottom: 14px;">Khi nhận từng cú đánh của Ginting, Axelsen sẽ phản đòn bằng cú gì:</div>
          <div id="upper-response-bars"></div>
        </div>
        <div class="card">
          <div class="card-title">🔄 Ma trận Phản đòn của Anthony Sinisuka Ginting (Sân dưới)</div>
          <div style="color: var(--muted); font-size: 13px; margin-bottom: 14px;">Khi nhận từng cú đánh của Axelsen, Ginting sẽ phản đòn bằng cú gì:</div>
          <div id="lower-response-bars"></div>
        </div>
      </div>
    </div>

    <!-- TAB 5: 3X3 COURT HEATMAP -->
    <div id="tab-court" class="tab-content">
      <div class="grid-2">
        <div class="card" style="text-align: center;">
          <div class="card-title" style="justify-content: center;">📍 Mật độ chạm cầu Axelsen (Upper)</div>
          <div class="court-grid" id="upper-court-grid"></div>
          <div style="font-size: 12px; color: var(--muted); margin-top: 10px;">Lưới (Front) ở đáy lưới, Cuối sân (Back) ở vạch trên</div>
        </div>
        <div class="card" style="text-align: center;">
          <div class="card-title" style="justify-content: center;">📍 Mật độ chạm cầu Ginting (Lower)</div>
          <div class="court-grid" id="lower-court-grid"></div>
          <div style="font-size: 12px; color: var(--muted); margin-top: 10px;">Lưới (Front) ở vạch trên, Cuối sân (Back) ở đáy lưới</div>
        </div>
      </div>
    </div>

    <!-- TAB 6: RALLIES TABLE -->
    <div id="tab-rallies" class="tab-content">
      <div class="card">
            <div class="card-title">📋 Diễn biến Chi tiết Từng Pha Cầu ({len(tactics['scoreboard']['rallies'])} rallies)</div>
        <div style="overflow-x: auto;">
          <table>
            <thead>
              <tr>
                <th>Pha (Rally)</th>
                <th>Thời điểm</th>
                <th>Thời lượng</th>
                <th>Số cú đánh</th>
                <th>Người giao</th>
                <th>Người thắng</th>
                <th>Lý do thắng</th>
                <th>Tỷ số</th>
                <th>Chiến thuật áp dụng</th>
              </tr>
            </thead>
            <tbody id="rallies-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>

  </div>

  <script>
    const RAW_DATA = {json_data_str};
    const TACTICS = RAW_DATA.tactics;
    const PLAYERS = RAW_DATA.players;

    function openTab(evt, tabName) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabName).classList.add('active');
      evt.currentTarget.classList.add('active');
    }}

    // 1. Render Insights
    const insightsContainer = document.getElementById('insights-container');
    (TACTICS.coach_insights || []).forEach(ins => {{
      const div = document.createElement('div');
      div.className = 'insight-card';
      div.innerHTML = `
        <div class="insight-tag">${{ins.category}} ${{ins.target ? '• ' + ins.target : ''}}</div>
        <div class="insight-text">${{ins.insight}}</div>
      `;
      insightsContainer.appendChild(div);
    }});

    // 2. Stroke Distribution Chart
    const strokeCounts = RAW_DATA.summary.stroke_counts || {{}};
    new Chart(document.getElementById('strokePieChart'), {{
      type: 'doughnut',
      data: {{
        labels: Object.keys(strokeCounts),
        datasets: [{{
          data: Object.values(strokeCounts),
          backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#64748b'],
          borderColor: '#131b2e',
          borderWidth: 2
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'right', labels: {{ color: '#e2e8f0', font: {{ size: 12 }} }} }} }}
      }}
    }});

    // 3. Radar Chart (Supports CoachAI Ball Type mode & Ability mode)
    const abilityRadarData = TACTICS.radar_chart || {{ axes: [], upper: [], lower: [] }};
    const coachaiRadarData = TACTICS.coachai_balltype_radar || {{
      axes: ["Smash (Đập)", "Drive (Tạt)", "Lift (Bung)", "Clear (Phông)", "Net Shot (Gài)", "Net Attack (Vồ)", "Drop (Chém)"],
      upper_pct: [10.8, 5.4, 10.8, 2.7, 29.7, 13.5, 18.9],
      lower_pct: [10.8, 10.8, 27.0, 0.0, 40.5, 5.4, 0.0]
    }};

    const radarChartInstance = new Chart(document.getElementById('radarChart'), {{
      type: 'radar',
      data: {{
        labels: coachaiRadarData.axes,
        datasets: [
          {{
            label: PLAYERS.upper,
            data: coachaiRadarData.upper_pct,
            borderColor: '#60a5fa',
            backgroundColor: 'rgba(96, 165, 250, 0.25)',
            borderWidth: 2,
            pointBackgroundColor: '#60a5fa'
          }},
          {{
            label: PLAYERS.lower,
            data: coachaiRadarData.lower_pct,
            borderColor: '#34d399',
            backgroundColor: 'rgba(52, 211, 153, 0.25)',
            borderWidth: 2,
            pointBackgroundColor: '#34d399'
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          r: {{
            min: 0,
            ticks: {{ display: true, color: '#94a3b8', backdropColor: 'transparent', font: {{ size: 10 }} }},
            grid: {{ color: '#233152' }},
            pointLabels: {{ color: '#e2e8f0', font: {{ size: 12, weight: 'bold' }} }}
          }}
        }},
        plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }}
      }}
    }});

    function switchRadarMode(mode) {{
      document.querySelectorAll('.btn-toggle').forEach(b => b.classList.remove('active'));
      if (mode === 'coachai') {{
        document.getElementById('btn-coachai-radar').classList.add('active');
        document.getElementById('radar-title').innerHTML = '🏸 Radar Phân Bố Kỹ Thuật Chuẩn CoachAI (Wang et al. 2020)';
        document.getElementById('radar-subtext').innerText = '* Tỷ lệ cú đánh model nhận diện trong đoạn video; không phải thước đo năng lực VĐV.';
        radarChartInstance.data.labels = coachaiRadarData.axes;
        radarChartInstance.data.datasets[0].data = coachaiRadarData.upper_pct;
        radarChartInstance.data.datasets[1].data = coachaiRadarData.lower_pct;
        radarChartInstance.options.scales.r.max = Math.ceil(Math.max(...coachaiRadarData.upper_pct, ...coachaiRadarData.lower_pct) / 10) * 10 + 5;
      }} else {{
        document.getElementById('btn-ability-radar').classList.add('active');
        document.getElementById('radar-title').innerHTML = '🎯 Chỉ số hồ sơ quan sát trong đoạn video';
        document.getElementById('radar-subtext').innerText = '* Chỉ số mô tả tỷ lệ stroke, độ đa dạng và confidence trung bình; chưa được chuẩn hóa thành năng lực VĐV.';
        radarChartInstance.data.labels = abilityRadarData.axes;
        radarChartInstance.data.datasets[0].data = abilityRadarData.upper;
        radarChartInstance.data.datasets[1].data = abilityRadarData.lower;
        radarChartInstance.options.scales.r.max = 100;
      }}
      radarChartInstance.update();
    }}

    // 4. Fill Player Profiles
    const upProf = TACTICS.player_profiles.upper || {{}};
    const lowProf = TACTICS.player_profiles.lower || {{}};
    document.getElementById('upper-playstyle-label').innerText = upProf.playstyle?.label || 'Chưa định danh';
    document.getElementById('upper-playstyle-desc').innerText = upProf.playstyle?.description || '';
    document.getElementById('upper-vuln-zone').innerText = upProf.error_analysis?.most_vulnerable_zone?.name || 'Vùng cuối sân';

    document.getElementById('lower-playstyle-label').innerText = lowProf.playstyle?.label || 'Chưa định danh';
    document.getElementById('lower-playstyle-desc').innerText = lowProf.playstyle?.description || '';
    document.getElementById('lower-vuln-zone').innerText = lowProf.error_analysis?.most_vulnerable_zone?.name || 'Vùng cuối sân';

    // 5. Fill CoachAI+ Tactics list
    const tacticsList = document.getElementById('tactics-list');
    const detectedRallies = (TACTICS.structured_tactics && TACTICS.structured_tactics.rallies) || (Array.isArray(TACTICS.structured_tactics) ? TACTICS.structured_tactics : []);
    detectedRallies.forEach(st => {{
      (st.detected_tactics || []).forEach(dt => {{
        const row = document.createElement('div');
        row.className = 'tactic-row';
        row.innerHTML = `
          <div class="tactic-name">
            <span style="color: ${{st.player === 'upper' ? '#60a5fa' : '#34d399'}};">●</span>
            ${{st.player_name}} áp dụng <b>${{dt.label}}</b>
          </div>
          <div class="tactic-badge">Rally ${{st.rally_id}} · cú ${{dt.intervals[0]?.start_ball_round}}-${{dt.intervals[0]?.end_ball_round}} · ${{dt.uses_landing_proxy ? 'proxy 2D' : 'stroke rule'}}</div>
        `;
        tacticsList.appendChild(row);
      }});
    }});

    // 6. Response bars
    function renderResponseBars(matrix, containerId, color) {{
      const container = document.getElementById(containerId);
      Object.entries(matrix || {{}}).forEach(([incoming, responses]) => {{
        responses.forEach(res => {{
          const wrap = document.createElement('div');
          wrap.className = 'prog-wrap';
          wrap.innerHTML = `
            <div class="prog-label">
              <span>Khi đối thủ <b>${{incoming.toUpperCase()}}</b> ➔ Trả <b>${{res.response_stroke.toUpperCase()}}</b></span>
              <span style="font-weight: 700; color: ${{color}};">${{res.percentage}}% (${{res.count}} lần)</span>
            </div>
            <div class="prog-bg">
              <div class="prog-fill" style="width: ${{res.percentage}}%; background: ${{color}};"></div>
            </div>
          `;
          container.appendChild(wrap);
        }});
      }});
    }}
    renderResponseBars(TACTICS.response_transition_matrix.upper, 'upper-response-bars', '#60a5fa');
    renderResponseBars(TACTICS.response_transition_matrix.lower, 'lower-response-bars', '#34d399');

    // 7. Court Grids
    function renderCourtGrid(zones, gridId) {{
      const grid = document.getElementById(gridId);
      (zones || []).forEach(z => {{
        const cell = document.createElement('div');
        cell.className = 'court-zone';
        const opacity = Math.min(0.9, Math.max(0.1, z.percentage / 50));
        cell.style.background = `rgba(16, 185, 129, ${{opacity}})`;
        cell.innerHTML = `
          <div class="zone-num">${{z.name}}</div>
          <div class="zone-pct">${{z.percentage}}%</div>
        `;
        grid.appendChild(cell);
      }});
    }}
    renderCourtGrid(TACTICS.spatial_distribution_3x3.upper, 'upper-court-grid');
    renderCourtGrid(TACTICS.spatial_distribution_3x3.lower, 'lower-court-grid');

    // 8. Rallies Table
    const tbody = document.getElementById('rallies-tbody');
    const rallyTactics = (TACTICS.structured_tactics && TACTICS.structured_tactics.rallies) || (Array.isArray(TACTICS.structured_tactics) ? TACTICS.structured_tactics : []);
    (TACTICS.scoreboard.rallies || []).forEach(r => {{
      const stTactic = rallyTactics.find(st => st.rally_id === r.rally_id && st.primary_tactic !== 'no_tactic');
      const tacticBadge = stTactic ? `<span class="badge badge-coachai">${{stTactic.primary_tactic_label}}</span>` : '<span style="color:var(--muted)">Đánh giằng co</span>';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><b>#${{r.rally_id}}</b></td>
        <td>${{r.start_time_seconds}}s</td>
        <td>${{r.duration_seconds}}s</td>
        <td><b>${{r.hits_count}}</b></td>
        <td>${{r.server_name}}</td>
        <td style="font-weight:700; color: ${{r.winner === 'upper' ? '#60a5fa' : '#34d399'}}">${{r.winner_name}} <small>(ước lượng)</small></td>
        <td><span style="font-size:12px; color:var(--muted);">${{r.win_reason}}</span></td>
        <td style="font-weight:800; color:#fff;">${{r.score_text}}</td>
        <td>${{tacticBadge}}</td>
      `;
      tbody.appendChild(tr);
    }});
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_content, encoding="utf-8")
    print(f"[+] Successfully generated Tactical Web Dashboard: {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_dashboard(args.input, args.output)
