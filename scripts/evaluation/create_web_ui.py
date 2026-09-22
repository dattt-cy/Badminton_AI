import json

compact = json.load(open('work_dirs/test_match40_10m_15m_full_pipeline_epoch20/compact_timeline.json'))
compact_str = json.dumps(compact)

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Badminton AI Action Recognition - Epoch 20 Fusion Pipeline</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .glass {{
      background: rgba(255, 255, 255, 0.03);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .badge-serve {{ background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }}
    .badge-smash {{ background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
    .badge-clear {{ background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .badge-drop {{ background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .badge-lift {{ background-color: rgba(139, 92, 246, 0.2); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.4); }}
    .badge-drive {{ background-color: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.4); }}
    .badge-net_shot {{ background-color: rgba(20, 184, 166, 0.2); color: #2dd4bf; border: 1px solid rgba(20, 184, 166, 0.4); }}
    .badge-net_attack {{ background-color: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }}
    
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.2); }}
    ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.3); }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] min-h-screen font-sans p-4 md:p-8 antialiased">

  <div class="max-w-7xl mx-auto space-y-6">

    <!-- Header & System Architecture Badge -->
    <header class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 pb-6 border-b border-[var(--border)]">
      <div>
        <div class="flex items-center gap-3">
          <span class="px-2.5 py-1 text-xs font-semibold uppercase tracking-wider rounded-md bg-emerald-500/20 text-emerald-400 border border-emerald-500/40">Production Checkpoint</span>
          <span class="text-xs text-[var(--muted-foreground)]">Full Multisense Pipeline v2.5</span>
        </div>
        <h1 class="text-2xl md:text-3xl font-bold tracking-tight mt-1.5 text-[var(--foreground)]">
          Badminton AI Analyzer & Action Classifier
        </h1>
        <p class="text-sm text-[var(--muted-foreground)] mt-1">
          Hệ thống phát hiện tiếp xúc & phân loại cú đánh cầu lông đa phương thức (RGB + Trajectory Fusion + Physics Engine).
        </p>
      </div>

      <div class="flex items-center gap-3">
        <div class="text-right hidden sm:block">
          <div class="text-xs text-[var(--muted-foreground)]">Độ chính xác trên Match 40 (5 phút)</div>
          <div class="text-lg font-bold text-emerald-400">80.9% Stroke | 70.6% Joint</div>
        </div>
        <button id="btnLoadPrecomputed" onclick="loadPrecomputedDataset()" class="px-4 py-2 text-sm font-medium rounded-lg bg-[var(--primary)] text-[var(--primary-foreground)] hover:opacity-90 transition shadow-sm flex items-center gap-2">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
          Nạp sẵn Match 40 (5m)
        </button>
      </div>
    </header>

    <!-- Pipeline Architecture Cards -->
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
      <div class="glass p-3.5 rounded-xl border border-[var(--border)]">
        <div class="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">1. HIT Detector</div>
        <div class="text-sm font-bold mt-1 text-[var(--foreground)]">R(2+1)D-18 Temporal</div>
        <div class="text-xs text-emerald-400 mt-0.5">Coverage: 81.9%</div>
      </div>
      <div class="glass p-3.5 rounded-xl border border-[var(--border)]">
        <div class="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">2. Candidate Filter</div>
        <div class="text-sm font-bold mt-1 text-[var(--foreground)]">Random Forest (RF)</div>
        <div class="text-xs text-blue-400 mt-0.5">Threshold: 0.55 (NMS 2s)</div>
      </div>
      <div class="glass p-3.5 rounded-xl border border-[var(--border)]">
        <div class="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">3. Shuttle Tracker</div>
        <div class="text-sm font-bold mt-1 text-[var(--foreground)]">FastTrackNet V3</div>
        <div class="text-xs text-purple-400 mt-0.5">In-memory 64f buffer</div>
      </div>
      <div class="glass p-3.5 rounded-xl border border-[var(--border)]">
        <div class="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">4. Feature Fusion</div>
        <div class="text-sm font-bold mt-1 text-[var(--foreground)]">Epoch 20 Fusion Head</div>
        <div class="text-xs text-amber-400 mt-0.5">F1 > 0.8026 (Offline)</div>
      </div>
      <div class="glass p-3.5 rounded-xl border border-[var(--border)]">
        <div class="text-[11px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">5. Consensus & Physics</div>
        <div class="text-sm font-bold mt-1 text-[var(--foreground)]">[-4, 0, +4] Multi-Offset</div>
        <div class="text-xs text-emerald-400 mt-0.5">Court geometry bounds</div>
      </div>
    </div>

    <!-- Input Video & Execution Control Section -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      
      <!-- Video Input / Path Box (Left 5 Cols) -->
      <div class="lg:col-span-5 space-y-4">
        <div class="glass p-5 rounded-2xl border border-[var(--border)] space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-base font-semibold text-[var(--foreground)] flex items-center gap-2">
              <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
              Đầu vào Video
            </h2>
            <span class="text-xs px-2 py-0.5 rounded bg-[var(--muted)] text-[var(--muted-foreground)] font-mono">Local Direct Path</span>
          </div>

          <!-- Input Mode Tabs -->
          <div class="space-y-2">
            <label class="text-xs font-medium text-[var(--muted-foreground)]">Đường dẫn tệp video trên máy (Local Absolute Path):</label>
            <div class="flex gap-2">
              <input type="text" id="videoPathInput" 
                value="E:\HK1-2026\PBL6\AI_Classifier\work_dirs\test_match40_10m_15m.mp4" 
                placeholder="Dán đường dẫn: C:\path\to\video.mp4"
                class="flex-1 px-3 py-2 text-sm bg-[var(--card)] border border-[var(--border)] rounded-lg text-[var(--foreground)] placeholder-[var(--placeholder)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)] font-mono"
              />
            </div>
          </div>

          <div class="space-y-2">
            <label class="text-xs font-medium text-[var(--muted-foreground)]">Hoặc tải video lên trình duyệt:</label>
            <input type="file" id="videoFileInput" accept="video/*" onchange="handleFileUpload(event)"
              class="block w-full text-xs text-[var(--muted-foreground)] file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-[var(--secondary)] file:text-[var(--secondary-foreground)] hover:file:opacity-80 cursor-pointer"
            />
          </div>

          <!-- Video Preview Container -->
          <div class="rounded-xl overflow-hidden bg-black/40 border border-[var(--border)] relative aspect-video flex items-center justify-center">
            <video id="mainVideoPlayer" controls class="w-full h-full object-contain hidden"></video>
            <div id="videoPlaceholder" class="text-center p-6 space-y-2">
              <svg class="w-10 h-10 mx-auto text-[var(--muted-foreground)] opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              <div class="text-xs text-[var(--muted-foreground)]">Xem trước video và nhảy frame chạm vợt</div>
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="flex gap-2 pt-2">
            <button onclick="triggerRunAnalysis()" class="flex-1 py-2.5 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition shadow flex items-center justify-center gap-2">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path></svg>
              Chạy Pipeline Phân Tích
            </button>
            <button onclick="loadPrecomputedDataset()" class="py-2.5 px-4 rounded-xl bg-[var(--secondary)] text-[var(--secondary-foreground)] hover:opacity-90 font-medium text-sm transition">
              Nạp Kết Quả Mẫu
            </button>
          </div>
        </div>

        <!-- Metric Card Overview -->
        <div class="glass p-5 rounded-2xl border border-[var(--border)] space-y-4">
          <h3 class="text-sm font-semibold text-[var(--foreground)]">Chỉ Số Tổng Quan (Match 40 Ground Truth)</h3>
          <div class="grid grid-cols-2 gap-3">
            <div class="p-3 rounded-xl bg-black/20 border border-[var(--border)]">
              <div class="text-xs text-[var(--muted-foreground)]">Stroke Accuracy</div>
              <div class="text-xl font-bold text-emerald-400 mt-1" id="metricStrokeAcc">80.9%</div>
              <div class="text-[11px] text-[var(--muted-foreground)] mt-0.5">55 / 68 cú đánh đúng</div>
            </div>
            <div class="p-3 rounded-xl bg-black/20 border border-[var(--border)]">
              <div class="text-xs text-[var(--muted-foreground)]">Side Accuracy</div>
              <div class="text-xl font-bold text-blue-400 mt-1" id="metricSideAcc">83.8%</div>
              <div class="text-[11px] text-[var(--muted-foreground)] mt-0.5">57 / 68 hướng tay đúng</div>
            </div>
            <div class="p-3 rounded-xl bg-black/20 border border-[var(--border)]">
              <div class="text-xs text-[var(--muted-foreground)]">Joint Accuracy</div>
              <div class="text-xl font-bold text-amber-400 mt-1" id="metricJointAcc">70.6%</div>
              <div class="text-[11px] text-[var(--muted-foreground)] mt-0.5">48 / 68 đúng trọn vẹn cả hai</div>
            </div>
            <div class="p-3 rounded-xl bg-black/20 border border-[var(--border)]">
              <div class="text-xs text-[var(--muted-foreground)]">Hit Coverage</div>
              <div class="text-xl font-bold text-purple-400 mt-1" id="metricCoverage">81.9%</div>
              <div class="text-[11px] text-[var(--muted-foreground)] mt-0.5">68 / 83 điểm chạm BWF</div>
            </div>
          </div>
        </div>

      </div>

      <!-- Timeline & Event Analytics (Right 7 Cols) -->
      <div class="lg:col-span-7 space-y-4">
        
        <!-- Filter & Search Bar -->
        <div class="glass p-4 rounded-2xl border border-[var(--border)] flex flex-wrap items-center justify-between gap-3">
          <div class="flex items-center gap-2">
            <h2 class="text-base font-semibold text-[var(--foreground)]">Timeline Cú Đánh</h2>
            <span id="eventCountBadge" class="text-xs px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-400 font-semibold border border-indigo-500/30">70 Events</span>
          </div>

          <div class="flex items-center gap-2 text-xs">
            <label class="text-[var(--muted-foreground)]">Lọc lớp:</label>
            <select id="strokeFilter" onchange="filterEvents()" class="px-2.5 py-1.5 rounded-lg bg-[var(--card)] border border-[var(--border)] text-[var(--foreground)] focus:outline-none">
              <option value="all">Tất cả cú đánh</option>
              <option value="serve">Serve (Giao cầu)</option>
              <option value="clear">Clear (Phông cầu)</option>
              <option value="smash">Smash (Đập cầu)</option>
              <option value="drop">Drop (Bỏ nhỏ)</option>
              <option value="lift">Lift (Bung cầu)</option>
              <option value="drive">Drive (Tạt cầu)</option>
              <option value="net_shot">Net Shot (Gài lưới)</option>
              <option value="net_attack">Net Attack (Vồ lưới)</option>
            </select>
          </div>
        </div>

        <!-- Event List Table -->
        <div class="glass rounded-2xl border border-[var(--border)] overflow-hidden">
          <div class="max-h-[600px] overflow-y-auto divide-y divide-[var(--border)]" id="timelineListContainer">
            <!-- Dynamic rows will be inserted here -->
          </div>
        </div>

        <!-- Per-Class Performance Breakdown Table -->
        <div class="glass p-5 rounded-2xl border border-[var(--border)] space-y-3">
          <h3 class="text-sm font-semibold text-[var(--foreground)]">Thống Kê Chi Tiết Từng Lớp Trên Match 40</h3>
          <div class="overflow-x-auto">
            <table class="w-full text-xs text-left">
              <thead>
                <tr class="text-[var(--muted-foreground)] border-b border-[var(--border)]">
                  <th class="py-2 px-3 font-semibold">Loại Cú Đánh</th>
                  <th class="py-2 px-3 font-semibold">Số lượng GT</th>
                  <th class="py-2 px-3 font-semibold">Dự đoán đúng</th>
                  <th class="py-2 px-3 font-semibold">Độ chính xác</th>
                  <th class="py-2 px-3 font-semibold">Đặc trưng nhận diện</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-[var(--border)] font-mono">
                <tr><td class="py-2 px-3 font-sans font-medium text-emerald-400">Clear (Phông sâu)</td><td class="py-2 px-3">14</td><td class="py-2 px-3">14</td><td class="py-2 px-3 font-bold text-emerald-400">100.0%</td><td class="py-2 px-3 font-sans text-[var(--muted-foreground)]">Quỹ đạo parabol cao về cuối sân</td></tr>
                <tr><td class="py-2 px-3 font-sans font-medium text-purple-400">Lift (Bung thủ)</td><td class="py-2 px-3">10</td><td class="py-2 px-3">10</td><td class="py-2 px-3 font-bold text-emerald-400">100.0%</td><td class="py-2 px-3 font-sans text-[var(--muted-foreground)]">Hất từ dưới lưới vọt lên cao</td></tr>
                <tr><td class="py-2 px-3 font-sans font-medium text-blue-400">Serve (Phát bóng)</td><td class="py-2 px-3">3</td><td class="py-2 px-3">3</td><td class="py-2 px-3 font-bold text-emerald-400">100.0%</td><td class="py-2 px-3 font-sans text-[var(--muted-foreground)]">Khởi phát pha rally, vận tốc đều</td></tr>
                <tr><td class="py-2 px-3 font-sans font-medium text-amber-400">Net Attack (Vồ lưới)</td><td class="py-2 px-3">6</td><td class="py-2 px-3">5</td><td class="py-2 px-3 font-bold text-emerald-400">83.3%</td><td class="py-2 px-3 font-sans text-[var(--muted-foreground)]">Đè cầu ngang mép lưới tốc độ cao</td></tr>
                <tr><td class="py-2 px-3 font-sans font-medium text-teal-400">Net Shot (Gài nhỏ)</td><td class="py-2 px-3">18</td><td class="py-2 px-3">13</td><td class="py-2 px-3 font-bold text-blue-400">72.2%</td><td class="py-2 px-3 font-sans text-[var(--muted-foreground)]">Cầu chạm vợt êm, rơi sát lưới</td></tr>
                <tr><td class="py-2 px-3 font-sans font-medium text-red-400">Smash (Đập cầu)</td><td class="py-2 px-3">12</td><td class="py-2 px-3">8</td><td class="py-2 px-3 font-bold text-amber-400">66.7%</td><td class="py-2 px-3 font-sans text-[var(--muted-foreground)]">Gia tốc cực lớn, cắm thẳng xuống sân</td></tr>
                <tr><td class="py-2 px-3 font-sans font-medium text-pink-400">Drive (Tạt cầu)</td><td class="py-2 px-3">2</td><td class="py-2 px-3">1</td><td class="py-2 px-3 font-bold text-amber-400">50.0%</td><td class="py-2 px-3 font-sans text-[var(--muted-foreground)]">Đánh ngang ngực, quỹ đạo bay phẳng</td></tr>
                <tr><td class="py-2 px-3 font-sans font-medium text-yellow-400">Drop (Chặt/bỏ nhỏ)</td><td class="py-2 px-3">3</td><td class="py-2 px-3">1</td><td class="py-2 px-3 font-bold text-rose-400">33.3%</td><td class="py-2 px-3 font-sans text-[var(--muted-foreground)]">Động tác giống smash nhưng hãm lực</td></tr>
              </tbody>
            </table>
          </div>
        </div>

      </div>

    </div>

  </div>

  <script>
    // Embedded Precomputed Ground Truth Results for Match 40 (5m)
    const PRECOMPUTED_EVENTS = {compact_str};

    let currentEvents = [];

    function renderEvents(events) {{
      const container = document.getElementById('timelineListContainer');
      container.innerHTML = '';

      if (!events || events.length === 0) {{
        container.innerHTML = '<div class="p-8 text-center text-xs text-[var(--muted-foreground)]">Không có cú đánh nào phù hợp bộ lọc.</div>';
        return;
      }}

      events.forEach((ev, idx) => {{
        const row = document.createElement('div');
        row.className = 'p-3.5 hover:bg-white/[0.04] transition flex items-center justify-between gap-3 text-xs cursor-pointer';
        row.onclick = () => seekVideoToTime(ev.time);

        const badgeClass = `badge-${{ev.stroke}}`;
        const timeFormatted = `${{Math.floor(ev.time / 60)}}:${{Math.floor(ev.time % 60).toString().padStart(2, '0')}}`;
        const playerSideBadge = ev.player === 'lower' 
          ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">Sân dưới</span>' 
          : '<span class="text-[10px] px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">Sân trên</span>';

        row.innerHTML = `
          <div class="flex items-center gap-3">
            <span class="font-mono text-[var(--muted-foreground)] w-7 text-right">#${{idx + 1}}</span>
            <span class="font-mono px-2 py-0.5 rounded bg-black/30 border border-[var(--border)] text-[var(--foreground)]">${{timeFormatted}} (${{ev.frame}}f)</span>
            ${{playerSideBadge}}
            <span class="font-semibold text-sm px-2.5 py-0.5 rounded-md ${{badgeClass}} uppercase tracking-wider">${{ev.stroke.replace('_', ' ')}}</span>
            <span class="text-[var(--muted-foreground)] font-medium capitalize font-mono">${{ev.side}}</span>
          </div>

          <div class="flex items-center gap-3">
            <div class="text-right">
              <span class="text-[11px] font-mono text-emerald-400">${{Math.round(ev.stroke_prob * 100)}}%</span>
              <span class="text-[10px] text-[var(--muted-foreground)]">conf</span>
            </div>
            <button class="p-1.5 rounded-lg bg-[var(--secondary)] hover:bg-[var(--primary)] hover:text-white transition" title="Nhảy đến khung hình này">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path></svg>
            </button>
          </div>
        `;
        container.appendChild(row);
      }});

      document.getElementById('eventCountBadge').innerText = `${{events.length}} Events`;
    }}

    function filterEvents() {{
      const filter = document.getElementById('strokeFilter').value;
      if (filter === 'all') {{
        renderEvents(currentEvents);
      }} else {{
        const filtered = currentEvents.filter(e => e.stroke === filter);
        renderEvents(filtered);
      }}
    }}

    function loadPrecomputedDataset() {{
      currentEvents = [...PRECOMPUTED_EVENTS];
      renderEvents(currentEvents);
      document.getElementById('videoPathInput').value = "E:\\\\HK1-2026\\\\PBL6\\\\AI_Classifier\\\\work_dirs\\\\test_match40_10m_15m.mp4";
      
      // Update UI Metrics
      document.getElementById('metricStrokeAcc').innerText = "80.9%";
      document.getElementById('metricSideAcc').innerText = "83.8%";
      document.getElementById('metricJointAcc').innerText = "70.6%";
      document.getElementById('metricCoverage').innerText = "81.9%";
    }}

    function handleFileUpload(event) {{
      const file = event.target.files[0];
      if (file) {{
        const videoPlayer = document.getElementById('mainVideoPlayer');
        const placeholder = document.getElementById('videoPlaceholder');
        videoPlayer.src = URL.createObjectURL(file);
        videoPlayer.classList.remove('hidden');
        placeholder.classList.add('hidden');
        document.getElementById('videoPathInput').value = file.name;
      }}
    }}

    function seekVideoToTime(seconds) {{
      const videoPlayer = document.getElementById('mainVideoPlayer');
      if (videoPlayer && !videoPlayer.classList.contains('hidden')) {{
        videoPlayer.currentTime = seconds;
        videoPlayer.play();
      }}
    }}

    function triggerRunAnalysis() {{
      const path = document.getElementById('videoPathInput').value.trim();
      if (!path) {{
        alert("Vui lòng nhập đường dẫn video local hoặc chọn file!");
        return;
      }}
      alert(`Pipeline Backend Command sẽ được gọi trên local:\\npython scripts/inference/analyze_long_video.py "${{path}}" --output-dir work_dirs/web_run --hit-checkpoint work_dirs/r2plus1d18_hit_full/best.pth --rgb-checkpoint work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth --fusion-checkpoint work_dirs/shuttleset_fusion_epoch20_full_b4/best.pth --event-offsets -4 0 4`);
    }}

    // Auto-load precomputed Match 40 on startup
    window.addEventListener('DOMContentLoaded', () => {{
      loadPrecomputedDataset();
    }});
  </script>
</body>
</html>
"""

open('C:/Users/ADMIN/.gemini/antigravity/brain/2613cee5-90a2-4966-b44b-e952b365d6dc/badminton_analyzer.html', 'w', encoding='utf-8').write(html_content)
print("Saved HTML file to artifact directory successfully.")

