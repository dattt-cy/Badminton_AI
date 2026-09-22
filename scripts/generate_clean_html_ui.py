import json
from pathlib import Path

# Load both match 40 and match 39 verified data
with open("work_dirs/match40_verified_eval.json", "r", encoding="utf-8") as f:
    match40_events = json.load(f)

with open("work_dirs/match39_verified_eval.json", "r", encoding="utf-8") as f:
    match39_events = json.load(f)

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Badminton Sports Analytics - Epoch 20 Fusion</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    body {{
      font-family: 'Inter', sans-serif;
      background-color: #0b0f19;
      color: #f1f5f9;
    }}
    .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
    .tab-active {{
      background-color: #1e293b;
      color: #38bdf8;
      border-color: #38bdf8;
    }}
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #0f172a; }}
    ::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 3px; }}
  </style>
</head>
<body class="min-h-screen flex flex-col antialiased">

  <!-- Header -->
  <header class="bg-[#101726] border-b border-slate-800 px-6 py-3">
    <div class="max-w-7xl mx-auto flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center text-white font-bold text-sm">
          🏸
        </div>
        <div>
          <h1 class="text-base font-bold text-white tracking-tight flex items-center gap-2">
            Badminton AI Classifier
            <span class="text-[11px] font-normal px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">v2.0 Fusion</span>
          </h1>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <div class="flex items-center gap-2 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-xs">
          <span class="w-2 h-2 rounded-full bg-emerald-400" id="serverDot"></span>
          <span class="text-slate-300 text-[11px]" id="serverStatus">Local Server: 5000</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="bg-[#0e1422] border-b border-slate-800">
    <div class="max-w-7xl mx-auto px-6 flex space-x-1 py-1.5">
      <button id="tabSingleBtn" onclick="switchTab('single')" class="tab-active px-4 py-2 rounded-lg text-xs font-semibold border-b-2 transition flex items-center gap-2">
        <i class="fa-solid fa-play text-xs"></i>
        <span>Phân Tích Clip Cú Đánh (1 - 5s)</span>
      </button>
      <button id="tabMatchBtn" onclick="switchTab('match')" class="px-4 py-2 rounded-lg text-xs font-semibold text-slate-400 hover:text-white border-b-2 border-transparent transition flex items-center gap-2">
        <i class="fa-solid fa-list-check text-xs"></i>
        <span>Phân Tích Trận Đấu 5 Phút (BWF Benchmark)</span>
      </button>
    </div>
  </div>

  <!-- Main Container -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-6">

    <!-- ============================================== -->
    <!-- TAB 1: CLIP PHÂN TÍCH ĐƠN LẺ -->
    <!-- ============================================== -->
    <div id="tabSingleContent" class="space-y-5">
      
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        <!-- Cột Trái: Input & Video Player (5 cols) -->
        <div class="lg:col-span-5 space-y-4">
          
          <div class="bg-[#111827] border border-slate-800 rounded-xl p-4 space-y-3.5">
            <h2 class="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <i class="fa-solid fa-sliders text-sky-400"></i> Cấu hình đầu vào
            </h2>

            <!-- Video File Input -->
            <div class="space-y-1.5">
              <div class="flex gap-2">
                <input type="text" id="videoPathInput" placeholder="Dán đường dẫn video hoặc chọn file..." 
                       value="C:/Users/ADMIN/Downloads/archive/forehand_clear/001.mp4"
                       class="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-sky-500">
                <input type="file" id="videoFileInput" accept="video/mp4" class="hidden" onchange="handleFileSelect(event)">
                <button onclick="document.getElementById('videoFileInput').click()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200">
                  <i class="fa-solid fa-folder-open"></i>
                </button>
              </div>
            </div>

            <!-- Parameters Grid -->
            <div class="grid grid-cols-2 gap-2 text-xs">
              <div>
                <label class="block text-slate-400 text-[11px] mb-1">Pipeline AI:</label>
                <select id="pipelineSelect" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 text-xs focus:outline-none">
                  <option value="fusion" selected>Fusion Pipeline (Epoch 20)</option>
                  <option value="rgb_fast">RGB Fast Mode</option>
                </select>
              </div>

              <div>
                <label class="block text-slate-400 text-[11px] mb-1">Vị trí người đánh:</label>
                <select id="playerSideSelect" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 text-xs focus:outline-none">
                  <option value="top" selected>Sân trên (Xa Camera - Archive)</option>
                  <option value="bottom">Sân dưới (Gần Camera)</option>
                  <option value="auto">Tự động (Auto)</option>
                </select>
              </div>
            </div>

            <!-- Quick Preset Clips -->
            <div class="pt-1">
              <span class="text-[11px] text-slate-500 block mb-1.5">Clip thử nghiệm nhanh:</span>
              <div class="flex flex-wrap gap-1.5">
                <button onclick="setClipPreset('C:/Users/ADMIN/Downloads/archive/forehand_clear/001.mp4', 'top')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-300 border border-slate-750">
                  forehand_clear/001
                </button>
                <button onclick="setClipPreset('C:/Users/ADMIN/Downloads/archive/backhand_drive/001.mp4', 'top')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-300 border border-slate-750">
                  backhand_drive/001
                </button>
                <button onclick="setClipPreset('C:/Users/ADMIN/Downloads/archive/forehand_clear/053.mp4', 'top')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-300 border border-slate-750">
                  forehand_clear/053
                </button>
              </div>
            </div>

            <!-- Submit Button -->
            <button id="runBtn" onclick="runAnalysis()" class="w-full py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 font-semibold text-xs text-white transition flex items-center justify-center gap-2">
              <i class="fa-solid fa-bolt"></i>
              <span id="runBtnText">Phân Tích Cú Đánh</span>
            </button>
          </div>

          <!-- Video Player -->
          <div class="bg-[#111827] border border-slate-800 rounded-xl p-3 space-y-2">
            <div class="flex items-center justify-between text-xs px-1">
              <span class="text-slate-400 font-medium">Trình phát video</span>
              <span id="videoNameBadge" class="text-slate-400 font-mono text-[11px] truncate max-w-[200px]">001.mp4</span>
            </div>
            <div class="aspect-video bg-black rounded-lg overflow-hidden border border-slate-800">
              <video id="clipPlayer" controls loop playsinline class="w-full h-full object-contain"></video>
            </div>
          </div>

        </div>

        <!-- Cột Phải: Kết Quả Phân Tích (7 cols) -->
        <div class="lg:col-span-7 space-y-4">
          
          <div class="bg-[#111827] border border-slate-800 rounded-xl p-5 space-y-4">
            
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 class="font-bold text-sm text-white">Kết quả nhận diện</h3>
                <p id="resultSub" class="text-xs text-slate-400 mt-0.5">Pipeline: <span class="text-sky-400 font-mono font-medium">Epoch 20 Fusion (TrackNet + Pose + Physics)</span></p>
              </div>
              <span id="execTime" class="text-[11px] text-slate-500 font-mono"></span>
            </div>

            <!-- 2 Thẻ Kết Quả Chính -->
            <div class="grid grid-cols-2 gap-3">
              
              <div class="bg-slate-900 border border-slate-800 rounded-lg p-3.5">
                <span class="text-[11px] text-slate-400 font-medium block">Kỹ thuật cú đánh (Stroke)</span>
                <div class="text-xl font-bold text-white mt-1" id="resStroke">CLEAR</div>
                <div class="mt-2 flex items-center justify-between text-xs">
                  <span class="text-slate-500">Độ tin cậy:</span>
                  <span class="font-mono font-bold text-emerald-400" id="resStrokeProb">--%</span>
                </div>
              </div>

              <div class="bg-slate-900 border border-slate-800 rounded-lg p-3.5">
                <span class="text-[11px] text-slate-400 font-medium block">Hướng tay đánh (Side)</span>
                <div class="text-xl font-bold text-white mt-1" id="resSide">FOREHAND</div>
                <div class="mt-2 flex items-center justify-between text-xs">
                  <span class="text-slate-500">Độ tin cậy:</span>
                  <span class="font-mono font-bold text-sky-400" id="resSideProb">--%</span>
                </div>
              </div>

            </div>

            <!-- Bảng phân bố xác suất 8 kỹ thuật -->
            <div class="space-y-2 pt-1">
              <span class="text-xs font-semibold text-slate-400 block">Phân bố xác suất 8 kỹ thuật:</span>
              <div id="probBarsContainer" class="space-y-1.5">
                <!-- Dynamic bars -->
              </div>
            </div>

            <!-- Metadata info -->
            <div class="bg-slate-900/60 border border-slate-800 rounded-lg p-3 text-xs text-slate-400 grid grid-cols-4 gap-2 text-center">
              <div>
                <span class="text-slate-500 block text-[10px]">Số frame:</span>
                <span class="font-mono text-slate-200 font-semibold" id="mFrames">--</span>
              </div>
              <div>
                <span class="text-slate-500 block text-[10px]">Tốc độ:</span>
                <span class="font-mono text-slate-200 font-semibold" id="mFps">-- fps</span>
              </div>
              <div>
                <span class="text-slate-500 block text-[10px]">Thời lượng:</span>
                <span class="font-mono text-slate-200 font-semibold" id="mDur">--s</span>
              </div>
              <div>
                <span class="text-slate-500 block text-[10px]">Vị trí nhận diện:</span>
                <span class="text-slate-200 font-semibold capitalize" id="mPlayer">--</span>
              </div>
            </div>

          </div>

        </div>

      </div>

    </div>

    <!-- ============================================== -->
    <!-- TAB 2: TRẬN ĐẤU 5 PHÚT (BWF BENCHMARK) -->
    <!-- ============================================== -->
    <div id="tabMatchContent" class="hidden space-y-4">
      
      <!-- Top Bar: Match Selector & Key Stats -->
      <div class="bg-[#111827] border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div class="flex items-center gap-3">
          <label class="text-xs font-semibold text-slate-400">Trận đấu:</label>
          <select id="matchSelect" onchange="switchMatchDataset()" class="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none">
            <option value="m39" selected>Match 39 (82.0% Acc) &bull; Anthony Ginting vs Lee Zii Jia (58 GT)</option>
            <option value="m40">Match 40 (80.9% Acc) &bull; BWF Finals (68 GT)</option>
          </select>
        </div>

        <div class="flex items-center gap-6 text-xs">
          <div>
            <span class="text-slate-500 block text-[10px]">Cú đánh đúng:</span>
            <span class="font-mono font-bold text-emerald-400 text-sm" id="mStatStroke">82.0%</span>
          </div>
          <div>
            <span class="text-slate-500 block text-[10px]">Hướng tay đúng:</span>
            <span class="font-mono font-bold text-sky-400 text-sm" id="mStatSide">82.0%</span>
          </div>
          <div>
            <span class="text-slate-500 block text-[10px]">Đúng cả hai:</span>
            <span class="font-mono font-bold text-purple-400 text-sm" id="mStatJoint">67.2%</span>
          </div>
        </div>
      </div>

      <!-- Video & Table -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-5">
        
        <!-- Video Player (5 cols) -->
        <div class="lg:col-span-5 bg-[#111827] border border-slate-800 rounded-xl p-3 space-y-2">
          <div class="flex items-center justify-between text-xs px-1">
            <span class="text-slate-300 font-medium">Video trận đấu</span>
            <span id="matchVideoTimer" class="font-mono text-slate-400 text-[11px]">00:00 / 05:00</span>
          </div>
          <div class="aspect-video bg-black rounded-lg overflow-hidden border border-slate-800">
            <video id="matchVideoPlayer" controls class="w-full h-full object-contain">
              <source id="matchVideoSource" src="/api/stream_video?path=work_dirs/test_match39_10m_15m.mp4" type="video/mp4">
            </video>
          </div>
        </div>

        <!-- Table (7 cols) -->
        <div class="lg:col-span-7 bg-[#111827] border border-slate-800 rounded-xl p-3.5 space-y-3">
          
          <div class="flex items-center justify-between gap-3 text-xs">
            <div class="flex items-center gap-2">
              <span class="text-slate-400 text-[11px]">Lọc kỹ thuật:</span>
              <select id="mFilterStroke" onchange="renderMatchEventsTable()" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none">
                <option value="ALL">Tất cả</option>
                <option value="clear">Clear</option>
                <option value="smash">Smash</option>
                <option value="drop">Drop</option>
                <option value="lift">Lift</option>
                <option value="net_shot">Net Shot</option>
                <option value="net_attack">Net Attack</option>
                <option value="drive">Drive</option>
                <option value="serve">Serve</option>
              </select>
            </div>

            <div class="flex items-center gap-2">
              <span class="text-slate-400 text-[11px]">Đánh giá:</span>
              <select id="mFilterStatus" onchange="renderMatchEventsTable()" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none">
                <option value="ALL">Tất cả</option>
                <option value="CORRECT">Chính xác (✓)</option>
                <option value="WRONG">Lệch nhãn (✗)</option>
              </select>
            </div>
          </div>

          <div class="max-h-[400px] overflow-y-auto rounded-lg border border-slate-800">
            <table class="w-full text-left text-xs border-collapse font-mono">
              <thead class="bg-slate-900 text-slate-400 sticky top-0 border-b border-slate-800">
                <tr>
                  <th class="py-2 px-3">Thời gian</th>
                  <th class="py-2 px-3">Sân</th>
                  <th class="py-2 px-3">Dự đoán AI</th>
                  <th class="py-2 px-3">Chuẩn BWF</th>
                  <th class="py-2 px-3 text-center">Kết quả</th>
                </tr>
              </thead>
              <tbody id="matchEventsTbody" class="divide-y divide-slate-800/60">
                <!-- Populated by JS -->
              </tbody>
            </table>
          </div>

        </div>

      </div>

    </div>

  </main>

  <script>
    const MATCH39_DATA = {json.dumps(match39_events, ensure_ascii=False)};
    const MATCH40_DATA = {json.dumps(match40_events, ensure_ascii=False)};
    let currentMatchData = MATCH39_DATA;

    let isServer = false;

    async function initServerStatus() {{
      try {{
        const res = await fetch("http://127.0.0.1:5000/api/status");
        if (res.ok) {{
          isServer = true;
          document.getElementById("serverDot").className = "w-2 h-2 rounded-full bg-emerald-400";
          document.getElementById("serverStatus").textContent = "Server Online (5000)";
          return;
        }}
      }} catch (e) {{}}
      document.getElementById("serverDot").className = "w-2 h-2 rounded-full bg-slate-500";
      document.getElementById("serverStatus").textContent = "Offline";
    }}

    function switchTab(t) {{
      const b1 = document.getElementById("tabSingleBtn");
      const b2 = document.getElementById("tabMatchBtn");
      const c1 = document.getElementById("tabSingleContent");
      const c2 = document.getElementById("tabMatchContent");
      if (t === "single") {{
        b1.className = "tab-active px-4 py-2 rounded-lg text-xs font-semibold border-b-2 transition flex items-center gap-2";
        b2.className = "px-4 py-2 rounded-lg text-xs font-semibold text-slate-400 hover:text-white border-b-2 border-transparent transition flex items-center gap-2";
        c1.classList.remove("hidden");
        c2.classList.add("hidden");
      }} else {{
        b2.className = "tab-active px-4 py-2 rounded-lg text-xs font-semibold border-b-2 transition flex items-center gap-2";
        b1.className = "px-4 py-2 rounded-lg text-xs font-semibold text-slate-400 hover:text-white border-b-2 border-transparent transition flex items-center gap-2";
        c2.classList.remove("hidden");
        c1.classList.add("hidden");
        renderMatchEventsTable();
      }}
    }}

    function setClipPreset(path, side) {{
      document.getElementById("videoPathInput").value = path;
      document.getElementById("playerSideSelect").value = side;
      document.getElementById("videoNameBadge").textContent = path.split("/").pop();
      const p = document.getElementById("clipPlayer");
      if (isServer) {{
        p.src = "http://127.0.0.1:5000/api/stream_video?path=" + path;
      }}
    }}

    function handleFileSelect(e) {{
      const file = e.target.files[0];
      if (!file) return;
      document.getElementById("videoPathInput").value = file.name;
      document.getElementById("videoNameBadge").textContent = file.name;
      const p = document.getElementById("clipPlayer");
      p.src = URL.createObjectURL(file);
      p.play();
    }}

    async function runAnalysis() {{
      const path = document.getElementById("videoPathInput").value.trim();
      const fileInput = document.getElementById("videoFileInput");
      const pipe = document.getElementById("pipelineSelect").value;
      const side = document.getElementById("playerSideSelect").value;
      const btn = document.getElementById("runBtn");
      const btnText = document.getElementById("runBtnText");

      btn.disabled = true;
      btnText.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-1"></i> Đang phân tích...';
      const t0 = performance.now();

      try {{
        let res;
        if (fileInput.files.length > 0) {{
          const fd = new FormData();
          fd.append("file", fileInput.files[0]);
          fd.append("pipeline_mode", pipe);
          fd.append("player_side", side);
          res = await fetch("http://127.0.0.1:5000/api/analyze_clip", {{ method: "POST", body: fd }});
        }} else if (path) {{
          res = await fetch("http://127.0.0.1:5000/api/analyze_clip", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ video_path: path, pipeline_mode: pipe, player_side: side }})
          }});
        }}

        if (res && res.ok) {{
          const data = await res.json();
          renderSingleResult(data, ((performance.now() - t0)/1000).toFixed(1));
        }}
      }} catch (err) {{
        console.error("Inference error:", err);
      }} finally {{
        btn.disabled = false;
        btnText.innerHTML = 'Phân Tích Cú Đánh';
      }}
    }}

    function renderSingleResult(d, timeSec) {{
      if (!d) return;
      document.getElementById("execTime").textContent = timeSec ? `${{timeSec}}s` : "";
      
      const s = d.prediction?.stroke || {{}};
      const side = d.prediction?.stroke_side || {{}};
      const sLabel = (s.label || "clear").toUpperCase();
      const sVn = s.vn_label || sLabel;
      const sProb = (s.probability ? s.probability * 100 : 0).toFixed(1);

      document.getElementById("resStroke").textContent = `${{sLabel}} (${{sVn.split(' ')[0]}})`;
      document.getElementById("resStrokeProb").textContent = `${{sProb}}%`;

      const sideLabel = (side.label || "forehand").toUpperCase();
      const sideVn = side.vn_label || sideLabel;
      const sideProb = (side.probability ? side.probability * 100 : 0).toFixed(1);

      document.getElementById("resSide").textContent = `${{sideLabel}} (${{sideVn.split(' ')[0]}})`;
      document.getElementById("resSideProb").textContent = `${{sideProb}}%`;

      // Bars
      const bc = document.getElementById("probBarsContainer");
      bc.innerHTML = "";
      const ranking = d.stroke_ranking || [];
      ranking.forEach((item, idx) => {{
        const pct = (item.probability * 100).toFixed(1);
        const isTop = idx === 0;
        const barColor = isTop ? "bg-emerald-500" : (idx === 1 ? "bg-sky-500" : "bg-slate-700");
        const row = document.createElement("div");
        row.innerHTML = `
          <div class="flex justify-between text-[11px] mb-0.5">
            <span class="${{isTop ? 'font-bold text-white' : 'text-slate-400'}} capitalize">${{item.vn_label || item.label}}</span>
            <span class="font-mono ${{isTop ? 'font-bold text-emerald-400' : 'text-slate-400'}}">${{pct}}%</span>
          </div>
          <div class="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
            <div class="${{barColor}} h-1.5 rounded-full" style="width: ${{pct}}%"></div>
          </div>
        `;
        bc.appendChild(row);
      }});

      // Meta
      const meta = d.video_metadata || {{}};
      document.getElementById("mFrames").textContent = meta.frames || "--";
      document.getElementById("mFps").textContent = meta.fps ? meta.fps.toFixed(1) : "--";
      document.getElementById("mDur").textContent = meta.duration_seconds ? `${{meta.duration_seconds}}s` : "--";
      document.getElementById("mPlayer").textContent = meta.player_side || "top";
    }}

    function switchMatchDataset() {{
      const sel = document.getElementById("matchSelect").value;
      const v = document.getElementById("matchVideoPlayer");
      if (sel === "m39") {{
        currentMatchData = MATCH39_DATA;
        v.src = "/api/stream_video?path=work_dirs/test_match39_10m_15m.mp4";
        document.getElementById("mStatStroke").textContent = "82.0%";
        document.getElementById("mStatSide").textContent = "82.0%";
        document.getElementById("mStatJoint").textContent = "67.2%";
      }} else {{
        currentMatchData = MATCH40_DATA;
        v.src = "/api/stream_video?path=work_dirs/test_match40_10m_15m.mp4";
        document.getElementById("mStatStroke").textContent = "80.9%";
        document.getElementById("mStatSide").textContent = "83.8%";
        document.getElementById("mStatJoint").textContent = "70.6%";
      }}
      renderMatchEventsTable();
    }}

    function renderMatchEventsTable() {{
      const tbody = document.getElementById("matchEventsTbody");
      tbody.innerHTML = "";
      const sFilter = document.getElementById("mFilterStroke").value;
      const stFilter = document.getElementById("mFilterStatus").value;

      currentMatchData.forEach(ev => {{
        if (sFilter !== "ALL" && ev.pred_stroke !== sFilter && ev.gt_stroke !== sFilter) return;
        if (stFilter === "CORRECT" && !ev.ok_stroke) return;
        if (stFilter === "WRONG" && ev.ok_stroke) return;

        const tr = document.createElement("tr");
        tr.className = "hover:bg-slate-800/60 cursor-pointer transition";
        tr.onclick = () => {{
          const vp = document.getElementById("matchVideoPlayer");
          vp.currentTime = Math.max(0, ev.time - 0.8);
          vp.play();
        }};

        const mins = Math.floor(ev.time / 60);
        const secs = Math.floor(ev.time % 60);
        const tStr = `${{String(mins).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}}`;
        const badge = ev.ok_stroke
          ? '<span class="text-emerald-400 font-bold">✓</span>'
          : '<span class="text-rose-400 font-bold">✗</span>';

        tr.innerHTML = `
          <td class="py-1.5 px-3 text-sky-400">${{tStr}}</td>
          <td class="py-1.5 px-3 capitalize text-slate-400">${{ev.player}}</td>
          <td class="py-1.5 px-3 text-white font-medium">${{ev.pred_stroke}} <span class="text-slate-500 text-[10px]">(${{ev.pred_side}})</span></td>
          <td class="py-1.5 px-3 text-slate-400">${{ev.gt_stroke || '--'}} <span class="text-slate-500 text-[10px]">(${{ev.gt_side || '--'}})</span></td>
          <td class="py-1.5 px-3 text-center">${{badge}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    const mVid = document.getElementById("matchVideoPlayer");
    if (mVid) {{
      mVid.addEventListener("timeupdate", () => {{
        const c = mVid.currentTime;
        const mins = Math.floor(c / 60);
        const secs = Math.floor(c % 60);
        document.getElementById("matchVideoTimer").textContent = `${{String(mins).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}} / 05:00`;
      }});
    }}

    window.addEventListener("DOMContentLoaded", () => {{
      initServerStatus();
      setClipPreset("C:/Users/ADMIN/Downloads/archive/forehand_clear/001.mp4", "top");
    }});
  </script>
</body>
</html>
"""

# Write to workspace
with open("badminton_analyzer.html", "w", encoding="utf-8") as f:
    f.write(html_content)

# Write to artifact
artifact_path = Path("C:/Users/ADMIN/.gemini/antigravity/brain/2613cee5-90a2-4966-b44b-e952b365d6dc/badminton_analyzer.html")
if artifact_path.parent.exists():
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(html_content)

print("Generated clean, sports-grade badminton_analyzer.html successfully!")

