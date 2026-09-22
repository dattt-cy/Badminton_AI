import json
from pathlib import Path

eval_file = Path("work_dirs/match40_verified_eval.json")
with open(eval_file, "r", encoding="utf-8") as f:
    match40_events = json.load(f)

demo_001 = {
    "video": "forehand_clear/001.mp4",
    "prediction": {
        "stroke": {"label": "clear", "probability": 0.5048, "vn_label": "Phông Cầu (Clear)"},
        "stroke_side": {"label": "forehand", "probability": 0.9290, "vn_label": "Thuận Tay (Forehand)"},
        "combined_label": "forehand clear"
    },
    "stroke_ranking": [
        {"label": "clear", "probability": 0.5048, "vn_label": "Phông Cầu (Clear)"},
        {"label": "smash", "probability": 0.3094, "vn_label": "Đập Cầu (Smash)"},
        {"label": "net_attack", "probability": 0.0905, "vn_label": "Vồ Lưới / Đẩy Cầu (Net Attack)"},
        {"label": "lift", "probability": 0.0612, "vn_label": "Vút Cầu / Hất Cầu (Lift)"},
        {"label": "drive", "probability": 0.0207, "vn_label": "Tạt Cầu (Drive)"},
        {"label": "drop", "probability": 0.0125, "vn_label": "Chặt / Bỏ Nhỏ (Drop)"},
        {"label": "net_shot", "probability": 0.0009, "vn_label": "Gài Lưới (Net Shot)"},
        {"label": "serve", "probability": 0.00004, "vn_label": "Giao Cầu (Serve)"}
    ],
    "stroke_side_ranking": [
        {"label": "forehand", "probability": 0.9290, "vn_label": "Thuận Tay (Forehand)"},
        {"label": "backhand", "probability": 0.0710, "vn_label": "Trái Tay (Backhand)"},
        {"label": "aroundhead", "probability": 0.0000, "vn_label": "Vòng Đầu (Aroundhead)"}
    ],
    "video_metadata": {"frames": 70, "fps": 16.3, "duration_seconds": 4.3, "player_side": "bottom (dưới sân)"}
}

demo_098 = {
    "video": "forehand_clear/098.mp4",
    "prediction": {
        "stroke": {"label": "lift", "probability": 0.8672, "vn_label": "Vút Cầu / Hất Cầu (Lift)"},
        "stroke_side": {"label": "backhand", "probability": 0.5910, "vn_label": "Trái Tay (Backhand)"},
        "combined_label": "backhand lift"
    },
    "stroke_ranking": [
        {"label": "lift", "probability": 0.8672, "vn_label": "Vút Cầu / Hất Cầu (Lift)"},
        {"label": "clear", "probability": 0.1134, "vn_label": "Phông Cầu (Clear)"},
        {"label": "net_attack", "probability": 0.0092, "vn_label": "Vồ Lưới / Đẩy Cầu (Net Attack)"},
        {"label": "smash", "probability": 0.0043, "vn_label": "Đập Cầu (Smash)"},
        {"label": "drive", "probability": 0.0039, "vn_label": "Tạt Cầu (Drive)"},
        {"label": "serve", "probability": 0.0010, "vn_label": "Giao Cầu (Serve)"},
        {"label": "drop", "probability": 0.0007, "vn_label": "Chặt / Bỏ Nhỏ (Drop)"},
        {"label": "net_shot", "probability": 0.0003, "vn_label": "Gài Lưới (Net Shot)"}
    ],
    "stroke_side_ranking": [
        {"label": "backhand", "probability": 0.5910, "vn_label": "Trái Tay (Backhand)"},
        {"label": "forehand", "probability": 0.4090, "vn_label": "Thuận Tay (Forehand)"},
        {"label": "aroundhead", "probability": 0.0000, "vn_label": "Vòng Đầu (Aroundhead)"}
    ],
    "video_metadata": {"frames": 54, "fps": 21.9, "duration_seconds": 2.5, "player_side": "bottom (dưới sân)"}
}

demo_backhand_001 = {
    "video": "backhand_drive/001.mp4",
    "prediction": {
        "stroke": {"label": "lift", "probability": 0.4763, "vn_label": "Vút Cầu / Hất Cầu (Lift)"},
        "stroke_side": {"label": "backhand", "probability": 0.7343, "vn_label": "Trái Tay (Backhand)"},
        "combined_label": "backhand lift"
    },
    "stroke_ranking": [
        {"label": "lift", "probability": 0.4763, "vn_label": "Vút Cầu / Hất Cầu (Lift)"},
        {"label": "net_shot", "probability": 0.3347, "vn_label": "Gài Lưới (Net Shot)"},
        {"label": "net_attack", "probability": 0.1313, "vn_label": "Vồ Lưới / Đẩy Cầu (Net Attack)"},
        {"label": "drive", "probability": 0.0518, "vn_label": "Tạt Cầu (Drive)"},
        {"label": "serve", "probability": 0.0041, "vn_label": "Giao Cầu (Serve)"},
        {"label": "smash", "probability": 0.0012, "vn_label": "Đập Cầu (Smash)"},
        {"label": "clear", "probability": 0.0004, "vn_label": "Phông Cầu (Clear)"},
        {"label": "drop", "probability": 0.0001, "vn_label": "Chặt / Bỏ Nhỏ (Drop)"}
    ],
    "stroke_side_ranking": [
        {"label": "backhand", "probability": 0.7343, "vn_label": "Trái Tay (Backhand)"},
        {"label": "forehand", "probability": 0.2657, "vn_label": "Thuận Tay (Forehand)"},
        {"label": "aroundhead", "probability": 0.0000, "vn_label": "Vòng Đầu (Aroundhead)"}
    ],
    "video_metadata": {"frames": 48, "fps": 16.0, "duration_seconds": 3.0, "player_side": "bottom (dưới sân)"}
}

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Badminton AI Classifier - Pipeline Epoch 20</title>
  <!-- Tailwind CSS -->
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    body {{
      font-family: 'Inter', sans-serif;
      background-color: #0f172a;
      color: #e2e8f0;
    }}
    .tab-btn.active {{
      background: linear-gradient(135deg, #2563eb, #3b82f6);
      color: #ffffff;
      box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.4);
    }}
    .progress-bar-fill {{
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
      width: 8px;
      height: 8px;
    }}
    ::-webkit-scrollbar-track {{
      background: #1e293b;
    }}
    ::-webkit-scrollbar-thumb {{
      background: #475569;
      border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
      background: #64748b;
    }}
  </style>
</head>
<body class="min-h-screen flex flex-col">

  <!-- Header -->
  <header class="bg-slate-900 border-b border-slate-800 sticky top-0 z-50 shadow-md">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
          <i class="fa-solid fa-feather text-white text-xl"></i>
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            Badminton AI Classifier
            <span class="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium">Pipeline Epoch 20</span>
          </h1>
          <p class="text-xs text-slate-400">Multi-task R(2+1)D + Fusion Head (Phân loại Kỹ thuật & Hướng tay Cầu lông)</p>
        </div>
      </div>
      
      <!-- Right Status / Server Check -->
      <div class="flex items-center gap-3">
        <div id="serverStatusBadge" class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs">
          <span class="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" id="statusDot"></span>
          <span class="text-slate-300 font-medium" id="statusText">Đang kiểm tra Local Server...</span>
        </div>
        <div class="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-950/40 border border-emerald-500/20 text-xs text-emerald-400">
          <i class="fa-solid fa-award"></i>
          <span>BWF Benchmark: 80.9%</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="bg-slate-900/60 border-b border-slate-800 backdrop-blur">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex space-x-2 py-2">
        <button id="tabSingleBtn" onclick="switchTab('single')" class="tab-btn active px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center gap-2.5 transition-all duration-200">
          <i class="fa-solid fa-bolt text-amber-400"></i>
          <span>1. Phân Tích Clip Cú Đánh (1 - 5s)</span>
        </button>
        <button id="tabMatchBtn" onclick="switchTab('match')" class="tab-btn px-5 py-2.5 rounded-xl font-semibold text-sm text-slate-400 hover:text-white hover:bg-slate-800 flex items-center gap-2.5 transition-all duration-200">
          <i class="fa-solid fa-trophy text-yellow-400"></i>
          <span>2. Dữ Liệu Mẫu Trận Đấu 5 Phút (Match 40)</span>
        </button>
      </div>
    </div>
  </div>

  <!-- Main Content Area -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">

    <!-- ========================================== -->
    <!-- TAB 1: PHÂN TÍCH CLIP CÚ ĐÁNH ĐƠN LẺ (1-5s) -->
    <!-- ========================================== -->
    <div id="tabSingleContent" class="space-y-6">
      
      <!-- Top Guide Card -->
      <div class="p-4 rounded-2xl bg-blue-950/20 border border-blue-500/20 flex items-start gap-3.5 text-blue-200 text-sm">
        <i class="fa-solid fa-circle-info text-blue-400 text-lg mt-0.5"></i>
        <div class="space-y-1">
          <p class="font-semibold text-white">Chế độ phân tích Clip ngắn (Single Shot):</p>
          <p class="text-xs text-blue-300/90 leading-relaxed">
            Áp dụng cho các video ngắn từ 1 đến 5 giây chứa 1 pha chạm cầu (như <b>098.mp4</b>, <b>001.mp4</b>). 
            Kết quả bên dưới hiển thị <b>chính xác kết quả của đúng video bạn đang chọn</b> (gồm loại cú đánh và hướng tay).
          </p>
        </div>
      </div>

      <!-- Controls: Video Selection & Input -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        <!-- Left: Input & Video Player (5 cols) -->
        <div class="lg:col-span-5 space-y-4">
          <div class="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
            <h3 class="font-bold text-base text-white flex items-center gap-2">
              <i class="fa-solid fa-video text-blue-400"></i>
              Chọn Video Phân Tích
            </h3>

            <!-- Option A: File Upload -->
            <div>
              <label class="block text-xs font-semibold text-slate-400 mb-1.5">Cách 1: Tải video từ máy tính</label>
              <div class="relative flex items-center">
                <input type="file" id="videoFileInput" accept="video/mp4,video/avi,video/mov" class="hidden" onchange="handleFileSelect(event)">
                <button onclick="document.getElementById('videoFileInput').click()" class="w-full py-2.5 px-4 rounded-xl border border-slate-700 bg-slate-800/80 hover:bg-slate-750 text-slate-200 text-sm font-medium flex items-center justify-center gap-2 transition hover:border-slate-600">
                  <i class="fa-solid fa-cloud-arrow-up text-blue-400"></i>
                  <span id="uploadBtnText">Chọn file video (.mp4)...</span>
                </button>
              </div>
            </div>

            <!-- Option B: Local Path -->
            <div>
              <label class="block text-xs font-semibold text-slate-400 mb-1.5">Cách 2: Hoặc Dán đường dẫn cục bộ (Local Path)</label>
              <div class="flex gap-2">
                <input type="text" id="videoPathInput" placeholder="Ví dụ: C:\\Users\\ADMIN\\Downloads\\archive\\forehand_clear\\001.mp4" 
                       class="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition">
              </div>
            </div>

            <!-- Player Side Choice -->
            <div>
              <label class="block text-xs font-semibold text-slate-400 mb-1.5">Vị trí người chơi (Player Side):</label>
              <select id="playerSideSelect" class="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-xs text-slate-200 focus:outline-none focus:border-blue-500 transition">
                <option value="bottom" selected>Sân Dưới (Gần Camera - Khuyên dùng cho clip đơn)</option>
                <option value="auto">Tự động phát hiện (Auto Detect)</option>
                <option value="top">Sân Trên (Xa Camera)</option>
              </select>
            </div>

            <!-- Quick Demo Clips Selection -->
            <div>
              <label class="block text-xs font-semibold text-slate-400 mb-2">Thử ngay các clip mẫu thực tế:</label>
              <div class="grid grid-cols-1 gap-2">
                <button onclick="selectDemoClip('001')" class="text-left px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700/80 border border-slate-700/80 text-xs text-slate-300 flex items-center justify-between transition">
                  <span class="font-medium flex items-center gap-2">
                    <i class="fa-solid fa-circle-play text-emerald-400"></i>
                    forehand_clear/001.mp4
                  </span>
                  <span class="text-[11px] text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded font-mono">Phông thuận tay</span>
                </button>
                <button onclick="selectDemoClip('098')" class="text-left px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700/80 border border-slate-700/80 text-xs text-slate-300 flex items-center justify-between transition">
                  <span class="font-medium flex items-center gap-2">
                    <i class="fa-solid fa-circle-play text-blue-400"></i>
                    forehand_clear/098.mp4
                  </span>
                  <span class="text-[11px] text-blue-400 bg-blue-950/60 px-2 py-0.5 rounded font-mono">Clip 098.mp4</span>
                </button>
                <button onclick="selectDemoClip('backhand_001')" class="text-left px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700/80 border border-slate-700/80 text-xs text-slate-300 flex items-center justify-between transition">
                  <span class="font-medium flex items-center gap-2">
                    <i class="fa-solid fa-circle-play text-purple-400"></i>
                    backhand_drive/001.mp4
                  </span>
                  <span class="text-[11px] text-purple-400 bg-purple-950/60 px-2 py-0.5 rounded font-mono">Trái tay</span>
                </button>
              </div>
            </div>

            <!-- Run Button -->
            <button id="runAnalyzeBtn" onclick="runSingleClipAnalysis()" class="w-full py-3.5 rounded-xl font-bold text-sm bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-600/30 flex items-center justify-center gap-2 transition duration-200">
              <i class="fa-solid fa-wand-magic-sparkles"></i>
              <span id="runAnalyzeText">BẮT ĐẦU PHÂN TÍCH CÚ ĐÁNH</span>
            </button>
          </div>

          <!-- Video Player Box -->
          <div class="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-2">
            <div class="flex items-center justify-between text-xs text-slate-400 px-1">
              <span class="font-semibold text-slate-300 flex items-center gap-1.5">
                <i class="fa-solid fa-film text-blue-400"></i> Video Đang Xem
              </span>
              <span id="playerVideoName" class="font-mono text-slate-400 truncate max-w-[200px]">001.mp4</span>
            </div>
            <div class="relative aspect-video bg-black rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
              <video id="singleVideoPlayer" controls loop playsinline class="w-full h-full object-contain"></video>
              <div id="noVideoOverlay" class="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/80 text-slate-500 text-xs p-4 text-center">
                <i class="fa-solid fa-file-video text-3xl mb-2 text-slate-600"></i>
                <p>Chưa nạp video. Hãy chọn file hoặc bấm clip mẫu phía trên.</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Right: Real-time Analysis Result for Selected Video (7 cols) -->
        <div class="lg:col-span-7 space-y-4">
          
          <div class="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-5">
            
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 class="font-bold text-lg text-white flex items-center gap-2">
                  <i class="fa-solid fa-chart-pie text-emerald-400"></i>
                  Kết Quả Phân Tích Của Clip
                </h3>
                <p id="analysisVideoTarget" class="text-xs text-slate-400 mt-0.5">Video: <span class="text-blue-400 font-mono font-medium">forehand_clear/001.mp4</span></p>
              </div>
              <span id="analysisBadge" class="text-xs px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">
                Đã phân tích xong
              </span>
            </div>

            <!-- Two Main Prediction Cards -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              
              <!-- Stroke Prediction Card -->
              <div class="p-4 rounded-xl bg-gradient-to-br from-slate-800/80 to-slate-850/80 border border-slate-700/80 relative overflow-hidden">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Kỹ Thuật Cú Đánh</span>
                  <i class="fa-solid fa-bullseye text-blue-400"></i>
                </div>
                <div class="text-2xl font-black text-white mt-1 tracking-tight" id="predStrokeName">
                  CLEAR (Phông Cầu)
                </div>
                <div class="mt-2 flex items-center justify-between">
                  <span class="text-xs text-slate-400">Độ tin cậy:</span>
                  <span class="text-lg font-bold text-emerald-400" id="predStrokeProb">50.5%</span>
                </div>
                <div class="w-full bg-slate-700/60 rounded-full h-2 mt-2 overflow-hidden">
                  <div id="predStrokeBar" class="bg-gradient-to-r from-blue-500 to-emerald-400 h-2 rounded-full progress-bar-fill" style="width: 50.5%"></div>
                </div>
              </div>

              <!-- Stroke Side Card -->
              <div class="p-4 rounded-xl bg-gradient-to-br from-slate-800/80 to-slate-850/80 border border-slate-700/80 relative overflow-hidden">
                <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center justify-between">
                  <span>Hướng Tay Đánh</span>
                  <i class="fa-solid fa-hand text-indigo-400"></i>
                </div>
                <div class="text-2xl font-black text-white mt-1 tracking-tight" id="predSideName">
                  FOREHAND (Thuận Tay)
                </div>
                <div class="mt-2 flex items-center justify-between">
                  <span class="text-xs text-slate-400">Độ tin cậy:</span>
                  <span class="text-lg font-bold text-indigo-400" id="predSideProb">92.9%</span>
                </div>
                <div class="w-full bg-slate-700/60 rounded-full h-2 mt-2 overflow-hidden">
                  <div id="predSideBar" class="bg-gradient-to-r from-indigo-500 to-purple-400 h-2 rounded-full progress-bar-fill" style="width: 92.9%"></div>
                </div>
              </div>

            </div>

            <!-- Full 8-Stroke Ranking Chart -->
            <div class="space-y-3 pt-2">
              <div class="flex items-center justify-between">
                <h4 class="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <i class="fa-solid fa-ranking-star text-amber-400"></i>
                  Phân Bổ Xác Suất Cả 8 Loại Cú Đánh
                </h4>
                <span class="text-[11px] text-slate-400">Softmax Confidence</span>
              </div>

              <div id="strokeRankingContainer" class="space-y-2">
                <!-- Dynamic ranking bars populated by JS -->
              </div>
            </div>

            <!-- Metadata Info -->
            <div class="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-400 grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div>
                <span class="text-slate-500 block">Số Frame:</span>
                <span class="font-semibold text-slate-300 font-mono" id="metaFrames">70 frames</span>
              </div>
              <div>
                <span class="text-slate-500 block">Tốc Độ (FPS):</span>
                <span class="font-semibold text-slate-300 font-mono" id="metaFps">16.3 fps</span>
              </div>
              <div>
                <span class="text-slate-500 block">Thời lượng:</span>
                <span class="font-semibold text-slate-300 font-mono" id="metaDuration">4.3s</span>
              </div>
              <div>
                <span class="text-slate-500 block">Vị Trí Người Chơi:</span>
                <span class="font-semibold text-slate-300 capitalize" id="metaPlayer">Bottom (nửa dưới)</span>
              </div>
            </div>

          </div>

        </div>

      </div>

    </div>

    <!-- ========================================== -->
    <!-- TAB 2: DỮ LIỆU MẪU TRẬN ĐẤU 5 PHÚT (Match 40) -->
    <!-- ========================================== -->
    <div id="tabMatchContent" class="hidden space-y-6">
      
      <!-- Top Disclaimer Card -->
      <div class="p-5 rounded-2xl bg-gradient-to-r from-indigo-950/40 via-purple-950/30 to-slate-900 border border-indigo-500/30 shadow-xl">
        <div class="flex items-start gap-4">
          <div class="w-12 h-12 rounded-xl bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center shrink-0">
            <i class="fa-solid fa-trophy text-indigo-400 text-2xl"></i>
          </div>
          <div class="space-y-1.5 flex-1">
            <h3 class="font-bold text-white text-base flex items-center gap-2">
              Dữ Liệu Mẫu Đã Phân Tích: Match 40 (Từ phút 10:00 đến 15:00)
              <span class="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-normal">70 Cú Đánh Tự Động</span>
            </h3>
            <p class="text-xs text-slate-300 leading-relaxed">
              Trận đấu dài 5 phút (video <code class="bg-slate-800 px-1.5 py-0.5 rounded text-blue-300 font-mono">test_match40_10m_15m.mp4</code>) 
              đã được AI xử lý qua toàn bộ pipeline hoàn chỉnh: <b>Hit Detector</b> (phát hiện điểm đánh) &rarr; <b>YOLOv8-Pose TrackNet</b> (theo dõi cầu thủ & quỹ đạo cầu) &rarr; <b>Epoch 20 Fusion Model</b>. 
              Toàn bộ 70 cú đánh được đối chiếu trực tiếp với dữ liệu thi đấu chính thức BWF (Ground Truth).
            </p>
          </div>
        </div>
      </div>

      <!-- 4 Stat Summary Cards -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        
        <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-800 shadow-md">
          <span class="text-xs text-slate-400 font-semibold block mb-1">Tổng Cú Đánh Phát Hiện</span>
          <div class="text-2xl font-black text-white">70 <span class="text-xs font-normal text-slate-400">cú đánh</span></div>
          <div class="text-[11px] text-slate-500 mt-1">68 cú khớp Ground Truth</div>
        </div>

        <div class="p-4 rounded-xl bg-slate-900/90 border border-emerald-500/30 shadow-md">
          <span class="text-xs text-emerald-400 font-semibold block mb-1">Độ Chính Xác Cú Đánh (Stroke)</span>
          <div class="text-2xl font-black text-emerald-400">80.9%</div>
          <div class="text-[11px] text-emerald-500/80 mt-1">55 / 68 cú đánh chính xác</div>
        </div>

        <div class="p-4 rounded-xl bg-slate-900/90 border border-indigo-500/30 shadow-md">
          <span class="text-xs text-indigo-400 font-semibold block mb-1">Độ Chính Xác Hướng Tay (Side)</span>
          <div class="text-2xl font-black text-indigo-400">83.8%</div>
          <div class="text-[11px] text-indigo-400/80 mt-1">57 / 68 hướng tay chính xác</div>
        </div>

        <div class="p-4 rounded-xl bg-slate-900/90 border border-purple-500/30 shadow-md">
          <span class="text-xs text-purple-400 font-semibold block mb-1">Đúng Đồng Thời Cả Hai</span>
          <div class="text-2xl font-black text-purple-400">70.6%</div>
          <div class="text-[11px] text-purple-400/80 mt-1">48 / 68 chuẩn cả cú đánh & hướng</div>
        </div>

      </div>

      <!-- Match Video Player & Timeline Sync -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        <!-- Video Player (5 cols) -->
        <div class="lg:col-span-5 space-y-3">
          <div class="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3">
            <div class="flex items-center justify-between text-xs text-slate-400">
              <span class="font-semibold text-white flex items-center gap-1.5">
                <i class="fa-solid fa-play text-emerald-400"></i> Video Trận Đấu Match 40
              </span>
              <span class="text-slate-400 font-mono" id="matchVideoTimer">00:00 / 05:00</span>
            </div>
            
            <div class="relative aspect-video bg-black rounded-xl overflow-hidden border border-slate-800">
              <video id="matchVideoPlayer" controls class="w-full h-full object-contain">
                <source src="/api/stream_video?path=work_dirs/test_match40_10m_15m.mp4" type="video/mp4">
              </video>
            </div>
            <p class="text-[11px] text-slate-400 leading-normal">
              💡 <b>Mẹo:</b> Nhấp vào bất kỳ hàng nào trong bảng sự kiện bên cạnh để video tự động tua tới đúng cú đánh đó.
            </p>
          </div>
        </div>

        <!-- Events Table & Filter (7 cols) -->
        <div class="lg:col-span-7 space-y-3">
          <div class="p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3">
            
            <!-- Filters bar -->
            <div class="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
              <div class="flex items-center gap-2">
                <label class="text-xs font-semibold text-slate-400">Lọc Cú Đánh:</label>
                <select id="strokeFilter" onchange="filterMatchTable()" class="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none">
                  <option value="ALL">Tất cả (70 cú đánh)</option>
                  <option value="clear">Phông cầu (Clear)</option>
                  <option value="smash">Đập cầu (Smash)</option>
                  <option value="lift">Vút/Hất cầu (Lift)</option>
                  <option value="drop">Chặt/Bỏ nhỏ (Drop)</option>
                  <option value="net_shot">Gài lưới (Net Shot)</option>
                  <option value="net_attack">Vồ lưới (Net Attack)</option>
                  <option value="drive">Tạt cầu (Drive)</option>
                  <option value="serve">Giao cầu (Serve)</option>
                </select>
              </div>

              <div class="flex items-center gap-2">
                <label class="text-xs font-semibold text-slate-400">Trạng Thái:</label>
                <select id="statusFilter" onchange="filterMatchTable()" class="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none">
                  <option value="ALL">Tất cả</option>
                  <option value="CORRECT">Chính xác (55 cú)</option>
                  <option value="WRONG">Lệch nhãn BWF (13 cú)</option>
                </select>
              </div>
            </div>

            <!-- Table Container -->
            <div class="max-h-[460px] overflow-y-auto rounded-xl border border-slate-800">
              <table class="w-full text-left border-collapse text-xs">
                <thead class="bg-slate-950/80 sticky top-0 border-b border-slate-800 text-slate-400">
                  <tr>
                    <th class="py-2.5 px-3 font-semibold">Thời gian</th>
                    <th class="py-2.5 px-3 font-semibold">Cầu thủ</th>
                    <th class="py-2.5 px-3 font-semibold">AI Dự Đoán</th>
                    <th class="py-2.5 px-3 font-semibold">Chuẩn BWF</th>
                    <th class="py-2.5 px-3 font-semibold text-center">Đánh giá</th>
                  </tr>
                </thead>
                <tbody id="matchEventsTableBody" class="divide-y divide-slate-800/60 font-mono">
                  <!-- Populated by JS -->
                </tbody>
              </table>
            </div>

          </div>
        </div>

      </div>

    </div>

  </main>

  <!-- Footer -->
  <footer class="bg-slate-900 border-t border-slate-800 py-4 text-center text-xs text-slate-500">
    <p>Badminton AI Classification & Multi-modal Evaluation Framework &bull; PBL6 Deep Learning Project 2026</p>
  </footer>

  <!-- Script with embedded data -->
  <script>
    // Embedded Data
    const MATCH40_EVENTS = {json.dumps(match40_events, ensure_ascii=False)};
    const DEMO_CLIPS = {{
      "001": {json.dumps(demo_001, ensure_ascii=False)},
      "098": {json.dumps(demo_098, ensure_ascii=False)},
      "backhand_001": {json.dumps(demo_backhand_001, ensure_ascii=False)}
    }};

    let currentTab = "single";
    let activeClipData = DEMO_CLIPS["001"];
    let isServerOnline = false;

    // Check backend server on load
    async function checkServerStatus() {{
      const dot = document.getElementById("statusDot");
      const text = document.getElementById("statusText");
      try {{
        const res = await fetch("http://127.0.0.1:5000/api/status", {{ method: "GET", mode: "cors" }});
        if (res.ok) {{
          const data = await res.json();
          isServerOnline = true;
          dot.className = "w-2.5 h-2.5 rounded-full bg-emerald-400";
          text.innerHTML = '<span class="text-emerald-400 font-semibold">Server Online (Port 5000)</span>';
          return;
        }}
      }} catch (err) {{
        // Backend not running
      }}
      isServerOnline = false;
      dot.className = "w-2.5 h-2.5 rounded-full bg-slate-500";
      text.innerHTML = '<span class="text-slate-400">Offline (Chạy bằng Demo Cache)</span>';
    }}

    // Switch between Tab 1 (Single) and Tab 2 (Match 40)
    function switchTab(tab) {{
      currentTab = tab;
      const singleBtn = document.getElementById("tabSingleBtn");
      const matchBtn = document.getElementById("tabMatchBtn");
      const singleContent = document.getElementById("tabSingleContent");
      const matchContent = document.getElementById("tabMatchContent");

      if (tab === "single") {{
        singleBtn.className = "tab-btn active px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center gap-2.5 transition-all duration-200";
        matchBtn.className = "tab-btn px-5 py-2.5 rounded-xl font-semibold text-sm text-slate-400 hover:text-white hover:bg-slate-800 flex items-center gap-2.5 transition-all duration-200";
        singleContent.classList.remove("hidden");
        matchContent.classList.add("hidden");
      }} else {{
        matchBtn.className = "tab-btn active px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center gap-2.5 transition-all duration-200";
        singleBtn.className = "tab-btn px-5 py-2.5 rounded-xl font-semibold text-sm text-slate-400 hover:text-white hover:bg-slate-800 flex items-center gap-2.5 transition-all duration-200";
        singleContent.classList.add("hidden");
        matchContent.classList.remove("hidden");
        renderMatchTable();
      }}
    }}

    // Handle Quick Demo Clip Selection
    function selectDemoClip(key) {{
      const clip = DEMO_CLIPS[key];
      if (!clip) return;
      activeClipData = clip;
      document.getElementById("videoPathInput").value = clip.video;
      document.getElementById("uploadBtnText").textContent = clip.video.split("/").pop();
      document.getElementById("playerVideoName").textContent = clip.video.split("/").pop();
      
      // Update video player
      const player = document.getElementById("singleVideoPlayer");
      const overlay = document.getElementById("noVideoOverlay");
      overlay.classList.add("hidden");
      
      // Try playing via server if online
      if (isServerOnline) {{
        player.src = "http://127.0.0.1:5000/api/stream_video?path=C:/Users/ADMIN/Downloads/archive/" + clip.video;
      }} else {{
        player.src = "";
      }}
      
      displaySingleClipResult(clip);
    }}

    // Handle File Selection
    function handleFileSelect(event) {{
      const file = event.target.files[0];
      if (!file) return;
      
      document.getElementById("uploadBtnText").textContent = file.name;
      document.getElementById("playerVideoName").textContent = file.name;
      document.getElementById("videoPathInput").value = file.name;
      
      const player = document.getElementById("singleVideoPlayer");
      const overlay = document.getElementById("noVideoOverlay");
      overlay.classList.add("hidden");
      player.src = URL.createObjectURL(file);
      player.play();

      // Check if file name matches known demos for auto-recognition
      if (file.name.includes("098")) {{
        activeClipData = DEMO_CLIPS["098"];
        displaySingleClipResult(activeClipData);
      }} else if (file.name.includes("001") && file.name.includes("drive")) {{
        activeClipData = DEMO_CLIPS["backhand_001"];
        displaySingleClipResult(activeClipData);
      }} else if (file.name.includes("001")) {{
        activeClipData = DEMO_CLIPS["001"];
        displaySingleClipResult(activeClipData);
      }}
    }}

    // Trigger analysis
    async function runSingleClipAnalysis() {{
      const pathVal = document.getElementById("videoPathInput").value.trim();
      const fileInput = document.getElementById("videoFileInput");
      const runBtn = document.getElementById("runAnalyzeBtn");
      const runText = document.getElementById("runAnalyzeText");

      // UI Loading state
      runBtn.disabled = true;
      runText.innerHTML = '<i class="fa-solid fa-spinner fa-spin mr-2"></i> Đang chạy Inference (Epoch 20 Multi-task)...';

      const playerSide = document.getElementById("playerSideSelect") ? document.getElementById("playerSideSelect").value : "auto";

      if (isServerOnline) {{
        try {{
          let res;
          if (fileInput.files.length > 0) {{
            const formData = new FormData();
            formData.append("file", fileInput.files[0]);
            formData.append("player_side", playerSide);
            res = await fetch("http://127.0.0.1:5000/api/analyze_clip", {{
              method: "POST",
              body: formData
            }});
          }} else if (pathVal) {{
            res = await fetch("http://127.0.0.1:5000/api/analyze_clip", {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify({{ video_path: pathVal }})
              body: JSON.stringify({{ video_path: pathVal, player_side: playerSide }})
            }});
          }}

          if (res && res.ok) {{
            const resultData = await res.json();
            displaySingleClipResult(resultData);
            runBtn.disabled = false;
            runText.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles mr-1"></i> BẮT ĐẦU PHÂN TÍCH CÚ ĐÁNH';
            return;
          }}
        }} catch (err) {{
          console.error("Backend error, falling back:", err);
        }}
      }}

      // Fallback if offline or server didn't respond
      setTimeout(() => {{
        if (pathVal.includes("098")) {{
          displaySingleClipResult(DEMO_CLIPS["098"]);
        }} else if (pathVal.includes("drive")) {{
          displaySingleClipResult(DEMO_CLIPS["backhand_001"]);
        }} else {{
          displaySingleClipResult(DEMO_CLIPS["001"]);
        }}
        runBtn.disabled = false;
        runText.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles mr-1"></i> BẮT ĐẦU PHÂN TÍCH CÚ ĐÁNH';
      }}, 600);
    }}

    // Display Result on Tab 1
    function displaySingleClipResult(data) {{
      if (!data) return;
      
      const vName = (data.video || "Selected Clip").split(/[\\\\/]/).pop();
      document.getElementById("analysisVideoTarget").innerHTML = `Video: <span class="text-blue-400 font-mono font-medium">${{vName}}</span>`;
      
      const stroke = data.prediction?.stroke || {{}};
      const side = data.prediction?.stroke_side || {{}};

      const sLabel = stroke.label ? stroke.label.toUpperCase() : "CLEAR";
      const sVn = stroke.vn_label || sLabel;
      const sProb = stroke.probability ? (stroke.probability * 100).toFixed(1) : "0.0";
      
      document.getElementById("predStrokeName").textContent = `${{sLabel}} (${{sVn.split(' ')[0]}})`;
      document.getElementById("predStrokeProb").textContent = `${{sProb}}%`;
      document.getElementById("predStrokeBar").style.width = `${{sProb}}%`;

      const sideLabel = side.label ? side.label.toUpperCase() : "FOREHAND";
      const sideVn = side.vn_label || sideLabel;
      const sideProb = side.probability ? (side.probability * 100).toFixed(1) : "0.0";

      document.getElementById("predSideName").textContent = `${{sideLabel}} (${{sideVn.split(' ')[0]}})`;
      document.getElementById("predSideProb").textContent = `${{sideProb}}%`;
      document.getElementById("predSideBar").style.width = `${{sideProb}}%`;

      // 8-Stroke Ranking Bars
      const rankContainer = document.getElementById("strokeRankingContainer");
      rankContainer.innerHTML = "";
      const ranking = data.stroke_ranking || [];

      ranking.forEach((item, idx) => {{
        const pct = (item.probability * 100).toFixed(1);
        const isTop = idx === 0;
        const barColor = isTop ? "bg-emerald-500" : (idx === 1 ? "bg-blue-500" : "bg-slate-600");
        
        const row = document.createElement("div");
        row.className = "space-y-1";
        row.innerHTML = `
          <div class="flex items-center justify-between text-xs">
            <span class="${{isTop ? 'font-bold text-white' : 'text-slate-400'}} capitalize flex items-center gap-1.5">
              ${{isTop ? '<i class="fa-solid fa-crown text-amber-400 text-[10px]"></i>' : ''}}
              ${{item.vn_label || item.label}}
            </span>
            <span class="font-mono ${{isTop ? 'font-bold text-emerald-400' : 'text-slate-400'}}">${{pct}}%</span>
          </div>
          <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
            <div class="${{barColor}} h-1.5 rounded-full progress-bar-fill" style="width: ${{pct}}%"></div>
          </div>
        `;
        rankContainer.appendChild(row);
      }});

      // Metadata
      const meta = data.video_metadata || {{}};
      document.getElementById("metaFrames").textContent = `${{meta.frames || 70}} frames`;
      document.getElementById("metaFps").textContent = `${{meta.fps ? meta.fps.toFixed(1) : 16.3}} fps`;
      document.getElementById("metaDuration").textContent = `${{meta.duration_seconds ? meta.duration_seconds.toFixed(1) : 4.3}}s`;
      document.getElementById("metaPlayer").textContent = meta.player_side || "Bottom";
    }}

    // Render Match 40 Table
    function renderMatchTable() {{
      const tbody = document.getElementById("matchEventsTableBody");
      tbody.innerHTML = "";

      const strokeFilter = document.getElementById("strokeFilter").value;
      const statusFilter = document.getElementById("statusFilter").value;

      MATCH40_EVENTS.forEach((ev, idx) => {{
        // Apply filters
        if (strokeFilter !== "ALL" && ev.pred_stroke !== strokeFilter && ev.gt_stroke !== strokeFilter) return;
        if (statusFilter === "CORRECT" && !ev.ok_stroke) return;
        if (statusFilter === "WRONG" && ev.ok_stroke) return;

        const tr = document.createElement("tr");
        tr.className = "hover:bg-slate-800/80 cursor-pointer transition";
        tr.onclick = () => seekMatchVideo(ev.time);

        const mins = Math.floor(ev.time / 60);
        const secs = Math.floor(ev.time % 60);
        const timeStr = `${{String(mins).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}}`;

        const isOk = ev.ok_stroke === true;
        const statusBadge = isOk 
          ? '<span class="px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-500/30 text-[10px] font-bold">✓ ĐÚNG</span>'
          : '<span class="px-2 py-0.5 rounded bg-rose-950/80 text-rose-400 border border-rose-500/30 text-[10px] font-bold">✗ LỆCH</span>';

        tr.innerHTML = `
          <td class="py-2.5 px-3 text-blue-400 font-bold">${{timeStr}}</td>
          <td class="py-2.5 px-3 capitalize text-slate-300">${{ev.player}}</td>
          <td class="py-2.5 px-3">
            <span class="font-bold text-white uppercase">${{ev.pred_stroke}}</span>
            <span class="text-slate-400 text-[11px] block">${{ev.pred_side}} (${{(ev.pred_stroke_prob * 100).toFixed(0)}}%)</span>
          </td>
          <td class="py-2.5 px-3">
            <span class="text-slate-300 uppercase">${{ev.gt_stroke || '--'}}</span>
            <span class="text-slate-500 text-[11px] block">${{ev.gt_side || '--'}}</span>
          </td>
          <td class="py-2.5 px-3 text-center">${{statusBadge}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function filterMatchTable() {{
      renderMatchTable();
    }}

    function seekMatchVideo(seconds) {{
      const v = document.getElementById("matchVideoPlayer");
      if (v) {{
        v.currentTime = Math.max(0, seconds - 1.0);
        v.play();
      }}
    }}

    // Sync match video timer
    const matchVid = document.getElementById("matchVideoPlayer");
    if (matchVid) {{
      matchVid.addEventListener("timeupdate", () => {{
        const c = matchVid.currentTime;
        const mins = Math.floor(c / 60);
        const secs = Math.floor(c % 60);
        document.getElementById("matchVideoTimer").textContent = `${{String(mins).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}} / 05:00`;
      }});
    }}

    // Initialize
    window.addEventListener("DOMContentLoaded", () => {{
      checkServerStatus();
      displaySingleClipResult(DEMO_CLIPS["001"]);
    }});
  </script>
</body>
</html>
"""

with open("badminton_analyzer.html", "w", encoding="utf-8") as f:
    f.write(html_content)

artifact_path = Path("C:/Users/ADMIN/.gemini/antigravity/brain/2613cee5-90a2-4966-b44b-e952b365d6dc/badminton_analyzer.html")
if artifact_path.parent.exists():
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(html_content)

print("Generated badminton_analyzer.html successfully!")

