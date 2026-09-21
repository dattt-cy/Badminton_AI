# Nội dung slide đã cập nhật theo hướng dự án mới

> Slide gốc: "Thị giác máy tính (2).pdf"  
> Hướng cũ: Phân loại 6 kỹ thuật + đánh giá lỗi sinh học + RAG tư vấn  
> Hướng mới: Định vị & phân loại cú đánh trong video trận đấu → stroke timeline

---

## So sánh nhanh — Cũ vs. Mới

| Hạng mục | Slide cũ | Slide mới |
|----------|----------|-----------|
| Đầu vào | Video 1 người, góc bên (side-view) | Video trận đơn, 2 người, camera cuối sân |
| Bài toán | Phân loại 6 kỹ thuật + đánh giá lỗi sinh học | Phân loại 8 nhóm cú đánh + định vị thời gian trong rally |
| Dataset | 953 clip tự thu, 2 lớp | Fine-Badminton (9.817 action, 8 lớp) + ShuttleSet |
| Model chính | ST-GCN++ 2 lớp, validation 96% | R(2+1)D-18 RGB + skeleton fusion, 8 lớp |
| Output | Góc khớp + lời khuyên RAG | StrokeEvent JSON + timeline video |
| Phạm vi | 1 người tập, offline | Trận đấu 2 người, pipeline end-to-end |

---

## Slide 1 — Tên đề tài (SỬA)

**Cũ:**
> HỆ THỐNG PHÂN LOẠI VÀ HỖ TRỢ CHỈNH SỬA KỸ THUẬT CHUYỂN ĐỘNG TRONG BỘ MÔN CẦU LÔNG

**Mới (đề xuất):**
> HỆ THỐNG ĐỊNH VỊ THỜI GIAN VÀ PHÂN LOẠI CÚ ĐÁNH TRONG VIDEO TRẬN ĐẤU CẦU LÔNG ĐƠN

*Lý do:* Hướng nghiên cứu đã chuyển sang phân tích trận đấu thực tế thay vì đánh giá kỹ thuật một người tập.

---

## Slide 2 — Nội dung trình bày (GIỮ KHUNG, SỬA NỘI DUNG)

**Cũ:**
- Phần 1: Tổng quan hệ thống
- Phần 2: Cơ sở công nghệ và thuật toán
- Phần 3: Xử lý dữ liệu và đánh giá mô hình

**Mới:**
- Phần 1: Tổng quan và động lực
- Phần 2: Dataset và bài toán
- Phần 3: Kiến trúc mô hình và kết quả

---

## Slide 3 — Vấn đề và động lực (SỬA)

**Cũ:** Tập trung vào lỗi sinh học (góc khớp sai), thiếu công cụ định lượng cho HLV.

**Nội dung mới:**

**VẤN ĐỀ THỰC TẾ**

Phân tích trận đấu cầu lông hiện nay phụ thuộc hoàn toàn vào con người:
- HLV và VĐV không có công cụ tự động để xem lại từng cú đánh trong trận.
- Không có cách tra cứu nhanh: "Trong 20 phút vừa đánh, có bao nhiêu cú smash?
  Bao nhiêu cú backhand net shot?"
- Broadcast chuyên nghiệp có đội phân tích riêng — phong trào và học đường không có.

**ĐỘNG LỰC**

→ Xây hệ thống tự động nhận diện từng cú đánh trong video trận đấu:
- Ai đánh, đánh loại gì, forehand hay backhand, tại thời điểm nào.
- Output là timeline có thể bấm để xem lại đúng cú đánh đó.
- Làm nền tảng cho phân tích chiến thuật phía trên.

---

## Slide 4 — Kiến trúc hệ thống (VIẾT LẠI)

**Cũ:** Kiến trúc 3 bước — YOLOv8 Pose → Toán học góc khớp → RAG tư vấn

**Nội dung mới:**

**PIPELINE ĐÃ CHỐT**

```
Video trận đấu (rally / đoạn ngắn)
  │
  ├─ [Bước 1] Court Detection
  │    Tự động nhận diện 4 góc sân
  │
  ├─ [Bước 2] Shuttle Tracking (TrackNet)
  │    Theo dõi quỹ đạo cầu từng frame
  │
  ├─ [Bước 3] Hit Detection
  │    Phát hiện điểm chạm vợt → xác định người đánh
  │
  ├─ [Bước 4] Stroke Classification (Fusion Model)
  │    RGB branch (R(2+1)D-18) + Skeleton branch
  │    → 8 nhóm cú đánh + forehand/backhand/aroundhead
  │
  └─ [Output] StrokeEvent JSON + Timeline web
```

**Lý do giữ pipeline này:**
Repository đã có API upload video, job bất đồng bộ, pose estimation, timeline viewer và 181 unit test đang vượt qua. Không viết lại từ đầu.

---

## Slide 5 — Phạm vi và ràng buộc (VIẾT LẠI)

**Cũ:** Video 1 người, góc bên, phân tích 6 kỹ thuật.

**Nội dung mới:**

**ĐẦU VÀO HỢP LỆ**
- Trận đơn — hai người chơi
- Camera cố định ở cuối sân, nhìn thấy toàn bộ sân
- Video ngang, tối thiểu 720p, FPS ổn định
- Phiên bản đầu: một rally hoặc đoạn ngắn; video toàn trận là bước sau

**8 NHÓM CÚ ĐÁNH (Taxonomy MVP)**

| Nhãn | Mô tả |
|------|-------|
| `serve` | Giao cầu |
| `clear` | Lốc cao và sâu |
| `smash` | Đập mạnh từ trên xuống |
| `drop` | Thả gần lưới đối phương |
| `net_shot` | Đánh sát lưới |
| `lift` | Bổng từ lưới về cuối sân |
| `drive` | Phẳng nhanh qua lưới |
| `net_attack` | Tấn công lưới (push/kill) |

**NGOÀI PHẠM VI PHIÊN BẢN ĐẦU**
- Đánh đôi
- Video điện thoại tự do, nhiều góc
- Đánh giá lỗi sinh học hoặc chẩn đoán chấn thương

---

## Slide 6 — Đối tượng và giá trị thực tiễn (SỬA NHẸ)

**Cũ:** Tập trung vào người chơi self-training và HLV xem góc khớp.

**Nội dung mới:**

**VĐV & NGƯỜI CHƠI**
- Xem lại trận đấu theo từng cú đánh, không cần tua video thủ công.
- Biết tỷ lệ từng loại cú đánh trong trận: bao nhiêu % smash, bao nhiêu % net shot.

**HUẤN LUYỆN VIÊN**
- Phân tích chiến thuật: VĐV đang dùng forehand hay backhand nhiều hơn?
- Tìm pattern cú đánh quen thuộc và điểm yếu chiến thuật.
- Mỗi cú đánh có link video để xem lại bằng chứng.

**NHÀ NGHIÊN CỨU / PHÂN TÍCH**
- Dataset có cấu trúc từ trận đấu thực: stroke timeline theo từng rally.
- Nền tảng để xây Shot Influence model và dự đoán kết quả rally.

---

## Slide 7 — Công nghệ và thuật toán (VIẾT LẠI)

**Cũ:** YOLOv8-Pose + ST-GCN không-thời gian + RAG hybrid search.

**Nội dung mới:**

**MÔ HÌNH PHÂN LOẠI CÚ ĐÁNH — RGB BRANCH**

- **R(2+1)D-18** pretrained trên Kinetics, fine-tune trên Fine-Badminton 8 lớp.
- Input: clip video 60 frame căn quanh điểm chạm cầu, crop theo người đánh.
- Kết quả epoch 10 full-data:
  - Validation stroke Macro-F1: **70.27%**
  - Side Macro-F1: **78.01%**
  - Test (200 clip ShuttleSet): stroke accuracy **67.50%**, side accuracy **74.50%**

**MÔ HÌNH PHÁT HIỆN ĐIỂM CHẠM — HIT DETECTOR**

- 3 lớp: `no_hit`, `upper_hit`, `lower_hit`
- Input: pose hai người + thông tin quỹ đạo cầu (TrackNet)
- Đánh giá tại tolerance ±2, ±5, ±15 frame

**THEO DÕI CẦU — TrackNetV3**

- Phát hiện và theo dõi cầu trong từng frame
- Output CSV tọa độ cầu → dùng để xác định điểm chạm vợt

**NHẬN DIỆN POSE — YOLOv8-Pose**

- Trích xuất 17 keypoint của từng người chơi
- Dùng để xác định người đánh (upper/lower) và crop đúng vùng

---

## Slide 8 — Dataset (VIẾT LẠI HOÀN TOÀN)

**Cũ:** 953 clip tự thu, 2 lớp (Backhand Drive + Forehand Clear).

**Nội dung mới:**

**DATASET CHÍNH — Fine-Badminton**

| Thông số | Giá trị |
|----------|---------|
| Số video | 10 video trận đấu |
| Số action | 9.817 intervals |
| Số lớp | 29 raw labels → 8 coarse classes |
| Split | 6 trận train / 2 trận val / 2 trận test |
| Train samples | 5.921 |
| Val samples | 1.980 |
| Test samples | 1.916 |

**Quy tắc split:** Theo **trận** (match-level), không split ngẫu nhiên theo clip để tránh data leakage.

**DATASET PHỤ — ShuttleSet**

- 44 trận, 36.482 stroke có annotation `hit_frame`, `player_side`, `shot_type`
- Dùng để train head phân loại **forehand / backhand / aroundhead**
- 35.041 stroke hợp lệ (sau khi lọc 1.441 dòng có cờ lỗi)

**DATASET ĐỐI CHIẾU — BFMD**

- 19 trận, 16.751 hit; có court, pose, shuttle, shot type
- Dùng cho external validation end-to-end (không tham gia train)

---

## Slide 9 — Kỹ thuật xử lý dữ liệu (CẬP NHẬT)

**Cũ:** Augmentation flip + nhiễu Gaussian + nội suy tọa độ cổ tay.

**Nội dung mới:**

**PIPELINE CHUẨN BỊ DỮ LIỆU**

```
Fine-Badminton MP4 (full match)
    │
    ├─ Virtual-clip loader: đọc trực tiếp theo start/end frame
    │   (không cắt video vật lý)
    │
    ├─ YOLOv8-Pose: trích skeleton → xác định player side
    │
    ├─ Crop người đánh: lấy bounding box ± margin
    │
    └─ Augmentation (chỉ khi train):
         - Rotation ±6.9°
         - Scale ±10%
         - Horizontal flip (mô phỏng tay thuận/trái)
```

**XỬ LÝ POSE QUALITY**
- Tracking: theo dõi liên tục, fill 0 nếu mất người
- Interpolation: nội suy tuyến tính tối đa 4 frame bị mất
- Smoothing: làm mượt tọa độ theo thời gian
- Confidence gate: loại keypoint có confidence < ngưỡng

---

## Slide 10 — Tech Stack (CẬP NHẬT)

**Cũ:** Next.js + Spring Boot + FastAPI + MySQL

**Nội dung mới (AI Classifier service):**

| Layer | Công nghệ | Vai trò |
|-------|-----------|---------|
| AI Pipeline | Python 3.11, PyTorch, OpenCV | Xử lý video, inference |
| Pose | Ultralytics YOLOv8 | Trích skeleton |
| Action Recognition | R(2+1)D-18, PySKL/ST-GCN++ | Phân loại cú đánh |
| Shuttle Tracking | TrackNetV3 | Theo dõi quỹ đạo cầu |
| API | FastAPI + Uvicorn | Nhận video, quản lý job |
| Job Store | File-based JSON | Lưu trạng thái & kết quả |
| Frontend | Static HTML / Web | Timeline viewer |

**Môi trường chạy (tách biệt tránh xung đột):**
```
main/API env   → Python 3.11, FastAPI, OpenCV, Ultralytics
pyskl env      → Python 3.10, PySKL/MMCV (ST-GCN++)
tracking env   → TrackNetV3
```

---

## Slide 11 — Quy hoạch dataset và pipeline (CẬP NHẬT)

**Cũ:** Pipeline YOLO → SQL → pkl → ST-GCN.

**Nội dung mới:**

**BỐN PHASE PHÁT TRIỂN**

```
Phase A (hiện tại)
  Phân loại clip với boundary đã biết
  → player_side + 8 lớp cú đánh + forehand/backhand

Phase B (tiếp theo)
  Phát hiện hit_frame trong rally chưa cắt
  → hit recall/precision tại ±2, ±5, ±15 frame

Phase C (end-to-end)
  Upload rally → stroke timeline đầy đủ
  → StrokeEvent JSON + web viewer

Phase D (tùy chọn)
  Phân tích chiến thuật từ event stream
  → pattern cú đánh, tỷ lệ thắng, heatmap
```

---

## Slide 12 — Kỹ thuật transform và chống nhiễu (GIỮ PHẦN LỚN, ĐIỀU CHỈNH)

**Giữ nguyên:** Augmentation flip, Gaussian noise, frame rate variation, nội suy khi confidence thấp.

**Bổ sung mới:**

**XỬ LÝ DOMAIN SHIFT GIỮA CÁC DATASET**
- Mỗi sample có `dataset_id`, `domain`, `match_id` để audit nguồn gốc.
- Không trộn Fine-Badminton và ShuttleSet rồi split ngẫu nhiên.
- Source-balanced sampler: ShuttleSet không lấn át Fine-Badminton khi train joint.

**MASKED LOSS CHO MULTI-TASK**
```
L = L_stroke + λ_side × mask_side × L_side
```
- Fine-Badminton chỉ đóng góp `L_stroke` (không có nhãn forehand/backhand).
- ShuttleSet đóng góp cả `L_stroke` và `L_side`.

---

## Slide 13 — Thuật toán định vị frame chạm cầu (CẬP NHẬT)

**Cũ:** Dùng peak velocity cổ tay (kinematic tracking).

**Nội dung mới:**

**HAI PHƯƠNG PHÁP PHÁT HIỆN HIT FRAME**

**Phương pháp 1 — Heuristic (Baseline, đã có)**
- Phát hiện đổi hướng quỹ đạo cầu từ TrackNet
- Kết hợp motion peak cổ tay/khuỷu từ pose
- Luật hai người đánh luân phiên (alternating rule)

**Phương pháp 2 — Hit Detector (đang phát triển)**
- Phân loại theo frame: `no_hit` / `upper_hit` / `lower_hit`
- Input: pose hai người + thông tin cầu quanh candidate frame
- Model: Random Forest → sau đó thử temporal model
- Đánh giá: Precision/Recall tại tolerance **±2, ±5, ±15 frame**

Pilot hiện tại (train 3 trận, test 1 trận):
- Validation Macro-F1: **58.92%**
- Test Macro-F1: **68.52%**, accuracy **75.20%**

---

## Slide 14 — Đánh giá hiệu năng mô hình (VIẾT LẠI)

**Cũ:** YOLOv8-Pose mAP + ST-GCN 2 lớp 96.15%.

> ⚠️ Kết quả 96.15% là trên bài toán **2 lớp** với data tự thu. Không dùng con số này làm kết quả nghiên cứu mới.

**Metric đánh giá (hướng mới):**

**Phân loại clip (Phase A)**

| Metric | Lý do dùng |
|--------|-----------|
| **Macro-F1** | Metric chính — lớp mất cân bằng |
| Accuracy, Balanced Accuracy | Báo cáo bổ sung |
| Precision, Recall, F1 từng lớp | Phân tích lỗi per-class |
| Confusion Matrix | Phát hiện nhầm lẫn giữa các cú tương tự |
| Joint Accuracy | Đúng cả `stroke` AND `stroke_side` cùng lúc |

**Phát hiện hit frame (Phase B)**

| Metric | Ngưỡng |
|--------|--------|
| Precision / Recall / F1 | ±2 frame, ±5 frame, ±15 frame |
| Mean / Median absolute frame error | — |

**End-to-end (Phase C)**

Một event chỉ đúng khi đồng thời:
```
tIoU ≥ ngưỡng
AND player_side đúng
AND stroke đúng
AND stroke_side đúng
```

---

## Slide 15 — Dataset thực tế (VIẾT LẠI)

**Cũ:** 1.580 samples sau augmentation, 2 lớp.

**Nội dung mới:**

**FINE-BADMINTON — Dataset chính Phase A**

```
10 video trận đấu chuyên nghiệp
9.817 intervals có annotation start/end

Split theo trận (không theo clip):
  Train:       5.921 clips  (6 trận)
  Validation:  1.980 clips  (2 trận)
  Test:        1.916 clips  (2 trận)

8 lớp đều có đủ sample ở 3 split:
  Lớp nhỏ nhất (net_attack):
    Train 140 / Val 55 / Test 41
```

**SO SÁNH VỚI DỮ LIỆU CŨ**

| | Dataset cũ | Dataset mới |
|-|-----------|------------|
| Số lớp | 2 | 8 |
| Nguồn | Tự thu | Chuyên nghiệp (Fine-Badminton) |
| Split | Theo người | Theo trận |
| Số samples | 1.580 | 9.817 |

---

## Slide 16 — Quá trình huấn luyện (CẬP NHẬT)

**Cũ:** ST-GCN++ 2 lớp, epoch 23, validation 96.15%.

**Nội dung mới:**

**RGB CLASSIFIER — R(2+1)D-18**

- Pretrained: Kinetics-400
- Fine-tune: Fine-Badminton 8 lớp
- Epoch chốt: **Epoch 10** (full-data)
- Lý do chốt: không tiếp tục tăng epoch vô hạn, ưu tiên đánh giá external

**Kết quả Epoch 10:**

| Split | Stroke Metric | Side Metric |
|-------|--------------|------------|
| Validation | Macro-F1 **70.27%** | Macro-F1 **78.01%** |
| Test (ShuttleSet 200 clips) | Accuracy **67.50%** | Side accuracy **74.50%** |
| Test joint accuracy | — | **50.50%** |

**SKELETON BASELINE — ST-GCN++**
- Tái sử dụng pipeline PySKL hiện có
- Đang train lại head 8 lớp trên Fine-Badminton
- Vai trò: đo giá trị bổ sung của pose so với RGB

---

## Slide 17 — Đánh giá kết quả mô hình (CẬP NHẬT)

**Cũ:** 300/312 đúng, 96.15% top-1, confusion matrix 2 lớp.

**Nội dung mới:**

**KẾT QUẢ RGB BASELINE (R(2+1)D-18 · 8 lớp)**

| Metric | Giá trị |
|--------|---------|
| Stroke Macro-F1 (val) | **70.27%** |
| Side Macro-F1 (val) | **78.01%** |
| Stroke accuracy (test ShuttleSet) | **67.50%** |
| Side accuracy (test ShuttleSet) | **74.50%** |
| Joint accuracy | **50.50%** |

**NHẬN XÉT**

- Model vượt majority baseline rõ ràng (baseline ~12.5% với 8 lớp đều nhau).
- Joint accuracy 50.5% cho thấy còn không gian cải thiện lớn khi kết hợp skeleton và trajectory.
- Các cặp hay nhầm: `clear/drop/smash` (đều là overhead), `lift/net_shot` (đều từ lưới).

**BƯỚC TIẾP THEO**
1. Thêm skeleton branch → fusion với RGB → đo cải thiện joint accuracy.
2. Train hit detector → đánh giá full pipeline trên 2 trận external.
3. So sánh hit detector vs. ActionFormer end-to-end.

---

## Output mẫu (thêm vào slide cuối)

Thêm slide minh họa output thực tế của hệ thống:

**Một cú đánh:**
```json
{
  "stroke":      "drop",
  "stroke_side": "forehand",
  "player_side": "upper",
  "hit_frame":   68,
  "confidence":  0.98
}
```

**Một rally (Phase C — tương lai):**
```json
{
  "events": [
    { "hit_frame": 1221, "player_side": "upper",
      "stroke": "net_shot", "stroke_side": "backhand", "confidence": 0.81 },
    { "hit_frame": 1362, "player_side": "lower",
      "stroke": "clear",    "stroke_side": "forehand", "confidence": 0.93 }
  ]
}
```

---

## Những gì cần XÓA hoặc KHÔNG ĐỀ CẬP trong slide mới

| Nội dung cũ | Lý do bỏ |
|------------|---------|
| Tính góc khớp (biomechanics) | Chưa có expert-labelled error dataset |
| RAG hybrid search / tư vấn lỗi | Không có cơ sở dữ liệu lỗi có cấu trúc |
| Kết quả 96.15% (2 lớp tự thu) | Không đại diện cho bài toán mới 8 lớp |
| Video 1 người, góc bên (side-view) | Hướng mới là trận đấu 2 người, camera cuối sân |
| Spring Boot + MySQL | Thuộc hệ thống khác, không phải AI module |
| 6 kỹ thuật cũ (forehand/backhand drive, lift, net, clear) | Thay bằng 8 nhóm taxonomy mới |

