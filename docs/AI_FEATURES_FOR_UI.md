# Chức năng AI — Tài liệu cho Frontend

> Mục đích: Mô tả những gì AI có thể làm và dữ liệu trả về,  
> để đội Frontend thiết kế giao diện phù hợp.  
> Backend & API sẽ do hệ thống khác đảm nhận.

---

## AI làm được gì?

Người dùng upload một video trận cầu lông → AI phân tích và trả về
**danh sách các cú đánh** xảy ra trong video, mỗi cú đánh có đầy đủ thông tin
để hiển thị trên giao diện.

---

## Chức năng 1 — Phân loại loại cú đánh

AI nhận diện cú đánh thuộc loại nào trong **8 loại chính**:

| Nhãn | Tên | Mô tả ngắn |
|------|-----|-----------|
| `serve` | Giao cầu | Lúc bắt đầu rally |
| `clear` | Lốc | Đánh cao và sâu về cuối sân đối phương |
| `smash` | Đập | Cú đập mạnh từ trên cao xuống |
| `drop` | Thả | Cú nhẹ rơi gần lưới đối phương |
| `net_shot` | Lưới | Đánh sát lưới, cầu vừa qua lưới |
| `lift` | Bổng | Đánh bổng từ gần lưới về cuối sân |
| `drive` | Phẳng | Cú đánh ngang nhanh, thấp qua lưới |
| `net_attack` | Tấn công lưới | Push / net kill sát lưới |

**Dữ liệu trả về:**

```json
"stroke": "drop",
"stroke_ranking": [
  { "label": "drop",       "probability": 0.98 },
  { "label": "clear",      "probability": 0.01 },
  { "label": "smash",      "probability": 0.005 },
  ...
]
```

> `stroke_ranking` là xếp hạng toàn bộ 8 loại, từ khả năng cao nhất đến thấp nhất.  
> UI có thể dùng để vẽ biểu đồ hoặc hiển thị top-3.

---

## Chức năng 2 — Nhận diện hướng đánh (forehand / backhand)

AI xác định người chơi đánh bằng tay thuận, trái tay hay vòng đầu:

| Nhãn | Mô tả |
|------|-------|
| `forehand` | Thuận tay |
| `backhand` | Trái tay |
| `aroundhead` | Thuận tay nhưng vòng qua đầu (đánh sang bên trái) |
| `unknown` | Không xác định được |

**Dữ liệu trả về:**

```json
"stroke_side": "forehand",
"stroke_side_ranking": [
  { "label": "forehand",   "probability": 1.0 },
  { "label": "aroundhead", "probability": 0.0001 },
  { "label": "backhand",   "probability": 0.000002 }
]
```

---

## Chức năng 3 — Xác định người đánh

AI xác định ai trong video là người thực hiện cú đánh đó:

| Nhãn | Mô tả |
|------|-------|
| `upper` | Người đứng phía trên trong frame (nửa sân xa) |
| `lower` | Người đứng phía dưới trong frame (nửa sân gần) |

**Dữ liệu trả về:**

```json
"player_side": "upper"
```

> Camera nhìn từ cuối sân, nên người phía dưới = người gần camera,
> người phía trên = người ở đầu sân bên kia.

---

## Chức năng 4 — Xác định thời điểm chạm cầu

AI tìm đúng frame trong video mà người chơi vợt chạm vào cầu.

**Dữ liệu trả về:**

```json
"hit_frame": 1221,
"start_frame": 1210,
"end_frame": 1234
```

| Field | Ý nghĩa |
|-------|---------|
| `hit_frame` | Frame chính xác lúc chạm vợt |
| `start_frame` | Frame bắt đầu cú đánh (lúc chuẩn bị swing) |
| `end_frame` | Frame kết thúc cú đánh (follow-through) |

> Để seek video đến đúng thời điểm: `giây = frame / fps`  
> (FPS thường là 25, 30 hoặc 60 tùy video)

---

## Chức năng 5 — Phát hiện mặt sân (Court Detection)

AI tự động nhận diện 4 góc của sân cầu lông trong video.

**Dữ liệu trả về:**

```json
"court_corners": [
  [520, 900],   // góc dưới trái
  [1400, 900],  // góc dưới phải
  [1620, 500],  // góc trên phải
  [300, 500]    // góc trên trái
],
"court_auto_detected": true
```

> Nếu `court_auto_detected: false` → AI dùng góc mặc định, kết quả kém chính xác hơn.  
> UI có thể hiển thị ảnh preview mặt sân đã được AI vẽ lên (`court_image`).

---

## Chức năng 6 — Theo dõi quỹ đạo cầu (Trajectory Tracking)

AI dùng model TrackNet để theo dõi cầu bay trong từng frame.  
Thông tin này dùng nội bộ để phát hiện điểm chạm cầu — **không trực tiếp trả
về cho UI** ở phiên bản hiện tại, nhưng sẽ có ở phase sau.

---

## Đầu ra tổng hợp — Một cú đánh

Khi phân tích xong **một cú đánh**, AI trả về cấu trúc như sau:

```json
{
  "stroke":      "drop",
  "stroke_side": "forehand",
  "player_side": "upper",
  "hit_frame":   1221,
  "confidence":  0.98,

  "stroke_ranking": [
    { "label": "drop",       "probability": 0.98 },
    { "label": "clear",      "probability": 0.01 },
    { "label": "smash",      "probability": 0.005 },
    { "label": "drive",      "probability": 0.0001 },
    { "label": "net_shot",   "probability": 0.0001 },
    { "label": "lift",       "probability": 0.000001 },
    { "label": "net_attack", "probability": 0.0000005 },
    { "label": "serve",      "probability": 0.0 }
  ],

  "stroke_side_ranking": [
    { "label": "forehand",   "probability": 1.0 },
    { "label": "aroundhead", "probability": 0.00006 },
    { "label": "backhand",   "probability": 0.0000017 }
  ]
}
```

---

## Đầu ra tổng hợp — Nhiều cú đánh (Phase sau)

Khi AI phân tích **cả một rally hoặc video dài**, trả về danh sách:

```json
{
  "events": [
    {
      "start_frame": 1210,
      "hit_frame":   1221,
      "end_frame":   1234,
      "player_side": "upper",
      "stroke":      "net_shot",
      "stroke_side": "backhand",
      "confidence":  0.81
    },
    {
      "start_frame": 1350,
      "hit_frame":   1362,
      "end_frame":   1378,
      "player_side": "lower",
      "stroke":      "clear",
      "stroke_side": "forehand",
      "confidence":  0.93
    }
  ]
}
```

> Mỗi phần tử trong `events[]` là một cú đánh.  
> UI cần render danh sách này thành **timeline** bên cạnh video player.

---

## Những gì AI KHÔNG làm (phạm vi ngoài)

| Tính năng | Trạng thái |
|-----------|-----------|
| Đánh đôi (4 người) | ❌ Không hỗ trợ |
| Video một mình tập | ❌ Không hỗ trợ |
| Real-time (stream live) | ❌ Không hỗ trợ |
| Đánh giá kỹ thuật đúng/sai | ❌ Chưa có (cần dataset chuyên gia) |
| Tư vấn chiến thuật bằng AI chat | ❌ Chưa có |
| Phân tích 29 loại cú đánh chi tiết | 🔜 Tương lai |
| Thống kê chiến thuật cả trận | 🔜 Tương lai |

---

## Gợi ý giao diện theo từng chức năng

### Màn hình upload

- Kéo thả hoặc chọn file video (MP4, AVI, MOV, MKV)
- Hiển thị yêu cầu: video trận đơn, camera cố định cuối sân, tối thiểu 720p
- Thanh tiến trình khi AI đang xử lý — hiển thị từng bước:
  - Nhận diện mặt sân
  - Theo dõi quỹ đạo cầu
  - Tìm điểm chạm vợt
  - Phân loại cú đánh

### Màn hình kết quả — một cú đánh

```
┌──────────────────────────────────────┐
│  🏸 Drop Shot · Forehand             │
│  Người đánh: phía trên sân           │
│  Thời điểm: 40.7 giây (frame 1221)   │
├──────────────────────────────────────┤
│  Xếp hạng AI:                        │
│  Drop     ████████████████  98%      │
│  Clear    █                 1.4%     │
│  Smash    ▌                 0.5%     │
├──────────────────────────────────────┤
│  Hướng đánh:                         │
│  Forehand ████████████████  100%     │
└──────────────────────────────────────┘
```

### Màn hình kết quả — nhiều cú đánh (Phase sau)

```
[Video Player                          ]
[======●==========●=========●=========]
       ↑drop/FH   ↑clear/FH ↑smash/BH

Sự kiện:
 0:40  upper  Drop · Forehand    98% ▶
 0:57  lower  Clear · Forehand   93% ▶
 1:12  upper  Smash · Backhand   87% ▶
```

---

## Trạng thái AI cần hiển thị

| Trạng thái | Mô tả | Gợi ý UI |
|-----------|-------|---------|
| Đang xử lý | AI đang chạy từng bước | Progress bar + text bước hiện tại |
| Thành công | Có kết quả | Hiển thị kết quả |
| Thất bại | Video không hợp lệ / lỗi | Thông báo lỗi cụ thể |
| Không chắc | confidence thấp | Badge "Không chắc" hoặc màu khác |
| Ngoài phạm vi | Video không đúng loại | Giải thích và hướng dẫn upload lại |

