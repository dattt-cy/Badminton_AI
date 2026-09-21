# Tài liệu API & Hướng phát triển UI — Badminton AI Classifier

> Cập nhật: 19/09/2026  
> Mục đích: Tài liệu tham chiếu cho đội phát triển UI, tóm tắt tất cả chức năng
> backend đang hoạt động, định dạng JSON trả về và hướng phát triển tiếp theo.

---

## Tổng quan hệ thống

Dự án phân tích video trận đấu cầu lông đơn, tự động nhận diện từng cú đánh
trong video và trả về timeline sự kiện. Pipeline gồm 4 bước chính:

```
Video upload
  → Phát hiện góc sân (Court Detection)
  → Theo dõi quỹ đạo cầu (TrackNet)
  → Tìm điểm chạm vợt (Hit Detection)
  → Phân loại cú đánh (Fusion AI Model)
  → Trả về StrokeEvent JSON + timeline
```

Backend dùng **FastAPI**, xử lý bất đồng bộ theo mô hình Job (upload → nhận
`job_id` → poll trạng thái → lấy kết quả).

---

## Base URL

```
http://localhost:8000
```

---

## Danh sách endpoint

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/health` | Kiểm tra server còn sống |
| `GET` | `/` | Trang web giao diện |
| `POST` | `/classify` | Upload file video để phân tích |
| `POST` | `/classify_path` | Gửi đường dẫn file local để phân tích |
| `GET` | `/status/{job_id}` | Kiểm tra tiến trình job |
| `GET` | `/jobs/{filename}` | Lấy file artifact (ảnh sân, JSON kết quả) |

> API nâng cao (v1) tại `/v1/analyses` — dùng cho tích hợp sâu hơn, chi tiết ở cuối tài liệu.

---

## Chi tiết từng endpoint

---

### `GET /health`

Kiểm tra server có đang hoạt động không.

**Response:**

```json
{
  "status": "ok",
  "accepting_jobs": true,
  "classifier_enabled": false,
  "max_upload_bytes": 524288000
}
```

| Field | Ý nghĩa |
|-------|---------|
| `status` | Luôn là `"ok"` nếu server chạy |
| `accepting_jobs` | Server có nhận job mới không |
| `classifier_enabled` | Có bật WSL classifier không |
| `max_upload_bytes` | Giới hạn file upload (mặc định 500 MB) |

---

### `POST /classify`

Upload video để bắt đầu phân tích. Gửi dưới dạng **multipart form**.

**Request:**

```
Content-Type: multipart/form-data
file: <video file>  (MP4 / AVI / MOV / MKV)
```

**Response (202 Accepted):**

```json
{
  "job_id": "a3f2b1c4"
}
```

Dùng `job_id` này để kiểm tra trạng thái ở endpoint `/status/{job_id}`.

---

### `POST /classify_path`

Gửi đường dẫn file video trên máy local (không cần upload lại).

**Request body (JSON):**

```json
{
  "path": "C:/Users/ADMIN/Downloads/match_video.mp4"
}
```

**Response (202 Accepted):**

```json
{
  "job_id": "b7d4e2f1"
}
```

---

### `GET /status/{job_id}`

Poll trạng thái của một job. UI nên gọi định kỳ (mỗi 2–3 giây) cho đến khi
`status` là `"done"` hoặc `"error"`.

**Các trạng thái có thể:**

| `status` | `step` ví dụ | `progress` |
|----------|-------------|------------|
| `queued` | "Đang chuẩn bị..." | 0 |
| `processing` | "Đang nhận diện mặt sân..." | 10 |
| `processing` | "Đang theo dõi quỹ đạo cầu (TrackNet)..." | 20 |
| `processing` | "Đang tìm điểm chạm vợt..." | 60 |
| `processing` | "Đang phân loại cú đánh (AI Fusion)..." | 80 |
| `done` | "Hoàn thành!" | 100 |
| `error` | "Lỗi: ..." | 0 |

**Response khi đang xử lý:**

```json
{
  "job_id": "a3f2b1c4",
  "filename": "match.mp4",
  "status": "processing",
  "step": "Đang theo dõi quỹ đạo cầu (TrackNet)...",
  "progress": 20,
  "created_at": 1726726754.123,
  "court_auto": true,
  "corners": [
    [520, 900],
    [1400, 900],
    [1620, 500],
    [300, 500]
  ]
}
```

**Response khi hoàn thành (`status: "done"`):**

```json
{
  "job_id": "a3f2b1c4",
  "filename": "match.mp4",
  "status": "done",
  "step": "Hoàn thành!",
  "progress": 100,
  "created_at": 1726726754.123,
  "court_auto": true,
  "corners": [[520, 900], [1400, 900], [1620, 500], [300, 500]],
  "court_image": "/jobs/a3f2b1c4_court.jpg",
  "result": {
    "stroke": "drop",
    "stroke_side": "forehand",
    "stroke_ranking": [
      { "label": "drop",       "probability": 0.9814 },
      { "label": "clear",      "probability": 0.0137 },
      { "label": "smash",      "probability": 0.0050 },
      { "label": "drive",      "probability": 0.0001 },
      { "label": "net_shot",   "probability": 0.0001 },
      { "label": "lift",       "probability": 0.0000 },
      { "label": "net_attack", "probability": 0.0000 },
      { "label": "serve",      "probability": 0.0000 }
    ],
    "stroke_side_ranking": [
      { "label": "forehand",   "probability": 1.0000 },
      { "label": "aroundhead", "probability": 0.0001 },
      { "label": "backhand",   "probability": 0.0000 }
    ],
    "event_frame": 68,
    "player_side": "upper",
    "court_auto_detected": true
  }
}
```

**Response khi lỗi (`status: "error"`):**

```json
{
  "job_id": "a3f2b1c4",
  "status": "error",
  "step": "Lỗi: Không tìm thấy điểm chạm vợt trong video.",
  "progress": 0
}
```

---

### `GET /jobs/{filename}`

Lấy file artifact của một job (ảnh sân, file JSON phụ...).

**Ví dụ:**

```
GET /jobs/a3f2b1c4_court.jpg
```

Trả về file ảnh trực tiếp (image/jpeg). UI dùng URL này để hiển thị ảnh mặt sân được phát hiện.

---

## Mô tả các trường kết quả

### `result.stroke` — Loại cú đánh

| Giá trị | Tên đầy đủ |
|---------|-----------|
| `serve` | Giao cầu |
| `clear` | Cú lốc (clear) |
| `smash` | Đập cầu (smash) |
| `drop` | Cú thả (drop shot) |
| `net_shot` | Cú lưới (net shot) |
| `lift` | Cú bổng (lift/lob) |
| `drive` | Cú phẳng (drive) |
| `net_attack` | Tấn công lưới (push/net kill) |

### `result.stroke_side` — Hướng đánh

| Giá trị | Mô tả |
|---------|-------|
| `forehand` | Thuận tay |
| `backhand` | Trái tay |
| `aroundhead` | Thuận tay vòng đầu (đánh bên trái nhưng dùng forehand) |
| `unknown` | Không xác định |

### `result.player_side` — Vị trí người đánh trong video

| Giá trị | Mô tả |
|---------|-------|
| `upper` | Người đứng phía trên trong frame |
| `lower` | Người đứng phía dưới trong frame |

### `result.event_frame` — Frame chạm vợt

Số frame trong video tại thời điểm người chơi chạm vợt vào cầu. UI có thể dùng
để seek video đến đúng thời điểm đó (`event_frame / fps` = giây).

### `stroke_ranking` & `stroke_side_ranking`

Danh sách xếp hạng xác suất cho tất cả các nhãn, từ cao đến thấp. Dùng để hiển
thị biểu đồ confidence hoặc top-N predictions.

---

## Định dạng StrokeEvent (mục tiêu tương lai — Phase B+)

Khi pipeline hỗ trợ video dài chứa nhiều lần đánh (rally), output sẽ là một
danh sách sự kiện:

```json
{
  "events": [
    {
      "start_frame": 1210,
      "hit_frame": 1221,
      "end_frame": 1234,
      "player_side": "upper",
      "stroke": "net_shot",
      "stroke_side": "backhand",
      "confidence": 0.81
    },
    {
      "start_frame": 1350,
      "hit_frame": 1362,
      "end_frame": 1378,
      "player_side": "lower",
      "stroke": "clear",
      "stroke_side": "forehand",
      "confidence": 0.93
    }
  ]
}
```

UI phải cho phép bấm vào từng event để seek video đến `hit_frame`.

---

## Gợi ý thiết kế UI theo từng phase

### Phase hiện tại (M1 — một cú đánh mỗi lần)

UI tối giản:

1. **Upload form** — kéo thả hoặc chọn file video.
2. **Progress bar** — hiển thị `step` + `progress` khi poll `/status/{job_id}`.
3. **Court preview** — hiển thị ảnh `/jobs/{job_id}_court.jpg` nếu có.
4. **Kết quả** — thẻ hiển thị:
   - Tên cú đánh lớn (`stroke` + `stroke_side`)
   - Hướng dẫn: người đánh (`player_side`), frame chạm cầu (`event_frame`)
   - Biểu đồ thanh ngang top-4 từ `stroke_ranking`
5. **Nút xem video** — seek đến `event_frame / fps` giây

**Trạng thái cần xử lý:**

| Trạng thái | Hiển thị |
|-----------|---------|
| Uploading | Spinner + "Đang tải lên..." |
| Queued | "Đang xếp hàng..." |
| Processing | Progress bar + text `step` |
| Done | Hiển thị kết quả |
| Error | Thông báo lỗi đỏ + nội dung `step` |

---

### Phase tiếp theo (M3 — timeline nhiều cú đánh)

UI đầy đủ cho rally dài:

1. **Video player** tích hợp với thanh timeline.
2. **Event list** — danh sách cú đánh bên cạnh video, click → seek.
3. **Tag màu sắc** — mỗi `stroke` một màu riêng trên timeline.
4. **Confidence badge** — hiển thị mức độ chắc chắn của từng nhãn.
5. **Filter** — lọc theo `player_side`, `stroke`, `stroke_side`.

---

## API v1 nâng cao (tích hợp sâu)

Dùng cho tích hợp backend-to-backend hoặc client nâng cao.

### `POST /v1/analyses`

**Form fields:**

| Field | Mặc định | Mô tả |
|-------|----------|-------|
| `video` | *(bắt buộc)* | File video |
| `technique` | `auto` | Kỹ thuật cú đánh (auto/forehand_clear/...) |
| `view` | `front` | Góc camera (`front` / `side`) |
| `handedness` | `right` | Tay thuận (`right` / `left`) |
| `target` | `single` | Người chơi mục tiêu |
| `floor_angle_degrees` | `0.0` | Góc nghiêng sàn (−45 đến 45) |
| `generate_preview` | `true` | Có tạo video preview không |
| `run_classifier` | `false` | Có chạy action classifier không |

**Response (202 Accepted):**

```json
{
  "id": "uuid-job-id",
  "status": "queued",
  "status_url": "http://localhost:8000/v1/analyses/uuid-job-id"
}
```

### `GET /v1/analyses/{job_id}`

**Response:**

```json
{
  "id": "uuid-job-id",
  "status": "succeeded",
  "created_at": "2026-09-19T10:00:00Z",
  "updated_at": "2026-09-19T10:02:30Z",
  "options": { "technique": "auto", "view": "front", ... },
  "input_filename": "match.mp4",
  "artifacts": ["analysis.json", "preview.mp4"],
  "result_url": "http://localhost:8000/v1/analyses/uuid-job-id/result",
  "artifact_urls": {
    "analysis.json": "http://localhost:8000/v1/analyses/uuid-job-id/artifacts/analysis.json",
    "preview.mp4":   "http://localhost:8000/v1/analyses/uuid-job-id/artifacts/preview.mp4"
  }
}
```

### `GET /v1/analyses/{job_id}/result`

Trả về JSON kết quả phân tích đầy đủ khi job đã `succeeded`.

### `GET /v1/analyses/{job_id}/artifacts/{name}`

Tải file artifact (video preview, ảnh, JSON phụ).

---

## Xử lý lỗi phía UI

| HTTP Status | Ý nghĩa | Xử lý UI |
|-------------|---------|----------|
| `400` | File không phải video | Thông báo định dạng không hợp lệ |
| `404` | Job không tồn tại | Redirect về trang chủ |
| `413` | File quá lớn (>500MB) | Thông báo giới hạn kích thước |
| `415` | Định dạng không hỗ trợ | Hiển thị danh sách định dạng hợp lệ |
| `422` | Classifier bị tắt | Thông báo tính năng không khả dụng |
| `500` | Lỗi server | Thông báo lỗi chung, cho phép thử lại |

---

## Lộ trình phát triển

### Milestone M1 ✅ — Phân loại một cú đánh (đang ở đây)

- Upload video ngắn (một rally, một cú đánh).
- Trả về: stroke, stroke_side, confidence, event_frame.
- UI: form upload + progress + thẻ kết quả.

### Milestone M1b — Forehand/backhand/aroundhead

- Thêm head phân loại hướng đánh chính xác hơn (multi-task model).
- Trả về thêm `stroke_side_confidence` riêng biệt.
- UI: hiển thị hai biểu đồ riêng (stroke + side).

### Milestone M2 — Phát hiện nhiều cú đánh (Hit Detection)

- Video rally dài → tự phát hiện nhiều điểm chạm vợt.
- Trả về danh sách `events[]` theo format StrokeEvent.
- UI: timeline có nhiều điểm, click để seek video.

### Milestone M3 — Video đầy đủ + Timeline web

- Upload một rally/trận nguyên → xuất timeline đầy đủ.
- Web viewer tích hợp video player + event list.
- Quality gate: từ chối video không hợp lệ (replay, camera nghiêng, đánh đôi...).

### Milestone M4 — Phân tích chiến thuật (tùy chọn)

- Thống kê chuỗi cú đánh.
- Tỷ lệ thắng theo pattern.
- Mỗi insight có link đến rally/frame tương ứng.

---

## Ghi chú quan trọng cho UI

- **Poll interval**: Không poll nhanh hơn 2 giây để tránh quá tải server.
- **Timeout**: Job có thể mất 2–5 phút. UI nên có timeout cảnh báo sau 10 phút.
- **court_image**: Chỉ có khi `court_auto_detected: true`. Nếu không có thì dùng fallback corners.
- **event_frame to seconds**: `giây = event_frame / fps`. FPS thường là 30 hoặc 60.
- **confidence**: Chưa được calibrate, dùng để so sánh tương đối, không hiển thị
  như xác suất tuyệt đối cho người dùng cuối.
- **unknown**: Một số job có thể trả `stroke: "unknown"` khi model không chắc.
  UI nên hiển thị khác biệt, không giống kết quả bình thường.

