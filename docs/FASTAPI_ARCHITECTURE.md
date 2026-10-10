# Kiến trúc FastAPI

Contract response và JSON mẫu dành cho frontend nằm tại
[`API_RESPONSE_CONTRACT.md`](API_RESPONSE_CONTRACT.md).

## 1. Mục tiêu

API cung cấp contract ổn định cho frontend mà không trộn HTTP với pipeline AI.
Video được xử lý theo mô hình job vì inference có thể chạy lâu và sử dụng GPU.

```text
Frontend -> POST /api/v1/analyses -> 202 + job_id
                                      |
                                      v
                              queue một GPU worker
                                      |
Frontend -> GET /api/v1/analyses/{job_id} -> trạng thái
                                      |
Frontend -> GET từng tài nguyên summary/strokes/rallies/...
```

## 2. Cấu trúc thư mục

Mỗi chức năng có router và response schema riêng. Router không xử lý nghiệp vụ;
service không phụ thuộc FastAPI.

```text
src/ai_classifier/api/
├── main.py                         # app factory, middleware, ghép router
├── dependencies.py                 # dependency injection
├── core/
│   ├── config.py                   # environment settings
│   └── exception_handlers.py       # HTTP error mapping tập trung
├── services/
│   ├── analysis_jobs.py            # tạo/xóa job, upload, queue pipeline
│   └── result_queries.py           # đọc, lọc, phân trang kết quả
├── routers/
│   ├── health.py                   # health check
│   ├── analyses.py                 # tạo, xem trạng thái, xóa job
│   ├── results.py                  # toàn bộ pipeline result
│   ├── video.py                    # metadata video
│   ├── summary.py                  # thống kê tổng quan
│   ├── reliability.py              # độ tin cậy và giới hạn dữ liệu
│   ├── strokes.py                  # cú đánh, filter, pagination
│   ├── rallies.py                  # danh sách rally
│   ├── scoreboard.py               # tỷ số ước lượng
│   ├── scoreboard_ocr.py           # kết quả đọc bảng điểm OCR
│   ├── tactical_patterns.py        # chuỗi và pattern chiến thuật
│   ├── structured_tactics.py       # chiến thuật CoachAI có cấu trúc
│   ├── movement.py                 # di chuyển người chơi
│   ├── court.py                    # phân bố trên sân 3x3
│   ├── players.py                  # hồ sơ người chơi
│   ├── radar.py                    # dữ liệu radar chart
│   ├── transitions.py              # ma trận phản hồi cú đánh
│   ├── insights.py                 # nhận xét coaching
│   └── timing.py                   # thời gian xử lý pipeline
└── schemas/
    ├── job.py
    ├── result.py
    ├── health.py
    ├── video.py
    ├── summary.py
    ├── reliability.py
    ├── stroke.py
    ├── rally.py
    ├── scoreboard.py
    ├── scoreboard_ocr.py
    ├── tactical_patterns.py
    ├── structured_tactics.py
    ├── movement.py
    ├── court.py
    ├── player.py
    ├── radar.py
    ├── transition.py
    ├── insight.py
    └── timing.py
```

Quy tắc phụ thuộc:

```text
router -> schema + service
service -> pipeline/domain
pipeline/domain -X-> FastAPI
main -> khởi tạo và ghép tất cả thành phần
```

## 3. Danh sách API

### Hệ thống và job

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/health` | Kiểm tra API còn hoạt động |
| `POST` | `/api/v1/analyses` | Upload video và tạo job |
| `GET` | `/api/v1/analyses/{job_id}` | Xem trạng thái job |
| `DELETE` | `/api/v1/analyses/{job_id}` | Xóa job không đang xử lý |
| `GET` | `/api/v1/analyses/{job_id}/result` | Lấy toàn bộ JSON gốc |

### Metadata và thống kê

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/api/v1/analyses/{job_id}/video` | Video, người chơi, schema version |
| `GET` | `/api/v1/analyses/{job_id}/summary` | Tổng số hit/rally và thống kê chung |
| `GET` | `/api/v1/analyses/{job_id}/reliability` | Cờ ước lượng và chất lượng dữ liệu |
| `GET` | `/api/v1/analyses/{job_id}/timing` | Thời gian từng bước pipeline |

### Cú đánh và rally

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/api/v1/analyses/{job_id}/strokes` | Danh sách cú đánh có pagination/filter |
| `GET` | `/api/v1/analyses/{job_id}/rallies` | Danh sách rally và người thắng |
| `GET` | `/api/v1/analyses/{job_id}/scoreboard` | Bảng điểm ước lượng |
| `GET` | `/api/v1/analyses/{job_id}/scoreboard-ocr` | Kết quả OCR bảng điểm |

### Phân tích chiến thuật

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/api/v1/analyses/{job_id}/tactical-patterns` | Transition, sequence, winning combination |
| `GET` | `/api/v1/analyses/{job_id}/structured-tactics` | Chiến thuật CoachAI có cấu trúc |
| `GET` | `/api/v1/analyses/{job_id}/movement` | Quãng đường và chuyển động |
| `GET` | `/api/v1/analyses/{job_id}/court-distribution` | Phân bố vị trí sân 3x3 |
| `GET` | `/api/v1/analyses/{job_id}/players` | Profile từng người chơi |
| `GET` | `/api/v1/analyses/{job_id}/insights` | Nhận xét dành cho huấn luyện |

### Dữ liệu biểu đồ

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/api/v1/analyses/{job_id}/charts/radar` | Dữ liệu radar chart |
| `GET` | `/api/v1/analyses/{job_id}/charts/transitions` | Ma trận phản hồi cú đánh |

## 4. Upload và theo dõi job

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyses \
  -F "video=@match.mp4" \
  -F "player_upper=Alice" \
  -F "player_lower=Bob"
```

API trả `202 Accepted`:

```json
{
  "id": "24f9...",
  "status": "queued",
  "filename": "match.mp4",
  "player_upper": "Alice",
  "player_lower": "Bob",
  "created_at": "2026-10-10T02:00:00Z",
  "updated_at": "2026-10-10T02:00:00Z",
  "error": null,
  "result_url": null
}
```

Frontend polling `GET /api/v1/analyses/{job_id}`. Khi `status=completed`, frontend
có thể gọi từng endpoint dữ liệu. Khi `status=failed`, hiển thị trường `error`.

## 5. Filter và pagination strokes

```http
GET /api/v1/analyses/{job_id}/strokes?page=1&page_size=20
GET /api/v1/analyses/{job_id}/strokes?player=upper
GET /api/v1/analyses/{job_id}/strokes?stroke=smash&rally=3
GET /api/v1/analyses/{job_id}/strokes?min_confidence=0.8
```

| Parameter | Giá trị |
|---|---|
| `page` | Số trang, bắt đầu từ 1 |
| `page_size` | 1-100 item |
| `player` | `upper` hoặc `lower` |
| `stroke` | Tên cú đánh chính xác |
| `rally` | Rally ID |
| `min_confidence` | Từ 0 đến 1 |

Response chứa `items`, `page`, `page_size`, `total`, `total_pages` và `filters`.

## 6. Chạy local

```bash
uv sync --extra api --extra test
uv run uvicorn ai_classifier.api.main:app --reload
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Health check: `http://127.0.0.1:8000/health`

CORS local mặc định cho phép `http://localhost:3000` và
`http://localhost:5173`. Cấu hình production:

```powershell
$env:AI_CLASSIFIER_CORS_ORIGINS='https://app.example.com,http://localhost:5173'
```

## 7. Production

Implementation hiện tại dành cho một API process và một GPU: metadata/result lưu
trên disk, executor có một worker. Không chạy nhiều Uvicorn worker vì mỗi process có
queue riêng và có thể cùng chiếm GPU.

Khi scale, giữ nguyên router/schema và thay service infrastructure:

- Redis/RabbitMQ cùng Celery, RQ hoặc Dramatiq cho durable queue;
- PostgreSQL cho job metadata;
- S3/MinIO cho video và result;
- WebSocket hoặc Server-Sent Events nếu muốn thay polling;
- authentication, rate limit và kiểm tra MIME/signature trước public Internet.
