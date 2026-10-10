# FastAPI architecture

## Mục tiêu

API biến pipeline phân tích video hiện có thành một contract ổn định cho frontend.
Xử lý AI không nằm trong router và không giữ request upload mở đến khi hoàn tất.

Luồng request:

```text
Frontend -> POST /api/v1/analyses -> 202 + job id
                                      |
                                      v
                            queue một worker (GPU)
                                      |
Frontend -> GET /api/v1/analyses/{id} +-- queued/processing/completed/failed
                                      |
Frontend -> GET /api/v1/analyses/{id}/result
```

## Tổ chức source

```text
src/ai_classifier/api/
├── main.py                 # app factory, lifespan, middleware, ghép router
├── dependencies.py         # dependency injection cho router
├── core/
│   └── config.py           # cấu hình từ environment
├── routers/
│   ├── health.py           # HTTP transport cho health check
│   └── analyses.py         # HTTP transport cho analysis jobs
├── schemas/
│   └── analysis.py         # request/response contract (Pydantic)
└── services/
    └── analysis_jobs.py    # use case, lưu job và gọi AI pipeline
```

Quy tắc phụ thuộc:

- `router` chỉ xử lý HTTP: parse input, status code và response.
- `schema` là contract chia sẻ với frontend, không chứa xử lý AI.
- `service` điều phối use case và có thể được gọi ngoài HTTP hoặc thay thế khi test.
- pipeline/domain hiện hữu không import FastAPI; vì vậy model vẫn chạy được từ CLI.
- `main.py` là composition root: nơi khởi tạo và nối các thành phần.

Không cần tạo repository/controller cho mọi thứ. Chỉ tách thêm tầng khi có database,
authentication hoặc nhiều implementation thật sự.

## Chạy local

```bash
uv sync --extra api --extra test
uv run uvicorn ai_classifier.api.main:app --reload
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`
- Health check: `http://127.0.0.1:8000/health`

Frontend local mặc định được phép chạy từ `http://localhost:3000` hoặc
`http://localhost:5173`. Có thể cấu hình bằng danh sách phân cách bởi dấu phẩy:

```powershell
$env:AI_CLASSIFIER_CORS_ORIGINS='https://app.example.com,http://localhost:5173'
```

## Contract cho frontend

Tạo job bằng `multipart/form-data`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyses \
  -F "video=@match.mp4" \
  -F "player_upper=Alice" \
  -F "player_lower=Bob"
```

Response `202 Accepted`:

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

Frontend polling `GET /api/v1/analyses/{id}` cho đến khi:

- `completed`: gọi URL trong `result_url`.
- `failed`: hiển thị trường `error`.
- `queued` hoặc `processing`: chờ 1-3 giây rồi gọi lại.

Ví dụ TypeScript tối giản:

```ts
const body = new FormData();
body.append("video", file);
body.append("player_upper", "Alice");
body.append("player_lower", "Bob");

const job = await fetch(`${API_URL}/api/v1/analyses`, {
  method: "POST",
  body,
}).then((response) => response.json());

while (true) {
  const current = await fetch(
    `${API_URL}/api/v1/analyses/${job.id}`,
  ).then((response) => response.json());
  if (current.status === "failed") throw new Error(current.error);
  if (current.status === "completed") {
    const result = await fetch(`${API_URL}${current.result_url}`).then(
      (response) => response.json(),
    );
    break;
  }
  await new Promise((resolve) => setTimeout(resolve, 2000));
}
```

## Khi đưa lên production

Implementation hiện tại phù hợp cho một API process và một GPU: metadata/result lưu
trên disk, executor có đúng một worker. Không chạy nhiều Uvicorn worker vì mỗi process
sẽ có queue riêng và có thể cùng chiếm GPU.

Khi cần scale, giữ nguyên router và schema, thay `AnalysisJobService` bằng:

- Redis/RabbitMQ + Celery/RQ/Dramatiq cho durable queue;
- PostgreSQL cho job metadata;
- S3/MinIO cho video và result;
- WebSocket hoặc Server-Sent Events nếu muốn thay polling;
- auth, rate limit và kiểm tra MIME/signature trước khi public Internet.

FastAPI `BackgroundTasks` hợp với tác vụ nhỏ chạy trong cùng process. Pipeline GPU dài
nên queue chuyên dụng là đích đến phù hợp khi triển khai nhiều máy.
