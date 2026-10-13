# API response contract dành cho frontend

Tài liệu này mô tả dữ liệu frontend nhận từ Badminton AI Classifier API.

## 1. Quy ước chung

Base URL khi chạy local:

```text
http://127.0.0.1:8000
```

Các endpoint nghiệp vụ sử dụng prefix:

```text
/api/v1
```

Quy ước:

- Response dùng `application/json`, trừ request upload dùng `multipart/form-data`.
- Thời gian video có đơn vị giây.
- `confidence`, tỷ lệ và score chuẩn hóa nằm trong khoảng `0..1`, trừ khi field ghi rõ khác.
- Tọa độ sân `[x, y]` được chuẩn hóa trong khoảng `0..1`.
- `upper` và `lower` là vị trí người chơi trong video, không phải tên người chơi.
- Các giá trị tỷ số, rally winner và landing proxy là kết quả ước lượng.
- Endpoint đọc kết quả chỉ gọi sau khi job có `status = "completed"`.

## 2. Luồng sử dụng từ frontend

```text
1. POST /api/v1/analyses
2. Nhận job.id
3. Poll GET /api/v1/analyses/{job_id}
4. status=completed -> gọi các endpoint dữ liệu
5. status=failed -> hiển thị job.error
```

Các trạng thái job:

| Status | Ý nghĩa |
|---|---|
| `queued` | Đã nhận video, đang chờ GPU worker |
| `processing` | Pipeline đang phân tích video |
| `completed` | Đã có dữ liệu kết quả |
| `failed` | Pipeline lỗi; xem field `error` |

## 3. Health check

### `GET /health`

```json
{
  "status": "ok",
  "service": "badminton-ai-classifier"
}
```

## 4. Analysis job

### `POST /api/v1/analyses`

Request `multipart/form-data`:

| Field | Type | Bắt buộc | Mô tả |
|---|---|---|---|
| `video` | `File` | Có | Video trận đấu |
| `player_upper` | `string` | Không | Mặc định `Player Upper` |
| `player_lower` | `string` | Không | Mặc định `Player Lower` |

Response `202 Accepted`:

```json
{
  "id": "fb13f93b9c584eac934802c47845f607",
  "status": "queued",
  "filename": "match.mp4",
  "player_upper": "Nguyen Van A",
  "player_lower": "Tran Van B",
  "created_at": "2026-10-10T02:00:00Z",
  "updated_at": "2026-10-10T02:00:00Z",
  "error": null,
  "result_url": null
}
```

### `GET /api/v1/analyses/{job_id}`

Response khi đang xử lý:

```json
{
  "id": "fb13f93b9c584eac934802c47845f607",
  "status": "processing",
  "filename": "match.mp4",
  "player_upper": "Nguyen Van A",
  "player_lower": "Tran Van B",
  "created_at": "2026-10-10T02:00:00Z",
  "updated_at": "2026-10-10T02:00:02Z",
  "error": null,
  "result_url": null
}
```

Response khi hoàn thành:

```json
{
  "id": "fb13f93b9c584eac934802c47845f607",
  "status": "completed",
  "filename": "match.mp4",
  "player_upper": "Nguyen Van A",
  "player_lower": "Tran Van B",
  "created_at": "2026-10-10T02:00:00Z",
  "updated_at": "2026-10-10T02:05:00Z",
  "error": null,
  "result_url": "/api/v1/analyses/fb13f93b9c584eac934802c47845f607/result"
}
```

Response khi thất bại có `status = "failed"`, `result_url = null` và `error` chứa
thông báo lỗi.

### `DELETE /api/v1/analyses/{job_id}`

```json
{
  "id": "fb13f93b9c584eac934802c47845f607",
  "deleted": true
}
```

Không thể xóa job có trạng thái `processing`.

## 5. Video metadata

### `GET /api/v1/analyses/{job_id}/video`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "schema_version": "2.0_pbl6_coachai",
  "video": {
    "path": "C:/data/match.mp4",
    "filename": "match.mp4",
    "total_frames": 9000,
    "fps": 30.0,
    "duration_seconds": 300.0,
    "court_corners": [
      [620.0, 180.0],
      [1260.0, 180.0],
      [1500.0, 920.0],
      [350.0, 920.0]
    ]
  },
  "players": {
    "upper": "Nguyen Van A",
    "lower": "Tran Van B"
  }
}
```

Frontend không nên hiển thị `video.path` trực tiếp vì đây là đường dẫn nội bộ của backend.

## 6. Summary

### `GET /api/v1/analyses/{job_id}/summary`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "summary": {
    "hit_count": 74,
    "estimated_rally_count": 8,
    "hits_per_minute": 14.8,
    "mean_confidence": 0.8231,
    "stroke_counts": {
      "clear": 8,
      "drive": 10,
      "drop": 7,
      "lift": 14,
      "net_attack": 6,
      "net_shot": 12,
      "serve": 8,
      "smash": 9
    },
    "stroke_percentages": {
      "clear": 10.81,
      "drive": 13.51,
      "drop": 9.46,
      "lift": 18.92,
      "net_attack": 8.11,
      "net_shot": 16.22,
      "serve": 10.81,
      "smash": 12.16
    },
    "estimated_score": {
      "upper": 3,
      "lower": 5,
      "text": "3 - 5",
      "leader": "lower",
      "is_estimate": true,
      "method": "heuristic_rally_outcome",
      "complete_rallies_counted": 8
    }
  }
}
```

## 7. Reliability

### `GET /api/v1/analyses/{job_id}/reliability`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "reliability": {
    "score_is_estimated": true,
    "rally_winners_are_estimated": true,
    "partial_first_rally": false,
    "partial_last_rally": true,
    "low_confidence_threshold": 0.7,
    "low_confidence_hits": 6,
    "landing_is_2d_proxy": true,
    "suppressed_weak_duplicate_serves": 1
  }
}
```

Frontend nên dùng các cờ này để hiển thị nhãn “ước lượng” và cảnh báo dữ liệu.

## 8. Strokes

### `GET /api/v1/analyses/{job_id}/strokes`

Query parameters:

| Parameter | Type | Mặc định | Giới hạn |
|---|---|---|---|
| `page` | `number` | `1` | Tối thiểu 1 |
| `page_size` | `number` | `20` | Từ 1 đến 100 |
| `player` | `string` | `null` | `upper` hoặc `lower` |
| `stroke` | `string` | `null` | Ví dụ `smash`, `serve` |
| `rally` | `number` | `null` | Tối thiểu 1 |
| `min_confidence` | `number` | `0` | Từ 0 đến 1 |

Ví dụ:

```http
GET /api/v1/analyses/{job_id}/strokes?player=upper&stroke=smash&min_confidence=0.7&page=1&page_size=20
```

Response:

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "items": [
    {
      "hit_index": 15,
      "frame": 2140,
      "time_seconds": 71.333,
      "refined_frame": 2138,
      "contact_offset": -2,
      "hitter": "upper",
      "hitter_name": "Nguyen Van A",
      "hit_score": 0.9776,
      "stroke": "smash",
      "confidence": 0.9169,
      "top2": ["smash", "drive"],
      "tta_used": true,
      "tta_windows": 3,
      "crop_box": [596, 181, 842, 427],
      "analytics": {
        "player_position_court": [0.6392, 0.33742],
        "opponent_position_court": [0.48465, 0.68804],
        "contact_shuttle_projection_court": [0.80347, 0.26861],
        "contact_height_image_ratio": 0.53333,
        "hitting_area_3x3": 5,
        "landing_position_court_proxy": [0.95556, 0.06016],
        "landing_area_3x3_proxy": 3,
        "landing_proxy_frame": 2168,
        "landing_proxy_confidence": 0.62,
        "flight_time_to_next_hit_seconds": 0.9,
        "average_shuttle_speed_image_per_second": 0.37919,
        "average_wrist_speed_image_per_second": 0.3038,
        "shuttle_visibility_ratio": 1.0,
        "attack_state_rule": "attacking",
        "landing_is_proxy": true
      },
      "rally": 3,
      "ball_round": 4,
      "gap_from_previous_hit_seconds": 0.87
    }
  ],
  "page": 1,
  "page_size": 20,
  "total": 9,
  "total_pages": 1,
  "filters": {
    "player": "upper",
    "stroke": "smash",
    "rally": null,
    "min_confidence": 0.7
  }
}
```

Các field landing, vị trí hoặc tốc độ có thể là `null` nếu pipeline không thu đủ dữ liệu.
`tta_used` cho biết hit có cần chạy thêm cửa sổ temporal hay không;
`tta_windows` thường là `1` hoặc `3`.

Các event bị loại sau phân loại được giữ trong full result để audit:

```json
{
  "suppression_audit": {
    "weak_duplicate_serves": [
      {
        "frame": 2144,
        "hit_score": 0.4431,
        "replacement_frame": 2188,
        "reason": "weak_duplicate_serve_without_shuttle_evidence"
      }
    ]
  }
}
```

## 9. Rallies

### `GET /api/v1/analyses/{job_id}/rallies`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "items": [
    {
      "rally_id": 3,
      "start_time_seconds": 67.07,
      "end_time_seconds": 78.27,
      "duration_seconds": 11.2,
      "hits_count": 14,
      "server": "lower",
      "server_name": "Tran Van B",
      "winner": "upper",
      "winner_name": "Nguyen Van A",
      "win_reason": "opponent_forced_error",
      "score_upper": 1,
      "score_lower": 2,
      "score_text": "1 - 2",
      "stroke_sequence": ["serve", "net_attack", "drive", "lift", "smash"]
    }
  ],
  "total": 8
}
```

## 10. Scoreboard

### `GET /api/v1/analyses/{job_id}/scoreboard`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "scoreboard": {
    "estimated_score": {
      "upper": 3,
      "lower": 5,
      "text": "3 - 5",
      "leader": "lower",
      "is_estimate": true,
      "method": "heuristic_rally_outcome",
      "complete_rallies_counted": 8
    },
    "rallies": []
  }
}
```

`rallies` có cùng cấu trúc item như endpoint rallies. Đối với màn hình bảng điểm,
frontend chỉ nên gọi scoreboard; đối với danh sách rally, dùng endpoint rallies.

### `GET /api/v1/analyses/{job_id}/scoreboard-ocr`

Khi chưa cấu hình ROI:

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "scoreboard_ocr": {
    "enabled": false,
    "reason": "score_roi_not_configured"
  }
}
```

Khi OCR được bật, object có thể chứa ROI, các score event và thông tin frame tương ứng.

## 11. Tactical patterns

### `GET /api/v1/analyses/{job_id}/tactical-patterns`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "tactical_patterns": {
    "top_transitions_2gram": [
      {"sequence": "lift -> smash", "count": 7}
    ],
    "top_sequences_3gram": [
      {"sequence": "serve -> net_shot -> lift", "count": 3}
    ],
    "estimated_winning_combinations": {
      "upper": [],
      "lower": []
    },
    "per_player_transitions": {
      "upper": [],
      "lower": []
    }
  }
}
```

## 12. Structured tactics

### `GET /api/v1/analyses/{job_id}/structured-tactics`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "structured_tactics": {
    "method": "coachai_plus_rules_adapted",
    "source_attribution": "CoachAI+ published tactic definitions (KuangDW/CoachAI-Plus)",
    "limitations": ["Landing locations are monocular 2-D proxies."],
    "rallies": [
      {
        "rally_id": 1,
        "player": "upper",
        "player_name": "Nguyen Van A",
        "primary_tactic": "four_corner",
        "primary_tactic_label": "Four-Corner Movement",
        "detected_tactics": [],
        "shot_count": 5
      }
    ],
    "usage_by_player": {
      "upper": {},
      "lower": {}
    }
  }
}
```

## 13. Movement

### `GET /api/v1/analyses/{job_id}/movement`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "movement": {
    "method": "hit_time_position_lower_bound",
    "is_full_tracking_distance": false,
    "court_dimensions_meters": {"width": 6.1, "length": 13.4},
    "total_lower_bound_distance_meters": {"upper": 24.3, "lower": 21.8},
    "position_samples": {"upper": 35, "lower": 39},
    "rallies": [],
    "limitations": ["Distance is a lower bound based on positions at hit times"]
  }
}
```

Không ghi nhãn dữ liệu này là “tổng quãng đường thực tế”; đây là lower-bound estimate.

## 14. Court distribution

### `GET /api/v1/analyses/{job_id}/court-distribution`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "spatial_distribution_3x3": {
    "upper": [
      {"zone_id": 1, "name": "Front-Left", "count": 3, "percentage": 8.57},
      {"zone_id": 2, "name": "Front-Center", "count": 8, "percentage": 22.86}
    ],
    "lower": [
      {"zone_id": 1, "name": "Front-Left", "count": 4, "percentage": 10.26}
    ],
    "quality": {
      "missing_position_hits": {"upper": 1, "lower": 0},
      "uses_stroke_type_fallback": false,
      "player_centric_orientation": true
    }
  }
}
```

`zone_id` có giá trị từ 1 đến 9 trên lưới sân 3x3.

## 15. Player profiles

### `GET /api/v1/analyses/{job_id}/players`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "profiles": {
    "upper": {
      "player_name": "Nguyen Van A",
      "side": "upper",
      "total_shots": 35,
      "stroke_counts": {"smash": 8, "drop": 4},
      "radar_attributes": {},
      "playstyle": {},
      "radar_is_ability_score": false,
      "error_analysis": {}
    },
    "lower": {
      "player_name": "Tran Van B",
      "side": "lower",
      "total_shots": 39,
      "stroke_counts": {"smash": 6, "lift": 10},
      "radar_attributes": {},
      "playstyle": {},
      "radar_is_ability_score": false,
      "error_analysis": {}
    }
  }
}
```

## 16. Radar chart

### `GET /api/v1/analyses/{job_id}/charts/radar`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "radar_chart": {
    "axes": [
      "Attack Stroke Share",
      "Defensive Stroke Share",
      "Net Stroke Share",
      "Shot Variety",
      "Mean Model Confidence",
      "Fast Stroke Share"
    ],
    "upper": [69, 34, 86, 91, 86, 59],
    "lower": [43, 68, 92, 73, 87, 54]
  }
}
```

Giữ nguyên thứ tự của `axes`, `upper` và `lower` khi truyền vào chart library.
Đây là hồ sơ quan sát trong đoạn video, không phải điểm năng lực VĐV đã được
chuẩn hóa.

## 17. Response transition matrix

### `GET /api/v1/analyses/{job_id}/charts/transitions`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "response_transition_matrix": {
    "upper": {
      "serve": [{"response_stroke": "net_shot", "count": 4, "probability": 0.5, "percentage": 50.0}],
      "smash": [{"response_stroke": "lift", "count": 5, "probability": 0.625, "percentage": 62.5}]
    },
    "lower": {
      "serve": [{"response_stroke": "lift", "count": 3, "probability": 0.6, "percentage": 60.0}]
    }
  }
}
```

## 18. Coach insights

### `GET /api/v1/analyses/{job_id}/insights`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "items": [
    {
      "category": "attack",
      "target": "upper",
      "insight": "Player Upper creates pressure effectively after lifts."
    }
  ],
  "total": 1
}
```

## 19. Pipeline timing

### `GET /api/v1/analyses/{job_id}/timing`

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "timing_seconds": {
    "hit_scan": 45.2,
    "pose_and_classification": 108.6,
    "rally_segmentation": 0.02,
    "tactical_analysis": 0.08,
    "total": 153.9,
    "speed_factor_realtime": "1.95x RT"
  }
}
```

## 20. Full result

### `GET /api/v1/analyses/{job_id}/result`

Endpoint này trả toàn bộ output của pipeline:

```json
{
  "job_id": "fb13f93b9c584eac934802c47845f607",
  "data": {
    "schema_version": "2.0_pbl6_coachai",
    "video": {},
    "players": {},
    "summary": {},
    "reliability": {},
    "scoreboard": {},
    "tactical_patterns": {},
    "structured_tactics": {},
    "movement": {},
    "scoreboard_ocr": {},
    "suppression_audit": {},
    "spatial_distribution_3x3": {},
    "player_profiles": {},
    "radar_chart": {},
    "response_transition_matrix": {},
    "coach_insights": [],
    "predictions": [],
    "timing_seconds": {}
  }
}
```

Frontend dashboard nên dùng các endpoint nhỏ. Chỉ dùng `/result` khi cần tải hoặc
export toàn bộ dữ liệu.

## 21. Error responses

### Job không tồn tại — `404`

```json
{
  "detail": "Analysis job not found"
}
```

### Kết quả chưa sẵn sàng — `409`

```json
{
  "detail": "Analysis result is not ready (processing)"
}
```

### Video quá lớn — `413`

```json
{
  "detail": "Upload exceeds 2147483648 bytes"
}
```

### Dữ liệu request không hợp lệ — `422`

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["query", "page"],
      "msg": "Input should be greater than or equal to 1",
      "input": "0"
    }
  ]
}
```

## 22. TypeScript types tối thiểu

```ts
export type JobStatus = "queued" | "processing" | "completed" | "failed";

export interface AnalysisJob {
  id: string;
  status: JobStatus;
  filename: string;
  player_upper: string;
  player_lower: string;
  created_at: string;
  updated_at: string;
  error: string | null;
  result_url: string | null;
}

export interface StrokeItem {
  hit_index: number;
  frame: number;
  time_seconds: number;
  refined_frame?: number;
  contact_offset?: number;
  hitter: "upper" | "lower";
  hitter_name: string;
  hit_score?: number;
  stroke: string;
  confidence: number;
  top2?: string[];
  tta_used?: boolean;
  tta_windows?: number;
  crop_box?: [number, number, number, number];
  analytics: Record<string, unknown>;
  rally: number | null;
  ball_round?: number;
  gap_from_previous_hit_seconds?: number | null;
}

export interface PaginatedStrokes {
  job_id: string;
  items: StrokeItem[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  filters: {
    player: string | null;
    stroke: string | null;
    rally: number | null;
    min_confidence: number;
  };
}

export interface ApiError {
  detail: string | Array<Record<string, unknown>>;
}
```

Type đầy đủ có thể được generate tự động từ:

```text
GET /openapi.json
```

Swagger để frontend thử request trực tiếp:

```text
http://127.0.0.1:8000/docs
```
