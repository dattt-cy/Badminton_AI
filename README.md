# AI Classifier

Pipeline nhận dạng kỹ thuật cầu lông từ video bằng Robust KAP
(Kinematics-Aware Perception). Hệ thống kết hợp RGB, pose, vị trí sân, quỹ đạo
cầu và thời điểm tiếp xúc để phát hiện và phân loại cú đánh.

## Cấu trúc chính

- `run_pipeline.py`: entrypoint end-to-end cho video mới và phân tích CoachAI.
- `scripts/inference/robust_aligned_video_pipeline.py`: trích đặc trưng và KAP.
- `scripts/training/`: huấn luyện hit detector và Robust KAP.
- `scripts/evaluation/`: đánh giá video có ground truth.
- `src/ai_classifier/`: model và các module dùng chung.
- `src/ai_classifier/analytics/match_analysis.py`: tổng hợp scoreboard ước lượng,
  pattern, profile và insight cấp trận.
- `src/ai_classifier/analytics/tactics/`: các bộ nhận diện chiến thuật có cấu
  trúc; hiện có bộ luật CoachAI+ được triển khai độc lập.
- `data/manifests/`: metadata phục vụ huấn luyện và đánh giá.

## Cài đặt

Yêu cầu Python 3.11 trở lên:

```bash
python -m pip install -e ".[test]"
```

## Chạy video mới

Pipeline tự quét hit, cache pose trong RAM và phân loại bằng direct Robust KAP:

```bash
python run_pipeline.py \
  --video path/to/video.mp4 \
  --output outputs/result.json
```

JSON đầu ra chứa các cú đánh, rally, đặc trưng chiến thuật và độ tin cậy.
`estimated_score` và người thắng rally là suy luận heuristic, không phải tỷ số
chính thức. Vị trí rơi cầu được ghi rõ là proxy 2D.

`structured_tactics` áp dụng độc lập các định nghĩa chiến thuật công bố bởi
CoachAI+ (Full-Court Pressure, Defensive Counterattack và Four-Corner). Do
CoachAI+ không công bố license phần mềm tại thời điểm tích hợp, dự án không sao
chép mã nguồn của họ; chỉ triển khai lại các định nghĩa đã công bố và giữ nguồn
tham khảo trong JSON.

API Python cấp cao cho phần phân tích là `build_match_analysis()`;
`build_tactical_analytics()` được giữ làm alias tương thích với code cũ.

`movement` báo quãng đường lower-bound từ các vị trí tại thời điểm đánh, không
phải tổng quãng đường tracking toàn video. Có thể bật OCR bảng điểm bằng ROI đã
hiệu chỉnh (tính năng tùy chọn):

```bash
python -m pip install -e ".[ocr]"
python run_pipeline.py --video match.mp4 --score-roi Y1 Y2 X1 X2
```

OCR được trả riêng trong `scoreboard_ocr` và chưa tự động ghi đè rally hoặc
người thắng cho đến khi ánh xạ hai cột điểm với người chơi được xác nhận.

Cấu hình mặc định ưu tiên accuracy: hit threshold `0.75`, NMS `8`, same-side
NMS `15`, YOLOv8x-pose,
contact refinement ±6 frame, KAP adaptation trên các trận online 13–17 được
chọn bằng balanced accuracy trên validation match41 và
không dùng reranker/specialist.

## Huấn luyện

```bash
python scripts/training/train_shuttleset_hit_rgb_full.py --help
python scripts/data/extract_sequence_features.py --help
python scripts/training/train_kap_scratch.py --help
```

## Đánh giá và kiểm tra

```bash
python scripts/evaluation/run_eval_robust.py --help
python scripts/evaluation/evaluate_kap_full_shuttleset_test.py --help
python scripts/evaluation/evaluate_coachai_output.py --help
python -m pytest -q
```

## FastAPI cho frontend

API bất đồng bộ theo mô hình job đã được đặt trong `src/ai_classifier/api/`:

```bash
uv sync --extra api --extra test
uv run uvicorn ai_classifier.api.main:app --reload
```

Mở `http://127.0.0.1:8000/docs` để thử API. Xem cấu trúc, contract và ví dụ
frontend tại [`docs/FASTAPI_ARCHITECTURE.md`](docs/FASTAPI_ARCHITECTURE.md).

Dataset, video, checkpoint và output lớn được giữ ngoài Git hoặc trong
`work_dirs/`; không xóa chúng khi dọn source code.
