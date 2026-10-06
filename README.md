# AI Classifier

Pipeline nhận dạng kỹ thuật cầu lông từ video bằng Robust KAP
(Kinematics-Aware Perception). Hệ thống kết hợp RGB, pose, vị trí sân, quỹ đạo
cầu và thời điểm tiếp xúc để phát hiện và phân loại cú đánh.

## Cấu trúc chính

- `scripts/inference/run_video_pipeline.py`: entrypoint duy nhất cho video mới.
- `scripts/inference/robust_aligned_video_pipeline.py`: trích đặc trưng và KAP.
- `scripts/training/`: huấn luyện hit detector và Robust KAP.
- `scripts/evaluation/`: đánh giá video có ground truth.
- `src/ai_classifier/`: model và các module dùng chung.
- `data/manifests/`: metadata phục vụ huấn luyện và đánh giá.

## Cài đặt

Yêu cầu Python 3.10 trở lên:

```bash
python -m pip install -e ".[test]"
```

## Chạy video mới

Pipeline tự quét hit, cache pose trong RAM và phân loại bằng direct Robust KAP:

```bash
python scripts/inference/run_video_pipeline.py \
  --video path/to/video.mp4 \
  --output outputs/result.json
```

Cấu hình mặc định ưu tiên accuracy: hit threshold `0.50`, YOLOv8x-pose,
contact refinement ±6 frame và không dùng reranker/specialist.

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
python -m pytest -q
```

Dataset, video, checkpoint và output lớn được giữ ngoài Git hoặc trong
`work_dirs/`; không xóa chúng khi dọn source code.
