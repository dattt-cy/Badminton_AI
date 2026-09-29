# AI Classifier

Pipeline nhận dạng kỹ thuật cầu lông từ video bằng mô hình KAP
(Kinematics-Aware Perception). Hệ thống kết hợp RGB, pose, quỹ đạo cầu và
thông tin thời điểm tiếp xúc để phân loại stroke và phía đánh.

## Thành phần chính

- `src/ai_classifier/`: model KAP, feature động học, pose và hit localization.
- `scripts/inference/`: pipeline suy luận video đã căn chỉnh và bản robust.
- `scripts/training/`: huấn luyện KAP, pair specialist và top-2 reranker.
- `scripts/evaluation/`: đánh giá trên ShuttleSet và video thực tế.
- `scripts/data/`: tạo manifest, cache và feature phục vụ huấn luyện.
- `configs/`: cấu hình pose, taxonomy, dataset và biomechanics.
- `data/manifests/`: metadata cần thiết để tái lập thí nghiệm; video/dataset
  gốc không được lưu trong Git.
- `models/`: chỉ giữ cấu trúc thư mục; checkpoint được lưu cục bộ.

## Cài đặt

Yêu cầu Python 3.10 trở lên:

```bash
python -m pip install -e ".[test]"
```

Checkpoint, dataset, output và thư mục thí nghiệm đều được `.gitignore` loại
khỏi repository.

## Kiểm tra

```bash
python -m pytest -q
```

## Huấn luyện KAP

Trích xuất sequence feature rồi huấn luyện mô hình:

```bash
python scripts/data/extract_sequence_features.py
python scripts/training/train_kap_scratch.py
```

Các bước reranking tùy chọn:

```bash
python scripts/training/train_pair_specialists.py
python scripts/training/train_top2_reranker.py
```

## Suy luận và đánh giá

Xem tham số cụ thể bằng `--help`:

```bash
python scripts/inference/robust_aligned_video_pipeline.py --help
python scripts/evaluation/run_eval_robust.py --help
python scripts/evaluation/evaluate_kap_full_shuttleset_test.py --help
```

Các file lớn cần giữ ngoài Git:

- dataset và video trong `data/`;
- checkpoint `.pt`, `.pth`, `.onnx` trong `models/` hoặc `work_dirs/`;
- kết quả inference trong `outputs/`;
- báo cáo và paper cục bộ trong `docs/`.
