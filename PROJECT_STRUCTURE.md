
# HƯỚNG DẪN CẤU TRÚC MÃ NGUỒN (PROJECT ARCHITECTURE & CODEBASE GUIDE)

Tài liệu này cung cấp bản đồ kiến trúc toàn bộ mã nguồn của Đề tài **"Hệ thống Đa phương thức Nhận dạng Kỹ thuật Cầu lông (Multimodal Badminton Action Recognition)"** nhằm giúp Hội đồng và Giảng viên hướng dẫn dễ dàng theo dõi, kiểm tra và chạy thử nghiệm.

---

## 1. Sơ đồ Cấu trúc Tổng quan

```
AI_Classifier/
│
├── src/ai_classifier/                   # [CỐT LÕI] THƯ VIỆN SOURCE CODE CHÍNH CỦA DỰ ÁN
│   ├── models/                          # ──► 1. ĐỊNH NGHĨA CÁC MẠNG NƠ-RON
│   │   ├── cr_gated_fusion.py           # [ĐÓNG GÓP MỚI] Contact-Aware Reliability-Gated Fusion (TCN + Contact + Gate)
│   │   ├── fusion_head.py               # Mạng FusionHead Concat-MLP (Baseline đối chứng)
│   │   ├── multi_task_r2plus1d.py       # Mạng MultiTaskR2Plus1D (Backbone trích xuất thị giác)
│   │   └── __init__.py                  # Package export sạch sẽ
│   │
│   ├── features/                        # ──► 2. KỸ NGHỆ ĐẶC TRƯNG ĐA PHƯƠNG THỨC
│   │   ├── kinematics.py                # Căn chỉnh cửa sổ tiếp xúc cầu, đạo hàm vận tốc (diff)
│   │   ├── hitter_crop.py               # Dynamic Hitter Crop: Bóc tách người đánh theo Pose
│   │   └── __init__.py                  # Package export
│   │
│   ├── localization/                    # ──► 3. PHÁT HIỆN THỜI ĐIỂM ĐÁNH & BÁM CẦU
│   │   ├── fast_tracknet.py             # TrackNetV3 bám vết tọa độ quả cầu
│   │   └── hit_spotting.py              # Thuật toán phát hiện điểm đổi chiều vận tốc cầu
│   │
│   └── inference/                       # ──► 4. PIPELINE PHÂN TÍCH ĐẦU-CUỐI (END-TO-END)
│       └── ...                          # Các module ghép nối xử lý video
│
├── run_ablation_study.py                # [ENTRY POINT NCKH] CHẠY CHUỖI THÍ NGHIỆM ABLATION STUDY ĐỐI CHỨNG
├── train_cr_gated_fusion.py             # [ENTRY POINT 1] HUẤN LUYỆN MẠNG CR-GATED FUSION
├── train_fusion.py                      # [ENTRY POINT 2] HUẤN LUYỆN MẠNG CONCAT-MLP FUSION (BASELINE)
├── evaluate.py                          # [ENTRY POINT 3] ĐÁNH GIÁ CHUẨN XÁC TRÊN TẬP TEST SET
├── app.py                               # [ENTRY POINT 4] FLASK BACKEND CHO WEB DASHBOARD
├── badminton_analyzer.html              # GIAO DIỆN WEB DASHBOARD TƯƠNG TÁC
│
├── docs/                                # BÁO CÁO & BIỂU ĐỒ KHOA HỌC
│   └── figures/                         # 10 Biểu đồ chuẩn IEEE/CVPR (Ablation, Loss, Confusion Matrix, Benchmark...)
│
├── models/checkpoints/                  # TRỌNG SỐ MÔ HÌNH PRETRAINED CƠ SỞ
│   └── pose/yolov8n-pose.pt             # Checkpoint trích xuất khớp xương
│
└── work_dirs/                           # NƠI LƯU TRỮ KẾT QUẢ HUẤN LUYỆN
    ├── cr_gated_fusion/                 # KẾT QUẢ MÔ HÌNH CR-GATED VÀ ABLATION STUDY
    │   ├── sequence_features.npz        # Cache chuỗi thời gian 32 frames (237.7 MB)
    │   ├── ablation_summary.json        # Bảng tổng hợp đối chứng 4 biến thể
    │   ├── cr_gated_full/               # Checkpoint & metrics mô hình Full đề xuất
    │   ├── cr_gated_no_gate/            # Checkpoint & metrics mô hình Temporal + Contact
    │   ├── cr_gated_no_contact/         # Checkpoint & metrics mô hình không có Contact Embedding
    │   └── cr_gated_no_temporal/        # Checkpoint & metrics mô hình không có Temporal Encoder
    │
    └── shuttleset_fusion_epoch20_full_b4/
        ├── best.pth                     # Trọng số Baseline Concat-MLP (Epoch 17)
        ├── features.npz                 # Cache đặc trưng Concat-MLP
        └── metrics.json                 # Lịch sử huấn luyện baseline chi tiết
```

---

## 2. Chi tiết Các Module Trọng tâm

### A. Kiến trúc Mô hình (`src/ai_classifier/models/`)

- **`cr_gated_fusion.py` (`class CRGatedFusion`) - [ĐÓNG GÓP KHOA HỌC MỚI]**:
    - **Tách riêng luồng thời gian từng modality**: Không làm phẳng (flatten) sớm; áp dụng 1D Temporal Convolutional Networks (TCN) với Residual Connection riêng cho `pose` $[T, 68]$, `court` $[T, 4]$, `shuttle` $[T, 2]$.
    - **Contact-Aware Relative Alignment**: Nhúng khoảng cách thời gian tương đối so với frame chạm cầu $d_t = t - t_{\text{hit}}$ qua Sinusoidal/Linear Embedding để mô hình phân biệt rõ các giai đoạn: Chuẩn bị vung vợt $\rightarrow$ Chạm cầu $\rightarrow$ Follow-through.
    - **Reliability Gate Module**: Tự động ước lượng độ tin cậy $g_m = \text{softmax}(\text{MLP}([z_m, q_m]))$ dựa trên tỷ lệ mất cầu (TrackNet missing rate) và che khuất khớp xương (Pose low confidence), tự động hạ trọng số modality bị nhiễu.
    - **Cross-Modal Transformer**: Cho phép token `[FUSION]` học tương tác chú ý đa chiều giữa các phương thức trước khi phân loại.
- **`fusion_head.py` (`class FusionHead`)**:
    - Kiến trúc Concat-MLP Baseline (Epoch 20) dùng làm mốc so sánh thực nghiệm.
- **`multi_task_r2plus1d.py` (`class MultiTaskR2Plus1D`)**:
    - Xương sống thị giác R(2+1)D-18 trích xuất vector RGB 512 chiều từ video clip.

### B. Xử lý Đặc trưng Động học (`src/ai_classifier/features/`)

- **`kinematics.py`**:
    - Căn chỉnh cửa sổ thời gian xung quanh điểm tiếp xúc cầu: $[-15, +30]$ frames.
    - Tính toán đạo hàm vận tốc bậc một ($\Delta \mathbf{x} = \mathbf{x}_t - \mathbf{x}_{t-1}$) và các chỉ số thống kê trung bình, độ lệch chuẩn.
- **`hitter_crop.py`**:
    - Module tự động nhận diện ai là người đánh cầu từ tọa độ 17 khớp xương của YOLOv8-Pose và crop khung hình người chơi, loại bỏ toàn bộ nhiễu bối cảnh sân thi đấu.

---

## 3. Hướng dẫn Chạy Thử nghiệm (Quick Start)

### 1. Chạy Chuỗi Thí nghiệm Ablation Study Đối chứng (Khoa học & Thực nghiệm)

Để tái hiện lại bảng so sánh thực nghiệm 4 cấu hình và kiểm chứng độ cải thiện của cú Drive/Net-Attack trên 871 mẫu Test Set:

```bash
python run_ablation_study.py
```

- **Kết quả đầu ra:** So sánh trực tiếp Baseline Concat-MLP (85.99% Acc, 80.26% F1, 48.3% Drive) với mô hình mới **Temporal + Contact Alignment** (Acc: **87.49%**, Macro-F1: **82.90%**, Drive F1: **56.88%** - cải thiện vượt bậc **+8.58%**).
- Tự động xuất file kết quả JSON tại: `work_dirs/cr_gated_fusion/ablation_summary.json`.

### 2. Huấn luyện Đơn lẻ Mạng CR-Gated Fusion (Training)

Để huấn luyện một biến thể cụ thể của kiến trúc mới (ví dụ mô hình `full`, `no_gate`, `no_contact`, `no_temporal`):

```bash
python train_cr_gated_fusion.py --ablation full --epochs 25 --batch-size 128 --learning-rate 8e-4
```

### 3. Đánh giá Mô hình Concat-MLP Baseline (Evaluation)

Để kiểm tra độ chính xác của mạng Concat-MLP Epoch 20 trên tập Test Set (871 mẫu):

```bash
python evaluate.py
```

### 4. Tạo Toàn bộ Biểu đồ Báo cáo Khoa học (Figures Generation)

Để render lại toàn bộ các biểu đồ chất lượng cao chuẩn IEEE/CVPR phục vụ viết báo cáo và slide thuyết trình:

```bash
python scripts/generate_ablation_figures.py
python scripts/generate_paper_figures.py
python scripts/plot_loss_and_convergence.py
```

- Các file ảnh PNG độ phân giải 300 DPI được lưu tại `docs/figures/`.

### 5. Chạy Giao diện Web Dashboard (Demo)

Khởi động máy chủ web Flask để phân tích video thực tế:

```bash
python app.py
```

- Mở trình duyệt tại địa chỉ: `http://127.0.0.1:5000` hoặc mở trực tiếp file `badminton_analyzer.html`.

