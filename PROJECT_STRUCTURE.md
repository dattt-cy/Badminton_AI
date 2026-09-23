
# HƯỚNG DẪN CẤU TRÚC MÃ NGUỒN (PROJECT ARCHITECTURE & CODEBASE GUIDE)

Tài liệu này cung cấp bản đồ kiến trúc toàn bộ mã nguồn của Đề tài **"Hệ thống Đa phương thức Nhận dạng Kỹ thuật Cầu lông (Multimodal Badminton Action Recognition)"** nhằm giúp Hội đồng và Giảng viên hướng dẫn dễ dàng theo dõi, kiểm tra và chạy thử nghiệm.

---

## 1. Sơ đồ Cấu trúc Tổng quan

```
AI_Classifier/
│
├── src/ai_classifier/                   # [CỐT LÕI] THƯ VIỆN SOURCE CODE CHÍNH CỦA DỰ ÁN
│   ├── models/                          # ──► 1. ĐỊNH NGHĨA CÁC MẠNG NƠ-RON
│   │   ├── fusion_head.py               # Mạng FusionHead (Kiến trúc đề xuất của nhóm)
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
├── train_fusion.py                      # [ENTRY POINT 1] HUẤN LUYỆN MẠNG FUSION TỪ CON SỐ 0
├── evaluate.py                          # [ENTRY POINT 2] ĐÁNH GIÁ CHUẨN XÁC TRÊN TẬP TEST SET
├── app.py                               # [ENTRY POINT 3] FLASK BACKEND CHO WEB DASHBOARD
├── badminton_analyzer.html              # GIAO DIỆN WEB DASHBOARD TƯƠNG TÁC
│
├── docs/                                # BÁO CÁO & BIỂU ĐỒ KHOA HỌC
│   └── figures/                         # 7 Biểu đồ chuẩn IEEE/CVPR (Loss, Confusion Matrix, Benchmark...)
│
├── models/checkpoints/                  # TRỌNG SỐ MÔ HÌNH PRETRAINED CƠ SỞ
│   └── pose/yolov8n-pose.pt             # Checkpoint trích xuất khớp xương
│
└── work_dirs/                           # NƠI LƯU TRỮ KẾT QUẢ HUẤN LUYỆN
    └── shuttleset_fusion_epoch20_full_b4/
        ├── best.pth                     # Trọng số tốt nhất của mạng Fusion (Epoch 17)
        ├── features.npz                 # Cache đặc trưng đa phương thức
        └── metrics.json                 # Lịch sử huấn luyện và đánh giá chi tiết
```

---

## 2. Chi tiết Các Module Trọng tâm

### A. Kiến trúc Mô hình (`src/ai_classifier/models/`)

- **`fusion_head.py` (`class FusionHead`)**:
    - Là **mô hình do nhóm tự thiết kế 100%**, khởi tạo ngẫu nhiên và huấn luyện từ đầu.
    - Hợp nhất 3 luồng đặc trưng: **Thị giác (RGB 512-dim)** + **Khớp xương & Vị trí sân** + **Quỹ đạo quả cầu**.
    - Chứa 2 đầu ra đa nhiệm (`stroke_head` cho 8 loại cú đánh, `side_head` cho 3 hướng đánh thuận/nghịch tay).
- **`multi_task_r2plus1d.py` (`class MultiTaskR2Plus1D`)**:
    - Đóng vai trò là **Feature Extractor (Xương sống thị giác)**, cắt bỏ lớp phân loại mặc định và thay thế bằng kiến trúc đa nhiệm.

### B. Xử lý Đặc trưng Động học (`src/ai_classifier/features/`)

- **`kinematics.py`**:
    - Căn chỉnh cửa sổ thời gian xung quanh điểm tiếp xúc cầu: $[-15, +30]$ frames.
    - Tính toán đạo hàm vận tốc bậc một ($\Delta \mathbf{x} = \mathbf{x}_t - \mathbf{x}_{t-1}$) và các chỉ số thống kê trung bình, độ lệch chuẩn.
- **`hitter_crop.py`**:
    - Module tự động nhận diện ai là người đánh cầu từ tọa độ 17 khớp xương của YOLOv8-Pose và crop khung hình người chơi, loại bỏ toàn bộ nhiễu bối cảnh sân thi đấu.

---

## 3. Hướng dẫn Chạy Thử nghiệm (Quick Start)

### 1. Đánh giá Mô hình trên Tập Test (Evaluation)

Để kiểm tra độ chính xác và xem bảng phân tích 8 loại cú đánh trên tập Test Set (871 mẫu ShuttleSet):

```bash
python evaluate.py
```

- **Kết quả đầu ra:** Top-1 Accuracy (**85.99%**), Top-2 Accuracy (**96.67%**), Macro-F1 (**80.26%**), và bảng Precision/Recall chi tiết từng lớp.

### 2. Huấn luyện lại Mạng Fusion (Training)

Để chạy lại quá trình huấn luyện 20 Epoch từ đầu bằng PyTorch:

```bash
python train_fusion.py --epochs 20 --batch-size 4 --learning-rate 1e-3
```

- Quá trình chạy sử dụng bộ tối ưu `AdamW` với Class-Weighted Cross Entropy, tự động lưu checkpoint tối ưu tại `work_dirs/shuttleset_fusion_epoch20_full_b4/best.pth`.

### 3. Chạy Giao diện Web Dashboard (Demo)

Khởi động máy chủ web Flask để phân tích video thực tế:

```bash
python app.py
```

- Mở trình duyệt tại địa chỉ: `http://127.0.0.1:5000` hoặc mở file `badminton_analyzer.html`.
