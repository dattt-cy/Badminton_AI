# Bố cục slide hoàn chỉnh theo 3 phần

Deck được sắp xếp theo mạch: **bài toán → dữ liệu và phương pháp → thực nghiệm và kết quả**. Không ép giữ 19 slide; bản này dùng **22 slide** để tránh nhồi quá nhiều nội dung vào một trang.

> Lưu ý: không đưa các lần tự kiểm tra nội bộ trên video ngoài vào slide báo cáo. Chỉ trình bày kết quả có tập dữ liệu, cách chia và protocol đánh giá rõ ràng.

## Nội dung không sử dụng

- Góc khuỷu tay, biomechanics hoặc tư vấn lỗi kỹ thuật.
- RAG hybrid, ST-GCN++ hai lớp và kết quả 96,15%.
- Spring Boot/MySQL nếu báo cáo chỉ tập trung vào module AI.
- Không tuyên bố fusion đã sẵn sàng triển khai thực tế.

---

## Slide 1 — Trang bìa

**HỆ THỐNG ĐỊNH VỊ THỜI GIAN VÀ PHÂN LOẠI CÚ ĐÁNH TRONG VIDEO TRẬN ĐẤU CẦU LÔNG ĐƠN**

Phụ đề: `Định vị cú đánh · Phân loại loại cú · Nhận diện hướng đánh`.

## Slide 2 — Nội dung trình bày

### 1. Tổng quan hệ thống và nghiên cứu liên quan

- **1.1.** Bài toán, phạm vi và giá trị của hệ thống
- **1.2.** Luồng hoạt động tổng quát
- **1.3.** Phương pháp hiện đại và nghiên cứu liên quan

### 2. Dataset và phương pháp

- **2.1.** Dataset, hệ thống nhãn và chuẩn hóa dữ liệu
- **2.2.** Các thành phần AI trong hệ thống
- **2.3.** Phương pháp đề xuất và chiến lược huấn luyện

### 3. Thực nghiệm và kết quả

- **3.1.** Thiết lập và tiêu chí đánh giá
- **3.2.** Các thử nghiệm chính
- **3.3.** Kết quả, giới hạn và hướng phát triển

---

# PHẦN 1 — TỔNG QUAN HỆ THỐNG VÀ NGHIÊN CỨU LIÊN QUAN

## Slide 3 — Bài toán và động lực

**Đầu vào:** video broadcast trận cầu lông đơn.

**Đầu ra:** timeline gồm:

```text
thời điểm chạm cầu · người đánh · loại cú · hướng đánh · độ tin cậy
```

- Tự động trả lời ai đánh, đánh loại cú gì và tại thời điểm nào.
- Cho phép tra cứu từng cú đánh theo timeline.
- Tạo dữ liệu nền cho thống kê chiến thuật theo rally và trận đấu.

Ảnh: một frame trong [`dataset_examples.png`](slide_assets/dataset_examples.png).

## Slide 4 — Phạm vi và hệ thống nhãn

- Trận đơn, hai người chơi, camera cuối sân, video ngang từ 720p.
- Tám nhóm cú: `serve, clear, smash, drop, net_shot, lift, drive, net_attack`.
- Hướng đánh gồm: `forehand, backhand, aroundhead`.
- Nếu giao diện chỉ hiển thị hai nhóm, `aroundhead` được gộp vào `forehand`.
- Chưa đánh giá lỗi kỹ thuật cơ thể vì chưa có expert-labelled error dataset.

## Slide 5 — Luồng hoạt động tổng quát

```text
Video trận đấu
→ lọc đoạn không thi đấu
→ tìm thời điểm có cú đánh
→ xác định người đánh
→ crop người đánh
→ phân loại loại cú và hướng đánh
→ tạo timeline kết quả
```

Slide này chỉ mô tả trải nghiệm end-to-end; chưa trình bày tên model.

## Slide 6 — Đối tượng và giá trị

- **Người chơi:** xem lại từng cú và thống kê loại cú.
- **Huấn luyện viên:** phân tích pattern chiến thuật và đối chiếu video.
- **Nhà nghiên cứu:** khai thác stroke timeline, rally analytics và forecasting.


## Slide 7 — Các hướng tiếp cận hiện đại

| Hướng | Mô hình tiêu biểu | Ưu điểm | Hạn chế |
|---|---|---|---|
| Phân loại đoạn ngắn | R(2+1)D, SlowFast, VideoMAE | Học hình ảnh và chuyển động | Cần cắt đúng đoạn có cú đánh |
| Định vị theo thời gian | ActionFormer, TriDet | Xử lý video chưa cắt | Cần nhiều nhãn thời gian |
| Khung xương và đa nguồn | TemPose, BST | Kết hợp tư thế, sân và cầu | Phụ thuộc chất lượng pose/TrackNet |

**Lựa chọn của nhóm:** định vị cú đánh trước, sau đó phân loại đoạn video quanh sự kiện.

Ảnh: [`modern_methods.svg`](slide_assets/modern_methods.svg).

## Slide 8 — Nghiên cứu liên quan

| Nghiên cứu | Đầu vào | Ý nghĩa đối với đề tài |
|---|---|---|
| TemPose — CVPRW 2023 | Khung xương người chơi + vị trí cầu | Vị trí cầu hỗ trợ phân biệt các động tác gần giống nhau |
| BST — CVPRW 2026 | Khung xương hai người + vị trí sân + quỹ đạo cầu | Baseline đa dữ liệu gần nhất với hướng fusion |

- BST-CG báo cáo **70,20% accuracy** và **63,34% macro-F1** trên protocol 25 lớp.
- Nhóm tái chạy checkpoint tác giả trên 3.499 mẫu: **83,77% accuracy**, **82,09% macro-F1**.
- Không kết luận hơn/kém trực tiếp vì BST dùng 25 lớp, còn nhóm dùng tám lớp.

Ảnh: [`related_work_badminton.svg`](slide_assets/related_work_badminton.svg).

## Slide 9 — Khoảng trống và hướng giải quyết

- TemPose và BST tập trung phân loại khi sự kiện cú đánh đã được xác định.
- Video đầu vào thực tế chưa có sẵn hit frame và thông tin người đánh.
- Nhóm bổ sung chuỗi xử lý: phát hiện hit, chọn sự kiện, xác định người đánh rồi mới phân loại.
- Hệ thống dùng R(2+1)D-18 để phát hiện hit và trích đặc trưng RGB; learned selector, YOLOv8-Pose, FastTrackNet và fusion tạo thành pipeline phân tích đầy đủ.

**Câu chuyển phần:** để thực hiện hướng này, nhóm kết hợp Fine-Badminton và ShuttleSet, sau đó chuẩn hóa về cùng hệ thống nhãn.

---

# PHẦN 2 — DATASET VÀ PHƯƠNG PHÁP

## Slide 10 — Dataset sử dụng trong dự án

| Dataset | Quy mô | Vai trò |
|---|---:|---|
| Fine-Badminton | 10 trận, 9.817 đoạn có nhãn, 29 nhãn gốc | Bổ sung đa dạng hình ảnh và loại cú |
| ShuttleSet | 44 trận, 3.685 rally, 36.492 cú đánh | Cung cấp hit frame, người đánh, hướng đánh và vị trí sân |

- Hai nguồn được ánh xạ về cùng tám nhóm cú.
- Chia dữ liệu theo trận để tránh rò rỉ.
- Kinetics-400 chỉ cung cấp trọng số khởi tạo.

Ảnh: [`dataset_overview.svg`](slide_assets/dataset_overview.svg).

## Slide 11 — Chuẩn hóa dữ liệu và nhãn

```text
Nhãn gốc
→ ánh xạ 8 lớp
→ chia theo trận
→ tạo bảng chỉ mục dữ liệu
→ đọc 16 frame khi huấn luyện
```

- Fine-Badminton: 29 nhãn gốc → tám nhóm cú.
- ShuttleSet: 18 nhãn gốc → cùng tám nhóm cú.
- Loại cú và hướng đánh được giữ thành hai đầu ra độc lập.
- Bảng chỉ mục lưu đường dẫn video, nhãn và vị trí frame để chương trình đọc đúng 16 khung hình khi huấn luyện; không cần sao chép video thành hàng nghìn clip nhỏ.
- Một trận chỉ xuất hiện trong một tập train, validation hoặc test.

## Slide 12 — Kiến trúc tổng thể

```text
Gate
→ R(2+1)D-18 HIT detector
→ threshold + side-aware NMS
→ learned selector
→ YOLOv8-Pose theo dõi và crop người đánh
→ FastTrackNet theo dõi quỹ đạo cầu
→ R(2+1)D-18 RGB embedding + pose/court/shuttle features
→ Fusion classifier
→ physics refinement
→ timeline
```

- YOLOv8-Pose chạy quanh hit đã chọn, ghép vị trí người chơi qua nhiều frame để tạo crop ổn định.
- FastTrackNet chỉ chạy quanh sự kiện đã chọn để lấy quỹ đạo cầu, không quét toàn video ngay từ đầu.
- Fusion kết hợp đặc trưng RGB với pose, vị trí sân và quỹ đạo cầu để dự đoán loại cú và hướng đánh.
- Physics refinement kiểm tra và hiệu chỉnh kết quả bằng các đặc trưng chuyển động/quỹ đạo hợp lệ.
- Các module định vị, trích đặc trưng và phân loại không dùng chung trọng số.

Ảnh: [`pipeline_current.svg`](slide_assets/pipeline_current.svg).

## Slide 13 — Định vị cú đánh

### HIT detector

- Đầu vào: 16 frame toàn sân.
- Đầu ra: `no-hit / upper-hit / lower-hit`.
- Sinh các candidate theo thời gian và dự đoán phía người đánh.

### Learned selector

- Nhận HIT score và đặc trưng của peak.
- Giảm false positive và chọn candidate tốt hơn.
- Là mô hình học máy Random Forest, không phải bộ lọc luật cố định.
- Được huấn luyện trên tập phát triển riêng; tập kiểm tra chỉ dùng để báo cáo kết quả.

## Slide 14 — Theo dõi và crop người đánh

### YOLOv8-Pose

- Chỉ chạy quanh selected hit, không chạy toàn video.
- Ghép pose qua nhiều frame để ưu tiên vận động viên đang chuyển động.
- Loại các đối tượng đứng yên không phải người chơi.
- Tạo union crop ổn định cho classifier.

Pose ở bước này phục vụ theo dõi và crop, không thay thế bộ phân loại RGB.

## Slide 15 — Phân loại loại cú và hướng đánh

### R(2+1)D-18 RGB Multi-task Classifier

- Đầu vào: 16 frame RGB crop người đánh.
- Backbone RGB dùng chung.
- Head 1: tám lớp loại cú.
- Head 2: ba lớp `forehand/backhand/aroundhead`.
- Checkpoint chính: mixed epoch 20.

Mô hình được khởi tạo từ Kinetics-400 rồi huấn luyện với Fine-Badminton và ShuttleSet.

## Slide 16 — FastTrackNet và fusion đa dữ liệu

```text
RGB embedding
+ pose hai người
+ vị trí trên sân
+ quỹ đạo cầu
→ Fusion head
→ loại cú + hướng đánh
```

- FastTrackNet, phiên bản tích hợp tối ưu từ TrackNetV3, cung cấp quỹ đạo cầu quanh selected hit.
- Nhánh fusion kế thừa ý tưởng đa dữ liệu từ BST.
- Fusion được đánh giá offline trên feature ShuttleSet chuẩn hóa.
- Physics refinement được áp dụng sau fusion để kiểm tra tính hợp lý của kết quả theo chuyển động và quỹ đạo.

## Slide 17 — Chiến lược huấn luyện

- Lật ngang đoạn RGB.
- Dịch thời điểm lấy mẫu ±2 frame.
- Cân bằng hai nguồn dữ liệu và các lớp.
- Dùng focal loss để tập trung vào mẫu khó.
- Đóng băng backbone, sau đó mở `layer4`.
- Chọn checkpoint theo macro-F1 validation.
- Chỉ tăng cường dữ liệu trên tập train.

---

# PHẦN 3 — THỰC NGHIỆM VÀ KẾT QUẢ

## Slide 18 — Thiết lập dữ liệu thực nghiệm

### Bộ phân loại kết hợp

- Fine-Badminton: 5.921 mẫu train / 1.980 mẫu validation.
- ShuttleSet: 24.834 mẫu train / 3.998 mẫu validation trong lần chạy kết hợp.
- Tám lớp loại cú; ba lớp hướng đánh lấy từ ShuttleSet.

### Bộ phát hiện cú đánh

- 41.524 mẫu train / 2.625 mẫu validation.
- Train: 29.660 no-hit / 5.855 upper-hit / 6.009 lower-hit.
- 30 trận train / 5 trận validation.

Ảnh: [`dataset_scale.png`](slide_assets/dataset_scale.png).

## Slide 19 — Tiêu chí đánh giá

| Bài toán | Tiêu chí |
|---|---|
| Loại cú | Accuracy, balanced accuracy, macro-F1, top-2, confusion matrix |
| Hướng đánh | Accuracy, macro-F1, tỷ lệ hai đầu ra cùng đúng |
| Định vị hit | Precision, Recall, F1 tại ±2/±5/±15 frame, frame error, side accuracy |
| Toàn hệ thống | Precision, Recall, joint F1, thời gian xử lý |

```text
Joint đúng = thời điểm ∧ người đánh ∧ loại cú ∧ hướng đánh
```

Ảnh: [`evaluation_metrics.svg`](slide_assets/evaluation_metrics.svg).

## Slide 20 — Kết quả định vị hit

Đánh giá trên năm trận ShuttleSet validation với 3.998 ground-truth hit. Cấu hình: threshold 0,8, NMS radius 8 frame và side-aware NMS.

| Dung sai | Precision | Recall | F1 | Side accuracy | Mean frame error |
|---|---:|---:|---:|---:|---:|
| ±2 frame | 63,46% | 63,68% | 63,57% | 99,18% | 1,082 |
| ±5 frame | **85,49%** | **85,79%** | **85,64%** | **98,92%** | 1,738 |
| ±15 frame | 92,10% | 92,42% | 92,26% | 98,05% | 2,191 |

**Nhận xét:** tại dung sai ±5 frame, mô hình cân bằng tốt giữa độ chính xác thời gian và khả năng thu hồi sự kiện.

## Slide 21 — Kết quả phân loại và ablation

### Mixed epoch 20 trên validation kết hợp

| Đầu ra | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Loại cú — 8 lớp | **74,29%** | **72,16%** | **71,06%** |
| Hướng đánh — 3 lớp | **73,61%** | **76,54%** | **75,82%** |

### So sánh trực tiếp trên cùng 871 mẫu ShuttleSet test

| Mô hình | Stroke accuracy | Stroke macro-F1 | Side macro-F1 |
|---|---:|---:|---:|
| R(2+1)D-18 RGB — epoch 20 | 80,02% | 74,49% | 85,58% |
| Fusion epoch-20-matched | **85,99%** | **80,26%** | **95,02%** |

Fusion cải thiện **+5,97 điểm accuracy**, **+5,77 điểm stroke macro-F1** và **+9,44 điểm side macro-F1**.

Ảnh: [`rgb_fusion_direct_ablation.png`](slide_assets/rgb_fusion_direct_ablation.png).

## Slide 22 — Kết luận, giới hạn và hướng phát triển

### Kết luận

- Xây dựng được pipeline từ video đến timeline cú đánh.
- HIT detector đạt **F1 85,64% tại ±5 frame** trên ShuttleSet validation.
- R(2+1)D-18 RGB epoch 20 đạt **71,06% macro-F1** cho tám lớp trên validation kết hợp.
- Fusion cho kết quả tốt hơn RGB khi so sánh trên cùng 871 mẫu ShuttleSet.

### Giới hạn

- Chưa thể so sánh trực tiếp với BST do khác taxonomy và protocol.
- Fusion mới được xác nhận trên feature ShuttleSet chuẩn hóa.
- Các lớp gần nhau như drive/lift vẫn khó phân biệt.

### Hướng phát triển

- Chuẩn hóa BST về cùng tám lớp và cùng test set.
- Cải thiện chất lượng feature pose/court/shuttle.
- Tối ưu tốc độ xử lý toàn pipeline.

---

## Đối chiếu yêu cầu của giảng viên

| Yêu cầu | Slide đáp ứng |
|---|---|
| Các phương pháp học máy hiện đại | Slide 7 |
| Bài báo gần nhất liên quan trực tiếp | Slide 8 — BST |
| Khoảng trống nghiên cứu | Slide 9 |
| Phương pháp của nhóm | Slide 12–17 |
| Bộ tiêu chí đánh giá | Slide 19 |
| So sánh trực tiếp trên cùng dữ liệu | Slide 21 |
| Giới hạn và hướng phát triển | Slide 22 |

## Nguồn

1. Chang, J.-Y., *BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports*, CVPR Workshops 2026.
2. Ibh, M. et al., *TemPose: A New Skeleton-Based Transformer Model Designed for Fine-Grained Motion Recognition in Badminton*, CVPR Workshops 2023.
3. Wang et al., *ShuttleSet*, 2023.
4. Wang et al., *Fine-Badminton*, dataset release 2026.
5. Shi et al., *TriDet*, CVPR 2023.

---

# PHỤ LỤC — HƯỚNG DẪN SỬA DECK PDF HIỆN TẠI

Phần này dùng để sửa file `C:\Users\ADMIN\Downloads\Thị giác máy tính (4).pdf`. Đây là ghi chú dựng slide, **không đưa vào nội dung trình chiếu**.

## 1. Thứ tự slide sau khi sắp xếp lại

| Slide mới | Nội dung | Nguồn từ PDF hiện tại | Cách xử lý |
|---:|---|---:|---|
| 1 | Trang bìa | 1 | Giữ bố cục |
| 2 | Nội dung trình bày | 2 | Giữ thiết kế ba cột, cập nhật mục 1.1–3.3 |
| 3 | Bài toán và động lực | 3 | Giữ, rút gọn chữ |
| 4 | Phạm vi và hệ thống nhãn | 5 | Đưa lên trước luồng hệ thống |
| 5 | Luồng hoạt động tổng quát | 4 | Đổi tên “Kiến trúc 4 bước” |
| 6 | Đối tượng và giá trị | 6 | Giữ, bỏ nội dung dễ gây hiểu nhầm về biomechanics |
| 7 | Các hướng tiếp cận hiện đại | 10 | Chuyển lên Phần 1 |
| 8 | Nghiên cứu liên quan | 11 | Chuyển lên Phần 1 |
| 9 | Khoảng trống và hướng giải quyết | 12 | Rút gọn, nhấn mạnh điểm khác BST |
| 10 | Dataset sử dụng | 7 | Đưa xuống sau related work |
| 11 | Chuẩn hóa dữ liệu và nhãn | 13 | Giữ bố cục |
| 12 | Kiến trúc tổng thể | 8 | Bổ sung Gate, selector và tracking |
| 13 | Định vị cú đánh | 15 | Giữ kiểu đồ thị, sửa lại toàn bộ chữ |
| 14 | Theo dõi và crop người đánh | 9 | Chỉ tập trung vào YOLOv8-Pose |
| 15 | Phân loại loại cú và hướng đánh | Tách từ 8 | Tạo slide riêng cho classifier multi-task |
| 16 | Nhánh fusion đa dữ liệu | Tách từ 9 và 12 | Tạo slide riêng |
| 17 | Chiến lược huấn luyện | 14 | Giữ bố cục |
| 18 | Thiết lập dữ liệu thực nghiệm | Thay slide 17 | Dùng số liệu Fine-Badminton/ShuttleSet thực tế |
| 19 | Tiêu chí đánh giá | Thay slide 16 | Dùng metric HIT, stroke và stroke-side |
| 20 | Kết quả định vị hit | Mới | Thêm bảng kết quả tại ±2/±5/±15 frame |
| 21 | Kết quả phân loại và fusion | Thay slide 18–19 | Dùng kết quả epoch 20 và phép so sánh 871 mẫu |
| 22 | Kết luận, giới hạn và hướng phát triển | Mới | Tạo trang kết thúc |

Mạch trình bày sau khi sửa:

```text
Bài toán
→ Phạm vi
→ Luồng hệ thống
→ Phương pháp hiện đại và related work
→ Khoảng trống nghiên cứu
→ Dataset
→ Phương pháp của nhóm
→ Thiết lập thực nghiệm
→ Kết quả
→ Kết luận
```

## 2. Các slide có thể giữ thiết kế

- PDF slide 1: trang bìa.
- PDF slide 2: mục lục ba phần.
- PDF slide 3: vấn đề và động lực.
- PDF slide 4: minh họa luồng bốn bước.
- PDF slide 5: tám nhóm cú đánh.
- PDF slide 7: hai thẻ dataset.
- PDF slide 10: ba hướng phương pháp hiện đại.
- PDF slide 11: TemPose và BST.
- PDF slide 13: quy trình chuẩn hóa dữ liệu.
- PDF slide 14: minh họa Original–Flip–Shift.

Các trang trên chỉ cần đổi vị trí, rút chữ hoặc cập nhật số liệu; không cần thiết kế lại hoàn toàn.

## 3. Các slide cần sửa mạnh

### PDF slide 8 — Kiến trúc hệ thống

Thay pipeline cũ bằng:

```text
Gate
→ R(2+1)D-18 HIT detector
→ threshold + side-aware NMS
→ learned selector
→ YOLOv8-Pose tracking/crop
→ FastTrackNet
→ R(2+1)D-18 RGB features + pose/court/shuttle
→ fusion classifier
→ physics refinement
→ timeline
```

Giữ bảng hai model R(2+1)D-18, nhưng bổ sung các module xử lý nằm giữa detector và classifier.

### PDF slide 9 — Pose và TrackNet

- YOLOv8-Pose: theo dõi và crop người đánh quanh selected hit.
- FastTrackNet: cung cấp quỹ đạo cầu quanh selected hit cho fusion.
- Ghi rõ TrackNet không quét toàn video từ đầu; nó chạy sau learned selector để giảm chi phí.
- Nếu nội dung quá dày, tách YOLOv8-Pose và fusion thành hai slide.

### PDF slide 12 — Phương pháp đề xuất

Đổi vai trò thành **Khoảng trống nghiên cứu và đóng góp của nhóm**. Chỉ giữ ba ý:

1. Bổ sung hit detector cho video chưa có hit frame.
2. Dùng R(2+1)D-18 để trích đặc trưng RGB của đoạn quanh hit.
3. Kết hợp RGB, pose, vị trí sân và quỹ đạo cầu bằng fusion, sau đó physics refinement.

Không lặp lại toàn bộ pipeline vì kiến trúc chi tiết đã nằm ở slide 12 mới.

### PDF slide 15 — Định vị khung hình chạm cầu

- Giữ kiểu đồ thị xác suất theo thời gian.
- Trình bày đây là phương pháp sinh candidate và loại trùng.
- Không dùng đồ thị minh họa như bằng chứng kết quả.
- Tạo một slide kết quả riêng với bảng Precision, Recall và F1.

## 4. Các slide phải thay toàn bộ

### PDF slide 16 — Metric cũ

Bỏ nội dung YOLOv8-Pose mAP và ST-GCN sáu kỹ thuật. Thay bằng:

- **HIT:** Precision, Recall, F1 tại ±2/±5/±15 frame, side accuracy và frame error.
- **Stroke:** accuracy, balanced accuracy, macro-F1, top-2 và confusion matrix.
- **Stroke-side:** accuracy và macro-F1.
- **Toàn hệ thống:** tỷ lệ thời điểm, người đánh, loại cú và hướng đánh cùng đúng.

### PDF slide 17 — Dataset hai lớp cũ

Bỏ toàn bộ số liệu 953 clip và hai lớp Backhand Drive/Forehand Clear. Thay bằng:

- Fine-Badminton: 5.921 train / 1.980 validation.
- ShuttleSet: 24.834 train / 3.998 validation trong lần huấn luyện kết hợp.
- HIT detector: 41.524 train / 2.625 validation.
- HIT train: 29.660 no-hit / 5.855 upper-hit / 6.009 lower-hit.

### PDF slide 18–19 — ST-GCN++ và kết quả 96,15%

Bỏ toàn bộ:

- ST-GCN++.
- Epoch 23.
- 1.268 train / 312 validation.
- Bài toán hai lớp.
- Validation top-1 96,15%.

Thay bằng kết quả hiện tại:

| Thí nghiệm | Kết quả chính |
|---|---:|
| HIT detector tại ±5 frame | F1 85,64% |
| Mixed epoch 20 — stroke | Macro-F1 71,06% |
| Mixed epoch 20 — stroke-side | Macro-F1 75,82% |
| RGB trên cùng 871 mẫu | Stroke macro-F1 74,49% |
| Fusion trên cùng 871 mẫu | Stroke macro-F1 80,26% |
| RGB → Fusion | Side macro-F1 85,58% → 95,02% |

## 5. Nội dung cần bổ sung

### Slide mới — Khoảng trống nghiên cứu

```text
Related work phân loại trên sự kiện đã biết
→ video đầu vào chưa có hit frame
→ nhóm bổ sung định vị hit và xác định người đánh
```

### Slide mới — Bộ phân loại multi-task

- Đầu vào: 16 frame crop người đánh.
- Backbone: R(2+1)D-18.
- Head 1: tám lớp loại cú.
- Head 2: ba lớp `forehand/backhand/aroundhead`.
- Checkpoint chính: mixed epoch 20.

### Slide mới — FastTrackNet và nhánh fusion

```text
RGB embedding + pose + court + shuttle trajectory
→ Fusion head
→ stroke + stroke-side
```

Ghi rõ FastTrackNet chỉ chạy quanh hit đã chọn; fusion nhận đặc trưng R(2+1)D-18 RGB cùng pose/court/shuttle và được physics refinement sau dự đoán.

### Slide mới — Kết quả định vị hit

| Dung sai | Precision | Recall | F1 | Side accuracy |
|---|---:|---:|---:|---:|
| ±2 frame | 63,46% | 63,68% | 63,57% | 99,18% |
| ±5 frame | **85,49%** | **85,79%** | **85,64%** | **98,92%** |
| ±15 frame | 92,10% | 92,42% | 92,26% | 98,05% |

### Slide mới — Kết luận

- Pipeline xử lý từ video đến timeline cú đánh.
- HIT detector đạt F1 85,64% tại ±5 frame.
- Fusion cải thiện rõ so với RGB trên cùng 871 mẫu ShuttleSet.
- Chưa so sánh trực tiếp với BST vì khác taxonomy và protocol.
- Hướng tiếp theo: đưa BST về cùng tám lớp và cải thiện feature đa dữ liệu.

## 6. Lỗi chữ cần sửa trong PDF

- Slide 6: `PHÂN TÍCH CHUYẾN ĐỘNG` → bỏ hoặc sửa thành `PHÂN TÍCH CHUYỂN ĐỘNG`.
- Slide 7: `SHUTTIESET` → `ShuttleSet`.
- Slide 8: `người trên/dươi` → `người trên/dưới`.
- Slide 8: `Multi-taskClassifier` → `Multi-task Classifier`.
- Slide 9: `đánh đấu` → `đánh dấu`.
- Slide 9: `Quỹ đảo cầu` → `Quỹ đạo cầu`.
- Slide 10: `tư thể` → `tư thế`.
- Slide 14: rà lại các từ `chỉ`, `mỗi`, `tập` đang thiếu dấu.
- Slide 15: nhập lại toàn bộ các cụm đang sai như `thheo`, `ô1s`, `hít frame`, `loại củ`, `cửa số` và `cổ hì thạo`.

## 7. Nguyên tắc trình bày khi sửa

- Mỗi slide chỉ trả lời một câu hỏi chính.
- Không lặp toàn bộ pipeline ở nhiều trang.
- Mỗi bảng kết quả phải ghi rõ tập dữ liệu và protocol.
- Không so trực tiếp các con số dùng taxonomy hoặc test set khác nhau.
- Không đưa các lần tự kiểm tra nội bộ vào kết quả báo cáo.
- Dùng thống nhất các thuật ngữ: `HIT detector`, `learned selector`, `YOLOv8-Pose`, `FastTrackNet`, `fusion`, `physics refinement`, `stroke` và `stroke-side`.
