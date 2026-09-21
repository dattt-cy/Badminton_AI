# Bản sửa bám đúng 19 slide PDF gốc

Nguồn: `C:\Users\ADMIN\Downloads\Thị giác máy tính (3).pdf`.

Mục tiêu của tài liệu này là để sửa trực tiếp trên deck hiện tại: giữ số slide, vai trò và nhịp trình bày cũ; chỉ thay nội dung sai hướng. Asset đã tạo nằm trong [`docs/slide_assets`](slide_assets/).

## Những nội dung phải bỏ

- Góc khuỷu tay/biomechanics, RAG hybrid và tư vấn lỗi kỹ thuật.
- Dataset tự thu hai lớp, ST-GCN++ hai lớp và con số 96,15%.
- Spring Boot/MySQL nếu thời lượng báo cáo chỉ tập trung module AI.
- Không tuyên bố fusion đã sẵn sàng production: fusion epoch-20-matched đã hoàn thành benchmark offline, nhưng adapter video ngoài còn domain gap giữa feature NPY sạch và feature pose/court/TrackNet tự trích.

## Đối chiếu yêu cầu của giảng viên

| Yêu cầu | Slide đáp ứng |
|---|---|
| Tóm tắt các phương pháp học máy hiện đại | Slide 10 |
| Bài báo gần nhất liên quan trực tiếp | Slide 11 — BST, CVPRW 2026 |
| Phương pháp nhóm đang làm và phần tùy chỉnh | Slide 12 và 18 |
| Bộ tiêu chí đánh giá tương tự nghiên cứu liên quan | Slide 11 và 16 |
| So sánh trực tiếp trên cùng tập dữ liệu | Slide 19 — RGB và mô hình kết hợp trên 871 mẫu |
| Trạng thái so sánh với bài báo | Slide 19 — chưa trực tiếp do khác số lớp/split; ghi rõ thí nghiệm cần bổ sung |
| Thành viên báo cáo luân phiên | Phân công ở cuối tài liệu |

---

## Slide 1 — Trang bìa (GIỮ BỐ CỤC)

**HỆ THỐNG ĐỊNH VỊ THỜI GIAN VÀ PHÂN LOẠI CÚ ĐÁNH TRONG VIDEO TRẬN ĐẤU CẦU LÔNG ĐƠN**

Phụ đề nhỏ: `Định vị cú đánh · Phân loại cú đánh · Nhận diện forehand/backhand`.

Ảnh nền: giữ ảnh sân cầu lông hiện tại.

## Slide 2 — Nội dung trình bày (GIỮ BỐ CỤC 3 PHẦN)

1. **Tổng quan hệ thống và nghiên cứu liên quan**
2. **Dataset và phương pháp**
3. **Thực nghiệm và kết quả**

## Slide 3 — Vấn đề và động lực (GIỮ, RÚT GỌN)

**Đầu vào:** video broadcast trận đơn.

**Đầu ra:** dòng thời gian `(khung hình chạm cầu, người đánh, loại cú, forehand/backhand, độ tin cậy)`.

- Tự động trả lời: ai đánh, đánh gì, forehand/backhand và tại thời điểm nào.
- Tra cứu từng cú theo mốc thời gian.
- Nền tảng cho thống kê chiến thuật theo rally/trận.

Ảnh: dùng một frame trong [`dataset_examples.png`](slide_assets/dataset_examples.png), crop một nửa làm hero image.

## Slide 4 — Luồng hoạt động của hệ thống (GIỮ BỐ CỤC 4 BƯỚC)

1. **Nhận video:** người dùng tải video trận đấu đơn.
2. **Tạo timeline:** hệ thống tìm các thời điểm có cú đánh.
3. **Phân tích sự kiện:** xác định người đánh, loại cú và forehand/backhand.
4. **Trả kết quả:** hiển thị danh sách cú đánh kèm mốc thời gian và độ tin cậy.

Kết quả của một sự kiện:

```text
khung hình chạm cầu · người đánh · loại cú · forehand/backhand · độ tin cậy
```

Slide này chỉ mô tả trải nghiệm end-to-end. Tên model và chi tiết kỹ thuật để ở slide 8, 9 và 12.

## Slide 5 — Phạm vi và hệ thống nhãn (GIỮ BỐ CỤC)

- Trận đơn, hai người chơi, camera cuối sân, video ngang ≥720p.
- 8 nhóm: `serve, clear, smash, drop, net_shot, lift, drive, net_attack`.
- Hướng đánh: `forehand, backhand`; aroundhead được gộp vào forehand khi chỉ hiển thị hai nhóm.
- Chưa đánh giá lỗi kỹ thuật cơ thể vì chưa có expert-labelled error dataset.

Ảnh: giữ tám biểu tượng/ô nhãn hiện tại; đổi mô tả “clear” thành **phông cầu cao và sâu**.

## Slide 6 — Đối tượng và giá trị (GIỮ BỐ CỤC)

- **Người chơi:** xem lại từng cú, thống kê loại cú.
- **HLV:** phân tích pattern chiến thuật và đối chiếu video.
- **Nhà nghiên cứu:** stroke timeline, rally analytics, forecasting.

Bỏ ô “phân tích chuyển động” nếu nó ngụ ý chấm lỗi biomechanics.

## Slide 7 — Các dataset sử dụng trong dự án

| Dataset | Quy mô | Vai trò |
|---|---:|---|
| Fine-Badminton | 10 trận, 9.817 đoạn có nhãn, 29 nhãn gốc | Học hình ảnh và chuyển động của loại cú |
| ShuttleSet | 44 trận, 3.685 rally, 36.492 cú đánh | Hit frame, người đánh, forehand/backhand, vị trí sân |

- Hai nguồn được ánh xạ về cùng 8 nhóm cú.
- Chia dữ liệu theo trận để tránh rò rỉ.
- Kinetics-400 chỉ cung cấp trọng số khởi tạo.
- Fine-Badminton bổ sung sự đa dạng về loại cú; ShuttleSet bổ sung thông tin thời điểm và ngữ cảnh trận đấu.

Ảnh: [`dataset_overview.svg`](slide_assets/dataset_overview.svg).

## Slide 8 — Hai mô hình R(2+1)D-18 trong hệ thống

| Thành phần | Đầu vào | Đầu ra | Vai trò |
|---|---|---|---|
| **R(2+1)D-18 Hit Detector** | 16 khung hình toàn sân | Không đánh / người trên / người dưới | Tìm thời điểm và người đánh |
| **R(2+1)D-18 Multi-task Classifier** | 16 khung hình crop người đánh | 8 loại cú + forehand/backhand/aroundhead | Phân loại cú đã định vị |

```text
Video toàn sân
 → R(2+1)D-18 Hit Detector
 → hit frame + người trên/dưới
 → crop người đánh
 → R(2+1)D-18 Multi-task Classifier
 → loại cú + hướng đánh
```

Hai model **chung kiến trúc nhưng không chung trọng số**:

- Hit detector được huấn luyện bằng nhãn `no_hit / upper_hit / lower_hit`.
- Classifier epoch 20 được huấn luyện bằng Fine-Badminton và ShuttleSet.
- Classifier dùng backbone chung rồi tách thành đầu ra loại cú và hướng đánh.

## Slide 9 — Công nghệ hỗ trợ (GIỮ BỐ CỤC TRACKNET/POSE)

### YOLOv8-Pose — dùng trong luồng xử lý nhanh

- Xác định hai người chơi và crop đúng vùng upper/lower.
- Không dùng khung xương để thay thế bộ phân loại RGB.

### TrackNetV3 — nhánh phân tích tùy chọn

- Theo dõi tọa độ cầu và hỗ trợ mô hình kết hợp nhiều loại dữ liệu.
- Không chạy toàn video mặc định vì chi phí lớn.
- Chỉ chạy quanh thời điểm nghi ngờ có cú đánh khi cần phân tích sâu.

Ảnh: giữ ảnh khung xương/quỹ đạo hiện có nhưng thêm nhãn **Nhánh phân tích tùy chọn** trên phần TrackNet.

## Slide 10 — Các phương pháp hiện đại cho video trận đấu

| Hướng | Mô hình tiêu biểu | Ưu điểm | Hạn chế |
|---|---|---|---|
| Phân loại đoạn ngắn | R(2+1)D, SlowFast, VideoMAE | Học hình ảnh và chuyển động | Phải cắt đúng cú đánh |
| Định vị theo thời gian | ActionFormer, TriDet | Xử lý video chưa cắt | Cần nhãn điểm đầu–cuối lớn |
| Khung xương, đa nguồn | TemPose, BST | Dùng tư thế, sân và quỹ đạo cầu | Phụ thuộc chất lượng pose/TrackNet |

**Lựa chọn của nhóm:** tìm thời điểm cú đánh trước, sau đó phân loại đoạn video quanh cú đánh. Cách tách hai bước tận dụng trực tiếp `hit_frame` của ShuttleSet và giúp xác định lỗi nằm ở định vị hay phân loại.

Ảnh: [`modern_methods.svg`](slide_assets/modern_methods.svg).

## Slide 11 — Nghiên cứu gần nhất và cách họ đánh giá

### TemPose — CVPRW 2023

- Khung xương người chơi + vị trí cầu.
- Cho thấy thông tin cầu giúp phân biệt các động tác gần giống nhau.

### BST — CVPRW 2026, baseline gần nhất

- Khung xương hai người + vị trí sân + quỹ đạo cầu.
- BST-CG: **70,20% độ chính xác, 63,34% F1 trung bình** trên protocol 25 lớp của paper.
- Nhóm tái chạy checkpoint tác giả trên 3.499 mẫu: **83,77% độ chính xác, 82,09% F1 trung bình**.

**Tiêu chí kế thừa:** độ chính xác, F1 trung bình, F1 lớp yếu nhất và top-2.

**Nhận xét:** BST chứng minh khung xương, vị trí sân và quỹ đạo cầu có giá trị bổ sung. Đây là cơ sở để nhóm xây dựng nhánh kết hợp dữ liệu bên cạnh mô hình RGB.

**Lưu ý:** BST dùng 25 lớp; nhóm dùng 8 lớp nên chưa kết luận hơn/kém trực tiếp.

Ảnh: [`related_work_badminton.svg`](slide_assets/related_work_badminton.svg).

## Slide 12 — Phương pháp đề xuất của nhóm

```text
Video → Tìm hit → Cắt người đánh → Phân loại 8 cú + forehand/backhand
```

1. **Tìm hit:** R(2+1)D-18 dự đoán không đánh/người trên/người dưới; loại dự đoán trùng theo thời gian.
2. **Cắt người:** YOLOv8-Pose lấy vùng người đánh trong cửa sổ ±1 giây.
3. **Phân loại:** mixed R(2+1)D-18 epoch 20 trả loại cú và forehand/backhand.
4. **Nhánh bổ sung:** kết hợp khung xương, vị trí sân và quỹ đạo cầu theo hướng BST.

**Khác BST:** nhóm bổ sung hit detector cho video chưa có hit frame, dùng RGB làm luồng chính và gom về 8 nhóm cú.

**Lý do lựa chọn:** RGB chạy nhanh và không phụ thuộc TrackNet ở chế độ mặc định; dữ liệu có cấu trúc chỉ được bổ sung khi cần tăng độ chính xác hoặc phân tích sâu.

Ảnh: [`pipeline_current.svg`](slide_assets/pipeline_current.svg).

## Slide 13 — Chuẩn hóa dữ liệu và nhãn

```text
Nhãn gốc → Ánh xạ 8 lớp → Chia theo trận → Đoạn video ảo → Bộ nhớ đệm tensor
```

- Fine-Badminton: 29 nhãn → 8 nhóm cú.
- ShuttleSet: 18 nhãn → cùng 8 nhóm cú.
- Loại cú và forehand/backhand được giữ thành hai đầu ra độc lập.
- Bảng dữ liệu chỉ lưu đường dẫn và mốc frame; video không bị sao chép thành hàng nghìn clip.
- Cùng một trận chỉ xuất hiện trong một tập dữ liệu.
- Khi huấn luyện, chương trình đọc 16 khung hình theo manifest và crop người đánh dựa trên nhãn upper/lower.

## Slide 14 — Tăng cường dữ liệu và chống học thuộc (GIỮ BỐ CỤC)

- Lật ngang đoạn RGB.
- Dịch thời điểm lấy mẫu ±2 khung hình.
- Cân bằng hai nguồn dữ liệu và các lớp.
- Tập trung học các mẫu khó bằng focal loss.
- Đóng băng backbone, sau đó mở layer4.
- Chia theo trận để tránh ghi nhớ sân và người chơi.

Các phép biến đổi chỉ áp dụng cho tập huấn luyện; tập kiểm định và kiểm thử giữ nguyên để phản ánh đúng khả năng tổng quát.

## Slide 15 — Định vị khung hình chạm cầu (GIỮ BỐ CỤC ĐỒ THỊ THỜI GIAN)

Thay “peak velocity cổ tay = impact frame” bằng:

```text
Cửa sổ RGB → xác suất có cú đánh → lọc ngưỡng → loại trùng → hit frame
```

- Ba lớp: không đánh, người trên đánh, người dưới đánh.
- Mỗi sự kiện có khung hình, người đánh và độ tin cậy.
- Lấy ±1 giây quanh sự kiện để phân loại loại cú.
- Việc loại trùng giúp nhiều cửa sổ liên tiếp quanh cùng một cú chỉ tạo ra một sự kiện trên timeline.

Đồ thị dùng trục Y là **xác suất có cú đánh**; đánh dấu các đỉnh còn lại sau khi loại trùng.

## Slide 16 — Bộ tiêu chí đánh giá của nhóm (GIỮ BỐ CỤC 4 Ô)

### 1. Phân loại loại cú

- Độ chính xác, độ chính xác cân bằng, F1 trung bình.
- Top-2, F1 lớp yếu nhất và ma trận nhầm lẫn.

### 2. Forehand/backhand

- Độ chính xác và F1 trung bình.
- Tỷ lệ loại cú và hướng đánh cùng đúng.

### 3. Định vị cú đánh

- Precision, Recall, F1 tại ±2, ±5, ±15 khung hình.
- Sai số khung hình và tỷ lệ đúng người trên/dưới.

### 4. Toàn hệ thống

```text
Đúng = thời điểm ∧ người đánh ∧ loại cú ∧ forehand/backhand
```

- Báo Precision, Recall, F1 toàn chuỗi và thời gian xử lý mỗi phút video.

Ảnh: [`evaluation_metrics.svg`](slide_assets/evaluation_metrics.svg).

---

# Ba slide cuối bắt buộc giữ đúng vai trò

## Slide 17 — DATASET HUẤN LUYỆN THỰC TẾ

### Bộ phân loại kết hợp

- Fine-Badminton: 5.921 mẫu huấn luyện / 1.980 mẫu kiểm định; tổng cộng 9.817 đoạn có nhãn.
- ShuttleSet: 24.834 mẫu huấn luyện / 3.998 mẫu kiểm định trong lần chạy kết hợp.
- Tám lớp loại cú; nhãn forehand/backhand lấy từ ShuttleSet.

### Bộ phát hiện cú đánh đầy đủ

- 41.524 mẫu huấn luyện / 2.625 mẫu kiểm định.
- Phân bố tập huấn luyện: 29.660 không đánh / 5.855 người trên đánh / 6.009 người dưới đánh.
- 30 trận huấn luyện / 5 trận kiểm định.

Ảnh: [`dataset_examples.png`](slide_assets/dataset_examples.png) và [`dataset_scale.png`](slide_assets/dataset_scale.png).

## Slide 18 — CÁC THỬ NGHIỆM VÀ QUÁ TRÌNH HUẤN LUYỆN

### Các thử nghiệm chính

| Thử nghiệm | Thay đổi | Kết quả |
|---|---|---:|
| Fine-only | Chỉ Fine-Badminton | F1 63,08%* |
| Mixed epoch 20 | Thêm ShuttleSet + hai đầu ra | F1 loại cú 71,06%* |
| RGB → fusion epoch-20-matched | Thêm pose, sân, cầu trên cùng 871 mẫu | F1 74,49% → 80,26% |

`*` Fine-only và Mixed dùng tập kiểm định khác nhau, không phải so sánh trực tiếp.

**Quy trình:** Fine-Badminton → huấn luyện kết hợp → crop người đánh → mở layer4 → chọn checkpoint theo F1 kiểm định.

Thử nghiệm RGB và kết hợp dữ liệu là phép so sánh công bằng nhất hiện có vì giữ nguyên tập test, taxonomy và cách chia dữ liệu.

Ảnh: [`epoch20_training_curve.png`](slide_assets/epoch20_training_curve.png).

## Slide 19 — KẾT QUẢ HUẤN LUYỆN

### Mô hình chính epoch 20 — tập kiểm định kết hợp

| Đầu ra | Độ chính xác | Độ chính xác cân bằng | F1 trung bình |
|---|---:|---:|---:|
| Loại cú — 8 lớp | **74,29%** | **72,16%** | **71,06%** |
| Hướng đánh — 3 lớp | **73,61%** | **76,54%** | **75,82%** |

### So sánh trực tiếp trên cùng 871 mẫu ShuttleSet

| Mô hình | Độ chính xác loại cú | F1 trung bình loại cú |
|---|---:|---:|
| RGB epoch 20 | 80,02% | 74,49% |
| Fusion epoch-20-matched | **85,99%** | **80,26%** |

**Cải thiện trực tiếp:** +5,97 điểm độ chính xác và +5,77 điểm F1 trung bình. Side macro-F1 tăng từ 85,58% lên 95,02%.

**Đối chiếu BST:** nhóm đã tái chạy baseline 25 lớp, nhưng chưa so trực tiếp với epoch 20 vì khác số lớp. Bước tiếp theo là quy về cùng 8 lớp và cùng mẫu kiểm thử.

Ảnh: [`epoch20_confusion_matrix.png`](slide_assets/epoch20_confusion_matrix.png) và [`rgb_fusion_direct_ablation.png`](slide_assets/rgb_fusion_direct_ablation.png).

## Nguồn

1. Chang, J.-Y., *BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports*, IEEE/CVF CVPR Workshops 2026, pp. 9889–9898. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/html/Chang_BST_Badminton_Stroke-type_Transformer_for_Skeleton-based_Action_Recognition_in_Racket_CVPRW_2026_paper.html).
2. Ibh, M., Grasshof, S., Witzner, D. và Madeleine, P., *TemPose: A New Skeleton-Based Transformer Model Designed for Fine-Grained Motion Recognition in Badminton*, IEEE/CVF CVPR Workshops 2023, pp. 5199–5208. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023W/CVSports/html/Ibh_TemPose_A_New_Skeleton-Based_Transformer_Model_Designed_for_Fine-Grained_Motion_CVPRW_2023_paper.html).
3. Wang et al., *ShuttleSet*, 2023.
4. Wang et al., *Fine-Badminton*, dataset release 2026.
5. Shi et al., *TriDet*, CVPR 2023.

## Phân công báo cáo luân phiên

- Lượt này: A báo cáo slide 1–9, B báo cáo slide 10–19.
- Lượt sau: đổi vai trò.
# CẬP NHẬT KẾT QUẢ 21/09/2026 — ƯU TIÊN ÁP DỤNG

Phần cập nhật này thay thế các nội dung tương ứng ở Slide 4, 8, 9, 12, 15, 18 và 19 bên dưới. Vẫn giữ tổng cộng 19 slide.

## Slide 4 — Luồng hoạt động cập nhật

```text
Video trận đấu
→ Gate LIVE / non-play
→ R(2+1)D-18 HIT detector
→ threshold + side-aware temporal NMS
→ learned HIT selector
→ motion tracking người đánh
→ RGB epoch 20
→ timeline cú đánh
```

- Gate loại close-up, chuyển cảnh và đoạn không thấy đủ sân/hai người.
- HIT detector sinh candidate theo thời gian và dự đoán người trên/dưới.
- Learned selector được huấn luyện trên 5 trận ShuttleSet validation, không dùng archive để train.
- Motion tracking dùng nhiều frame quanh hit để crop người đánh ổn định hơn.
- Fusion RGB + pose + vị trí sân + quỹ đạo cầu là nhánh nghiên cứu bổ sung, chưa thay luồng RGB mặc định trên video ngoài.

## Slide 8 — Các thành phần AI hiện tại

| Thành phần | Đầu vào | Đầu ra | Vai trò |
|---|---|---|---|
| Gate | Frame lấy mẫu | LIVE / non-play | Giảm đoạn không cần xử lý |
| R(2+1)D-18 HIT detector | 16 frame toàn sân | no-hit / upper / lower | Sinh hit candidates |
| Learned selector | HIT score và đặc trưng peak | Xác suất candidate đúng | Giảm false positive, chọn event |
| YOLOv8 motion tracking | Pose quanh selected hit | Trajectory/crop người đánh | Crop đúng người qua nhiều frame |
| RGB epoch 20 | 16 frame crop người đánh | 8 stroke + 3 stroke-side | Classifier production mặc định |
| Fusion head | RGB embedding + pose + court + shuttle | 8 stroke + 3 stroke-side | Nhánh đa dữ liệu thử nghiệm |

Các thành phần không chung trọng số. HIT/selector/tracking giải quyết định vị và người đánh; RGB/fusion giải quyết loại cú và forehand/backhand/aroundhead.

## Slide 9 — Pose và TrackNet trong production

### YOLOv8-Pose

- Chỉ chạy quanh selected HIT, không chạy toàn video.
- Ghép pose qua nhiều frame để ưu tiên vận động viên đang chuyển động và loại official đứng yên.
- Tạo union crop cho classifier RGB/fusion.

### TrackNetV3

- Chỉ cần cho nhánh fusion có quỹ đạo cầu.
- Cấu hình nhanh đã kiểm chứng: batch 64, non-overlap, cache CSV trajectory.
- TrackNet vẫn là bottleneck lớn hơn fusion MLP.
- Fusion offline dùng ShuttleSet NPY sạch; feature trích từ video ngoài còn domain gap nên chưa bật mặc định.

## Slide 12 — Phương pháp đề xuất cập nhật

### Luồng production mặc định

```text
Gate
→ HIT epoch 3
→ threshold 0,8 + side-aware NMS
→ learned selector
→ motion-track crop
→ RGB epoch 20
→ stroke + forehand/backhand
```

### Nhánh nghiên cứu đa dữ liệu

```text
RGB embedding
+ pose hai người
+ vị trí trên sân
+ quỹ đạo cầu
→ Fusion head
```

Khác BST: hệ thống bổ sung bước tìm hit cho video chưa có annotation, selector candidate và adapter video production; taxonomy đầu ra được gom về 8 nhóm cú.

## Slide 15 — Kết quả định vị hit

Đánh giá trên 5 trận ShuttleSet validation với 3.998 ground-truth hit. Cấu hình chọn trên validation: threshold 0,8, NMS radius 8 frame, side-aware NMS.

| Dung sai | Precision | Recall | F1 | Side accuracy | Mean frame error |
|---|---:|---:|---:|---:|---:|
| ±2 frame | 63,46% | 63,68% | 63,57% | 99,18% | 1,082 |
| ±5 frame | **85,49%** | **85,79%** | **85,64%** | **98,92%** | 1,738 |
| ±15 frame | 92,10% | 92,42% | 92,26% | 98,05% | 2,191 |

Kết luận: khi HIT ghép đúng event, xác định upper/lower rất chính xác; learned selector tiếp tục giảm candidate giả trước khi chạy pose/classifier.

## Slide 18 — Thử nghiệm chính cập nhật

| Thử nghiệm | Thay đổi | Kết quả chính |
|---|---|---:|
| Fine-only | Chỉ Fine-Badminton | Macro-F1 stroke 63,08%* |
| Mixed epoch 20 | Fine-Badminton + ShuttleSet, hai đầu ra | Macro-F1 stroke 71,06%* |
| HIT epoch 3 | Dense scan + threshold/NMS | F1@±5 = 85,64% |
| HIT + selector + tracking | Test archive khác dữ liệu train | Joint 46,7–55,2% tùy bộ 30 clip |
| Fusion epoch-20-matched | RGB + pose + court + shuttle, cùng 871 mẫu | Stroke F1 74,49% → 80,26% |

`*` Fine-only và Mixed epoch 20 không dùng cùng tập kiểm thử nên không phải so sánh trực tiếp.

Các thử nghiệm đã loại:

- Fine-tune class-weight/drive-clean: tăng nhẹ ShuttleSet nhưng không tăng archive.
- Confuser linear head: validation tăng rất nhỏ, archive không tăng.
- RGB 16×160 và 24×160 pilot: giảm kết quả archive.
- Quality selector thủ công: side tăng nhưng stroke giảm; learned selector tốt hơn.

## Slide 19 — Kết quả chính và giới hạn

### Epoch 20 trên validation hỗn hợp

| Đầu ra | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Stroke — 8 lớp | 74,29% | 72,16% | 71,06% |
| Stroke-side — 3 lớp | 73,61% | 76,54% | 75,82% |

### So sánh trực tiếp trên cùng 871 mẫu ShuttleSet test

| Mô hình | Stroke accuracy | Stroke macro-F1 | Side macro-F1 |
|---|---:|---:|---:|
| RGB epoch 20 | 80,02% | 74,49% | 85,58% |
| Fusion epoch-20-matched | **85,99%** | **80,26%** | **95,02%** |

Cải thiện trực tiếp của fusion trên ShuttleSet test:

- Stroke accuracy: **+5,97 điểm %**.
- Stroke macro-F1: **+5,77 điểm %**.
- Side macro-F1: **+9,44 điểm %**.

### Kiểm tra video archive ngoài tập train

Pipeline RGB + learned selector + motion tracking:

- Bộ 30 clip thứ nhất: stroke 58,6%, side 93,1%, joint 55,2%.
- Bộ 30 clip khác: stroke 56,7%, side 76,7%, joint 46,7%.
- Kết quả dao động cho thấy drive/lift và góc người xa camera vẫn là nút thắt.

Fusion production trên 30 archive chỉ đạt stroke 40,0%, side 83,3%, joint 30,0% trong lần kiểm tra adapter hiện tại. Vì vậy fusion mới chỉ được kết luận tốt **offline trên feature ShuttleSet chuẩn hóa**, chưa thay RGB trong luồng upload.

### Kết luận trung thực

- HIT + selector + tracking đã tạo pipeline end-to-end chạy được trên video ngoài.
- Fusion chứng minh pose/court/shuttle bổ sung tín hiệu mạnh trên ShuttleSet.
- Khoảng cách giữa feature NPY sạch và feature trích từ production là giới hạn chính cần xử lý tiếp.
- Không dùng các số fusion cũ `68,96% → 81,72%` làm kết quả mới; thay bằng phép so sánh epoch-20-matched ở bảng trên.

---
