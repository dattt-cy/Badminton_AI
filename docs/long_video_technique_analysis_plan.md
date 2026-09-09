# Kế hoạch phân tích kỹ thuật cầu lông từ video dài

## 1. Mục tiêu

Xây dựng pipeline nhận video cầu lông dài khoảng 1 phút, tự động:

1. Theo dõi người chơi và trích xuất bộ xương.
2. Phát hiện các đoạn có khả năng chứa một cú đánh.
3. Phân loại từng cú đánh thành `backhand_drive`, `forehand_lift` hoặc `background`.
4. Tính đặc trưng hình học cho từng cú đánh.
5. So sánh chuyển động với tập mẫu chuẩn cùng loại kỹ thuật.
6. Chỉ ra đoạn thời gian, khớp và đặc trưng có độ lệch lớn.

Hệ thống không train một mô hình AI để phán trực tiếp động tác đúng hay sai. AI được dùng để trích xuất pose và nhận diện loại hành động; kết quả kỹ thuật được đánh giá bằng phân tích sinh cơ học có khả năng giải thích.

## 2. Phạm vi MVP

- Video đầu vào tối đa khoảng 1 phút.
- Một người chơi mục tiêu.
- Camera cố định, thấy rõ toàn thân.
- Người chơi thuận tay phải.
- Hai kỹ thuật hiện có:
    - `backhand_drive`
    - `forehand_lift`
- Mỗi kỹ thuật chỉ đánh giá 2–3 lỗi quan sát được bằng COCO-17.
- Kết quả được mô tả là "độ lệch so với mẫu tham chiếu", không phải kết luận chuyên môn tuyệt đối.
- Không đánh giá góc mặt vợt, cách cầm vợt hoặc thời điểm chạm cầu chính xác vì YOLOv8 Pose không phát hiện vợt và quả cầu.

## 3. Kiến trúc tổng thể

```text
Video dài
    |
    v
YOLOv8 Pose + tracking + smoothing
    |
    v
Motion Proposal
(vận tốc cổ tay + thay đổi góc tay)
    |
    v
Các đoạn ứng viên
    |
    v
ST-GCN++ 3 lớp
├── background
├── backhand_drive
└── forehand_lift
    |
    v
Tinh chỉnh biên bắt đầu/kết thúc từng cú
    |
    v
Trích xuất đặc trưng sinh cơ học
    |
    v
DTW căn chỉnh với tập mẫu chuẩn
    |
    v
Reference Motion Envelope
(median + MAD/percentile theo thời gian)
    |
    v
Phát hiện và gom các frame lệch liên tiếp
    |
    v
JSON kết quả + video trực quan hóa + timeline
```

## 4. Phân chia trách nhiệm AI và hình học

| Thành phần                   | Phương pháp                |   Cần train? |
| ---------------------------- | -------------------------- | -----------: |
| Trích xuất khớp              | YOLOv8 Pose                | Model có sẵn |
| Nhận diện loại cú đánh       | ST-GCN++                   |           Có |
| Nhận diện `background`       | Thêm lớp cho ST-GCN++      |           Có |
| Tạo đoạn ứng viên            | Xử lý tín hiệu chuyển động |        Không |
| Tính góc/khoảng cách/vận tốc | Hình học vector + NumPy    |        Không |
| Căn chỉnh cú nhanh/chậm      | Dynamic Time Warping       |        Không |
| Xây vùng tham chiếu          | Median + MAD/percentile    |        Không |
| Đánh giá độ lệch             | Bộ tiêu chí sinh cơ học    |        Không |
| Phát hiện đúng/sai bằng ML   | Không thực hiện trong MVP  |        Không |

## 5. Giai đoạn 1 — Chuẩn hóa dữ liệu đầu vào

### 5.1. Tận dụng pipeline hiện tại

Tiếp tục sử dụng:

- `YOLOv8PoseEstimator` để lấy `(T, 17, 3)`.
- `KeypointSmoother` để nội suy gap ngắn và giảm rung.
- `PoseSequence` để lưu keypoint, FPS và kích thước video.

### 5.2. Bổ sung kiểm soát chất lượng

Với mỗi frame, tính:

- Tỷ lệ khớp hợp lệ.
- Confidence trung bình.
- Confidence của vai, khuỷu và cổ tay tay đánh.

Không sử dụng frame để đánh giá nếu các khớp liên quan có confidence thấp hơn ngưỡng cấu hình.

### 5.3. Offline và realtime

Pipeline MVP xử lý video sau khi upload nên có thể dùng bộ lọc hai chiều. Nếu triển khai realtime sau này, cần thay bằng EMA hoặc One Euro Filter một chiều và chấp nhận độ trễ cửa sổ.

## 6. Giai đoạn 2 — Phát hiện đoạn ứng viên trong video dài

Không đưa toàn bộ video 1 phút vào ST-GCN++ một lần.

### 6.1. Tạo tín hiệu chuyển động

Từ pose đã chuẩn hóa, tính theo từng frame:

- Vận tốc cổ tay trái/phải.
- Gia tốc cổ tay.
- Vận tốc góc khuỷu.
- Vận tốc góc vai.
- Mức thay đổi tổng hợp của phần thân trên.

Chuẩn hóa vận tốc theo FPS và kích thước cơ thể để các video có độ phân giải khác nhau vẫn so sánh được.

### 6.2. Tạo motion score

Ví dụ:

```text
motion_score[t]
  = w1 * wrist_speed[t]
  + w2 * elbow_angular_speed[t]
  + w3 * shoulder_angular_speed[t]
```

Ngưỡng nên lấy theo phân bố của chính video, ví dụ percentile, thay vì dùng một ngưỡng pixel cố định.

### 6.3. Gom frame thành đoạn ứng viên

- Đánh dấu frame có `motion_score` cao.
- Gom các frame gần nhau thành một đoạn.
- Bỏ đoạn quá ngắn.
- Gộp hai đoạn nếu khoảng nghỉ giữa chúng nhỏ.
- Thêm padding trước và sau khoảng 0,3–0,5 giây.
- Giới hạn độ dài đoạn ứng viên phù hợp với một cú đánh.

Đầu ra dự kiến:

```json
[
    { "start_frame": 132, "end_frame": 201, "peak_frame": 171 },
    { "start_frame": 488, "end_frame": 562, "peak_frame": 527 }
]
```

Motion Proposal chỉ tạo ứng viên, không tự kết luận đó là cú đánh.

## 7. Giai đoạn 3 — ST-GCN++ xác nhận và phân loại cú đánh

### 7.1. Vấn đề của model hiện tại

Model hiện chỉ có hai lớp. Khi đầu vào là đứng chờ, đi bộ hoặc nhặt cầu, model vẫn bị buộc chọn `backhand_drive` hoặc `forehand_lift`.

### 7.2. Thêm lớp `background`

Train lại ST-GCN++ với ba lớp:

```text
0: background
1: backhand_drive
2: forehand_lift
```

`background` bao gồm:

- Đứng chờ.
- Di chuyển trên sân.
- Nhặt cầu.
- Điều chỉnh tư thế.
- Vung tay không thuộc hai kỹ thuật mục tiêu.
- Các motion proposal sai.

Đây là bài toán nhận diện hành động, không phải bài toán đánh giá đúng/sai kỹ thuật.

### 7.3. Hard-negative mining

Sau lần train đầu:

1. Chạy model trên video dài.
2. Lưu các đoạn `background` bị nhận nhầm thành cú đánh.
3. Xác nhận lại bằng tay.
4. Thêm chúng vào train set `background`.
5. Train lại model.

### 7.4. Điều kiện chấp nhận cú đánh

Một đoạn chỉ được chuyển sang đánh giá hình học khi:

- Không bị phân loại là `background`.
- Confidence hành động vượt ngưỡng.
- Pose quality đạt yêu cầu.
- Có đỉnh chuyển động tay rõ ràng.

## 8. Giai đoạn 4 — Tinh chỉnh biên cú đánh

Motion Proposal cho biên gần đúng. Sau khi ST-GCN++ xác nhận loại động tác:

1. Tìm đỉnh vận tốc cổ tay chính trong đoạn.
2. Đi lùi tới vùng chuyển động ổn định trước cú đánh.
3. Đi tới vùng vận tốc giảm ổn định sau cú đánh.
4. Thêm padding nhỏ hai phía.
5. Lưu ánh xạ frame của clip con về frame video gốc.

Mỗi cú đánh cần lưu:

```json
{
    "stroke_id": 1,
    "action": "forehand_lift",
    "start_frame": 142,
    "end_frame": 195,
    "peak_frame": 171,
    "action_confidence": 0.91
}
```

## 9. Giai đoạn 5 — Trích xuất đặc trưng sinh cơ học

### 9.1. Chuẩn hóa theo cơ thể

- Gốc tọa độ: trung điểm hai hông.
- Tỷ lệ: chiều rộng vai hoặc chiều dài thân.
- Mirror người thuận tay trái nếu mở rộng sau này.
- Giữ confidence cho từng đặc trưng.

### 9.2. Đặc trưng mỗi frame

Ưu tiên các đặc trưng quan sát được từ pose 2D:

- Góc khuỷu tay đánh.
- Góc vai tay đánh.
- Góc nghiêng thân.
- Góc đầu gối trái/phải.
- Khoảng cách cổ tay–vai.
- Độ cao cổ tay so với vai/đầu.
- Khoảng cách hai chân chia cho chiều rộng vai.
- Vận tốc cổ tay.
- Vận tốc góc khuỷu và vai.

Đầu ra một cú đánh:

```text
features.shape = (T, F)
```

Trong đó `T` là số frame và `F` là số đặc trưng.

### 9.2b. Đặc trưng chuỗi động học (cải tiến, tham khảo Guo & Lin 2026)

Các nghiên cứu kinematic trên smash cho thấy lực truyền theo chuỗi `gối → hông → vai → khuỷu → cổ tay`, với vận tốc góc đỉnh tăng dần theo thứ tự này. Có thể bổ sung đặc trưng thời gian thay vì chỉ đặc trưng góc tức thời:

- Thời điểm đạt vận tốc góc đỉnh của từng khớp trong đoạn cú đánh.
- Độ trễ (time lag) giữa các cặp khớp liên tiếp: `hông sau gối`, `vai sau hông`, `khuỷu sau vai`, `cổ tay sau khuỷu`.
- Thứ tự đỉnh vận tốc góc có đúng chuỗi trên hay không (kiểm tra hoán vị).

Đặc trưng này phục vụ trực tiếp tiêu chí `unstable_base`/`limited_follow_through` và một tiêu chí mới về phối hợp lực (xem mục 12.3).

**Đã kiểm chứng trên dữ liệu thật, kết quả không như kỳ vọng:** chạy trên 65 clip `forehand_lift`, chỉ 2/65 có đúng thứ tự đỉnh vận tốc góc gối→hông→vai→khuỷu→cổ tay (xem mục 12.3). Nguyên nhân là chuỗi động học này đúng cho cú lực mạnh dùng toàn thân (smash), còn `forehand_lift`/`backhand_drive` là cú nhẹ chủ yếu dùng tay nên vận tốc góc gối/hông gần như là nhiễu đo. Đặc trưng và hàm `extract_kinetic_chain_timing` vẫn giữ trong code vì tổng quát và đúng về mặt tính toán, nhưng **không dùng làm rule đánh giá lỗi cho hai kỹ thuật MVP hiện tại**.

### 9.2c. Đặc trưng trọng tâm (gravity line / support line)

Tham khảo Guo & Lin (2026): tính đường thẳng đứng từ trọng tâm cơ thể (xấp xỉ trung điểm hông hoặc trung bình có trọng số các khớp) chiếu xuống mặt đất, và đường nối hai bàn chân. Khoảng cách từ điểm chiếu trọng tâm tới đường nối hai chân là một đặc trưng ổn định tư thế, bổ sung cho `unstable_base` ở mục 12.1 vốn hiện chỉ dùng độ rộng chân và góc gối.

### 9.3. Không kết luận từ một frame

- Dùng median trong cửa sổ nhỏ.
- Một độ lệch phải tồn tại tối thiểu một số frame.
- Bỏ qua đoạn có confidence thấp.

## 10. Giai đoạn 6 — Xây tập mẫu tham chiếu

### 10.1. Thu thập mẫu

Với mỗi kỹ thuật:

- Khoảng 10–20 cú được chuyên gia xác nhận là đạt cho bản MVP.
- Nhiều người chơi nếu có thể.
- Cùng bố trí camera với điều kiện triển khai.
- Mỗi clip chỉ chứa một cú.

Tập này không dùng để train model đúng/sai. Nó dùng để xây vùng tham chiếu và kiểm thử.

### 10.2. Gắn nhãn pha trên mẫu

Đánh dấu thủ công các khoảng:

- `preparation`
- `backswing`
- `forward_swing`
- `contact_estimated`
- `follow_through`

Mốc `contact_estimated` chỉ là ước lượng vì chưa phát hiện vợt và cầu.

### 10.3. Tạo Reference Motion Envelope

1. Căn chỉnh các cú chuẩn bằng multivariate DTW.
2. Đưa chúng về một timeline chuẩn.
3. Tính median theo thời gian cho từng đặc trưng.
4. Tính MAD hoặc percentile 10–90 làm vùng biến thiên cho phép.
5. Lưu nhãn pha của timeline chuẩn.

**Cải tiến (tham khảo Wang 2026):** sau khi có vùng percentile 10–90 tính từ dữ liệu, tổ chức một vòng **chuyên gia hiệu chỉnh thủ công** từng ngưỡng trước khi đóng băng thành reference profile (ví dụ nới ngưỡng nếu percentile quá chặt so với biến thể phong cách hợp lệ). Ghi lại ngưỡng cuối cùng dưới dạng bảng tương tự:

| Tiêu chí                     | Đặc trưng                       | Vùng tham chiếu (dữ liệu) | Vùng sau hiệu chỉnh chuyên gia |
| ---------------------------- | ------------------------------- | ------------------------- | ------------------------------ |
| `elbow_too_low_in_backswing` | `elbow_angle` tại pha backswing | percentile 10–90          | ...                            |

Bảng này nên lưu cùng file cấu hình `configs/biomechanics/*.yaml` để tách biệt ngưỡng khỏi code.

Artifact tham chiếu dự kiến:

```text
models/references/
├── forehand_lift_reference.npz
└── backhand_drive_reference.npz
```

## 11. Giai đoạn 7 — Căn chỉnh và phát hiện độ lệch

### 11.1. Multivariate DTW

DTW chỉ chạy trên từng cú đã được tách, không chạy trên toàn bộ video dài.

Các đặc trưng dùng để căn chỉnh nên tập trung vào nhịp động tác:

- Góc khuỷu.
- Góc vai.
- Vị trí cổ tay tương đối.
- Khoảng cách cổ tay–vai.
- Vận tốc cổ tay.

Giới hạn đường căn chỉnh bằng cửa sổ Sakoe–Chiba để tránh ghép đầu cú với cuối cú.

### 11.2. Chuyển nhãn pha từ mẫu sang cú người dùng

Đường DTW ánh xạ frame người dùng sang timeline mẫu. Nhãn pha của frame mẫu được truyền sang frame người dùng.

Nhờ vậy không cần train thêm model phân loại pha trong MVP.

### 11.3. Điểm lệch theo đặc trưng

Với frame `t`, đặc trưng `j`:

```text
deviation[t, j]
  = abs(user[t, j] - reference_median[r(t), j])
    / (reference_MAD[r(t), j] + epsilon)
```

Trong đó `r(t)` là frame tham chiếu tương ứng do DTW tạo ra.

### 11.4. Gom thành sự kiện

- Chỉ giữ độ lệch có confidence tốt.
- Yêu cầu lệch liên tục trong một khoảng tối thiểu.
- Gộp các đoạn lỗi cách nhau rất ngắn.
- Chọn frame có độ lệch lớn nhất làm `worst_frame`.
- Ánh xạ ngược về frame và timestamp của video gốc.

## 12. Tiêu chí đánh giá đề xuất

### 12.1. `forehand_lift`

MVP có thể chọn:

1. `elbow_too_low_in_backswing`
    - So sánh vị trí khuỷu tương đối với vai trong pha backswing.
2. `arm_not_extended_near_contact`
    - So sánh góc khuỷu trong vùng contact ước lượng.
3. `unstable_base`
    - Xem độ rộng chân, góc gối và dao động tâm hông ở preparation/contact.

### 12.2. `backhand_drive`

MVP có thể chọn:

1. `elbow_too_close_to_torso`
    - Khoảng cách khuỷu–thân thấp hơn vùng tham chiếu.
2. `insufficient_reach`
    - Khoảng cách cổ tay–vai thấp ở contact ước lượng.
3. `limited_follow_through`
    - Quỹ đạo và biên độ cổ tay sau contact nhỏ hơn vùng tham chiếu.

Tên và ý nghĩa các lỗi phải được chuyên gia cầu lông xác nhận trước khi cố định.

### 12.3. Cải tiến — mô hình hybrid cho lỗi phối hợp nhiều khớp (tham khảo Wang 2026)

Các lỗi so sánh một đặc trưng đơn lẻ với ngưỡng (12.1, 12.2) dùng rule engine như hiện tại là đủ. Nhưng lỗi "thiếu phối hợp chuỗi động học" (vd tay không truyền lực đúng thứ tự gối→hông→vai→khuỷu→cổ tay) cần nhìn nhiều khớp theo thời gian cùng lúc, khó biểu diễn bằng một ngưỡng. Đề xuất bổ sung:

- Một mô hình nhẹ (GBDT hoặc tương đương) nhận đầu vào là chuỗi thời điểm/độ trễ đỉnh vận tốc góc các khớp (mục 9.2b), huấn luyện để phân loại "chuỗi lực hợp lý" hay không, dựa trên nhãn của tập mẫu chuẩn và các mẫu lỗi được chuyên gia gắn nhãn.
- Rule engine vẫn là thành phần chính; mô hình ML chỉ bổ trợ cho các lỗi không thể mô tả bằng ngưỡng đơn.
- Không thay thế Reference Motion Envelope, chỉ thêm một loại `event` mới trong đầu ra JSON (mục 13.1).

**Giới hạn phát hiện thực nghiệm (script `scripts/data/build_reference_profiles.py` chạy trên 65 clip `forehand_lift` thật):** chỉ 2/65 clip có thứ tự đỉnh vận tốc góc đúng chuỗi gối→hông→vai→khuỷu→cổ tay, nhiều `transfer_time` còn âm. Chuỗi động học gối→hông→vai→khuỷu→cổ tay đo được trong paper Guo & Lin (2026) là cho **smash** — cú lực mạnh cần toàn thân. `forehand_lift` là cú nhẹ gần lưới, chân gần như không tham gia truyền lực nên vận tốc góc gối/hông chủ yếu là nhiễu đo, không phải tín hiệu thật. Do đó rule `kinetic_chain_out_of_order`/lag-timing (9.2b, 12.3) **chỉ nên áp dụng nếu sau này mở rộng sang cú lực mạnh (smash/clear)**, không dùng cho `forehand_lift` hoặc `backhand_drive` trong phạm vi MVP hiện tại.

### 12.4. Cải tiến — vùng đệm không chắc chắn quanh ngưỡng

Thay vì coi ngưỡng là ranh giới nhị phân đúng/sai, thêm một vùng đệm (ví dụ ±5% quanh biên MAD/percentile) mà tại đó sự kiện được gắn nhãn `review` thay vì `deviation` khẳng định. Việc này giảm cảnh báo sai ở các trường hợp biên và khớp với cách mô tả "độ lệch" thay vì "kết luận tuyệt đối" đã nêu ở mục 2.

## 13. Đầu ra hệ thống

### 13.1. JSON

```json
{
    "video_duration": 60.0,
    "detected_strokes": 3,
    "strokes": [
        {
            "stroke_id": 1,
            "action": "forehand_lift",
            "start_time": 4.73,
            "end_time": 6.5,
            "action_confidence": 0.91,
            "technique_score": 76,
            "events": [
                {
                    "phase": "forward_swing",
                    "start_time": 5.47,
                    "end_time": 5.67,
                    "worst_frame": 169,
                    "features": ["right_elbow_angle", "wrist_height"],
                    "message": "Tay đánh có độ lệch lớn so với vùng tham chiếu.",
                    "evidence_confidence": 0.87
                }
            ]
        }
    ]
}
```

### 13.2. Video kết quả

- Skeleton trên người chơi.
- Góc tại khớp liên quan.
- Màu xanh/vàng/đỏ theo mức lệch.
- Nhãn loại cú đánh.
- Nhãn pha hiện tại.
- Timeline đánh dấu các cú và đoạn lệch.

### 13.3. Báo cáo

- Số cú phát hiện được.
- Phân bố loại cú đánh.
- Điểm theo từng tiêu chí.
- Frame lệch nhất của mỗi cú.
- Biểu đồ chuỗi người dùng và vùng tham chiếu.

## 14. Cấu trúc file dự kiến

```text
src/ai_classifier/
├── segmentation/
│   ├── motion_proposal.py
│   ├── boundary_refinement.py
│   └── types.py
├── biomechanics/
│   ├── angles.py
│   ├── normalization.py
│   ├── features.py
│   └── types.py
├── reference_analysis/
│   ├── dtw.py
│   ├── reference_profile.py
│   └── alignment.py
└── error_detection/
    ├── detector.py
    ├── rules.py
    └── types.py

configs/
├── segmentation/
│   └── motion_proposal.yaml
└── biomechanics/
    ├── forehand_lift.yaml
    └── backhand_drive.yaml

scripts/
├── data/
│   ├── build_background_dataset.py
│   └── build_reference_profiles.py
└── inference/
    └── analyze_long_video.py
```

## 15. Kiểm thử và đánh giá

### 15.1. Phát hiện cú đánh

- Precision/Recall/F1 theo sự kiện cú đánh.
- Một dự đoán được xem là đúng nếu temporal IoU với đoạn thật vượt ngưỡng.
- Đo sai số biên bắt đầu/kết thúc theo frame hoặc mili-giây.

### 15.2. Phân loại động tác

- Accuracy và macro F1 cho ba lớp.
- Confusion matrix.
- Chia train/validation/test theo video nguồn hoặc người chơi, không chia ngẫu nhiên các clip cùng nguồn.

### 15.3. Đánh giá độ lệch kỹ thuật

- Chuyên gia xác nhận các sự kiện trên một tập test nhỏ.
- Đo precision/recall theo từng loại cảnh báo.
- Đo sai số vị trí `worst_frame` so với vùng chuyên gia đánh dấu.
- Báo cáo riêng trường hợp pose không đủ chất lượng.
- **Cải tiến (tham khảo Wang 2026):** tính riêng F1-score cho từng loại lỗi (`elbow_too_low_in_backswing`, `unstable_base`, ...) so với nhãn của ít nhất hai chuyên gia độc lập, thay vì chỉ một số precision/recall gộp. Việc này giúp phát hiện loại lỗi nào model/rule đang yếu để ưu tiên hiệu chỉnh ngưỡng.

### 15.4. Ablation study

So sánh:

1. Chỉ dùng Motion Proposal.
2. Motion Proposal + ST-GCN++.
3. So sánh timeline tuyến tính, không DTW.
4. So sánh có DTW.
5. Một mẫu chuẩn so với Reference Motion Envelope từ nhiều mẫu.

## 16. Thứ tự triển khai

### Milestone 1 — Motion Proposal

- Tính vận tốc và vận tốc góc.
- Phát hiện đoạn ứng viên trong video dài.
- Xuất JSON và preview timeline.
- Viết unit test với chuỗi pose tổng hợp.

### Milestone 2 — ST-GCN++ có `background`

- Thu và gán nhãn background.
- Cập nhật config/dataset thành ba lớp.
- Train và đánh giá theo nguồn độc lập.
- Chạy hard-negative mining ít nhất một vòng.

### Milestone 3 — Biomechanics

- Chuẩn hóa tọa độ.
- Tính góc, khoảng cách, vận tốc và confidence.
- Xuất feature CSV/NPZ.
- Vẽ biểu đồ để kiểm tra trực quan.

### Milestone 4 — Reference profile và DTW

- Chọn/gán nhãn các mẫu đạt.
- Cài multivariate DTW có giới hạn.
- Xây median/MAD envelope.
- Chuyển nhãn pha qua đường căn chỉnh.

### Milestone 5 — Error localization

- Tính độ lệch từng frame/từng đặc trưng.
- Gom frame thành sự kiện.
- Tạo điểm, thông báo và mức tin cậy.
- Render kết quả lên video.

### Milestone 6 — Đánh giá end-to-end

- Chuẩn bị video dài test chưa xuất hiện khi phát triển.
- Đánh giá phát hiện cú, phân loại và cảnh báo riêng biệt.
- Phân tích failure cases.
- Hoàn thiện báo cáo và demo.

## 17. Rủi ro và phương án giới hạn

| Rủi ro                                         | Cách xử lý                                       |
| ---------------------------------------------- | ------------------------------------------------ |
| Model ép mọi đoạn thành một trong hai kỹ thuật | Thêm lớp `background`                            |
| Chuyển động nhặt cầu tạo peak cổ tay           | ST-GCN++ xác nhận motion proposal                |
| Pose 2D sai do góc camera                      | Cố định camera cho MVP                           |
| Người chơi che khuất tay                       | Confidence-aware filtering                       |
| Cú nhanh/chậm khác mẫu                         | DTW theo từng cú                                 |
| Một mẫu chuẩn mang tính cá nhân                | Dùng nhiều mẫu và median/MAD                     |
| Không thấy vợt/cầu                             | Chỉ gọi `contact_estimated`                      |
| Luật kỹ thuật thiếu cơ sở                      | Chuyên gia xác nhận tiêu chí và mẫu              |
| Video chứa kỹ thuật ngoài phạm vi              | Gán `background`/`unsupported` và không đánh giá |

## 18. Tiêu chí hoàn thành MVP

MVP được xem là hoàn thành khi:

1. Nhận video dài khoảng 1 phút.
2. Trả ra danh sách các cú đánh kèm timestamp.
3. Phân biệt `background`, `backhand_drive` và `forehand_lift`.
4. Tạo được đặc trưng hình học có kiểm soát confidence.
5. Căn chỉnh từng cú với reference profile.
6. Hiển thị ít nhất hai loại cảnh báo cho một kỹ thuật.
7. Chỉ ra `start_frame`, `end_frame` và `worst_frame` của cảnh báo.
8. Có video hoặc biểu đồ giải thích kết quả.
9. Có tập test tách theo nguồn và xác nhận của người có chuyên môn.

## 19. Cách mô tả trong báo cáo

> Hệ thống sử dụng YOLOv8 Pose để trích xuất chuỗi khớp xương và ST-GCN++ để nhận diện các kỹ thuật cầu lông trong video dài. Các cú đánh được đề xuất bằng tín hiệu động học và xác nhận bởi mô hình nhận diện hành động. Sau đó, chuyển động được chuẩn hóa theo cơ thể, căn chỉnh với tập mẫu đạt bằng Dynamic Time Warping và đánh giá bằng mô hình tham chiếu sinh cơ học theo thời gian. Hệ thống trả về mức độ lệch, vị trí thời gian và các khớp liên quan thay vì huấn luyện một bộ phân loại đúng/sai khó giải thích.

## 20. Tài liệu tham khảo

- Wang, Q. (2026). _Research on Badminton Technique Movement Diagnosis and Instructional Feedback Optimization Based on Video Analysis._ — nguồn cho: xây ngưỡng bằng percentile 10–90 + hiệu chỉnh chuyên gia (mục 10.3), mô hình hybrid rule engine + GBDT cho lỗi phối hợp nhiều khớp (mục 12.3), vùng đệm không chắc chắn quanh ngưỡng (mục 12.4), đánh giá F1 theo từng loại lỗi (mục 15.3).
- Guo, X.-P., & Lin, L.-W. (2026). _Kinematic analysis of badminton smash techniques of badminton players._ Journal of Human Sport and Exercise, 21(2). — nguồn cho: đặc trưng chuỗi động học và độ trễ đỉnh vận tốc góc giữa các khớp (mục 9.2b), đặc trưng gravity line/support line cho ổn định tư thế (mục 9.2c).
- Gao, Y. (2026). _Physics-Guided One-Dimensional Convolutional Attention Network for Badminton Pose-Trajectory Refinement._ Journal of Computing and Electronic Information Management, 22(2). — tham khảo cho bước tiền xử lý/làm mượt pose (mục 5), có thể cân nhắc nếu `KeypointSmoother` hiện tại chưa đủ ổn định độ dài xương.
- Wu, Y. et al. (2025). _Enhanced Pose Estimation for Badminton Players via Improved YOLOv8-Pose with Efficient Local Attention._ Sensors, 25(14), 4446. — tham khảo cho giai đoạn trích xuất pose (mục 5), không ảnh hưởng trực tiếp tới phần đánh giá hình học.
