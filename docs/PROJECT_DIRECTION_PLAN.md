# Kế hoạch chuyển hướng AI Classifier

## Quyết định chốt ngày 17/09/2026

### Phạm vi sản phẩm chính

- Đầu vào mục tiêu là video trận đấu cầu lông đơn dạng broadcast: hai người,
  camera cuối sân tương đối cố định và nhìn thấy toàn sân.
- Không hỗ trợ video một người tập trong phạm vi đề tài. Mọi quyết định kiến
  trúc, huấn luyện và đánh giá đều phục vụ video trận đấu có hai người.
- Đầu ra MVP của một video là timeline từng lần chạm cầu gồm `hit_frame`, thời
  gian, người đánh, một trong tám lớp cú đánh, `stroke_side`, confidence và
  artifact JSON/video gắn nhãn.
- Phân tích chiến thuật chỉ được xây dựng sau khi timeline đã được đánh giá
  end-to-end; không suy luận chiến thuật trực tiếp từ một nhãn cho cả video.

### Trạng thái và vai trò của RGB baseline

- R(2+1)D-18 hiện tại không phải hướng train sai. Đây là classifier cho clip
  ngắn đã căn quanh một lần chạm cầu và sẽ trở thành nhánh RGB của pipeline.
- Checkpoint epoch 10 full-data đạt validation stroke Macro-F1 `70,27%`, side
  Macro-F1 `78,01%`. Trên bộ test cân bằng 200 clip ShuttleSet: stroke accuracy
  `67,50%`, stroke Macro-F1 `66,67%`, side accuracy constrained `74,50%` và
  joint accuracy `50,50%`.
- Epoch 10 là RGB baseline đã chốt cho giai đoạn hiện tại. Không tăng epoch RGB
  vô hạn; ưu tiên đánh giá trên trận broadcast external rồi chuyển sang hit
  detection và skeleton fusion.
- YOLO Pose hiện mới phục vụ phát hiện/crop người đánh; keypoint chưa phải đầu
  vào của R(2+1)D-18.

### Pipeline đã chốt

```text
Video trận đấu/rally
  -> hit-frame detector (no_hit / upper_hit / lower_hit)
  -> xác định đúng người đánh và cắt clip quanh hit
  -> RGB branch + skeleton branch
  -> fusion heads: 8-class stroke + forehand/backhand/aroundhead
  -> confidence/unknown gate + temporal smoothing
  -> StrokeEvent JSON + video timeline
  -> thống kê chuỗi cú đánh và chiến thuật
```

### Thứ tự triển khai sau RGB epoch 10

1. Chốt checkpoint RGB bằng validation và test theo trận; giữ bộ test 200 clip
   cố định để so sánh, không dùng nó để chỉnh threshold.
2. Train hit-frame detector từ annotation ShuttleSet và đo precision/recall tại
   tolerance ±2, ±5 và ±15 frame.
3. Trích/cache skeleton đúng người đánh quanh hit-frame; train skeleton-only
   baseline cho tám lớp và `stroke_side`.
4. Ghép RGB + skeleton bằng late fusion hoặc feature fusion; chỉ giữ fusion nếu
   cải thiện rõ trên cùng split theo trận.
5. Test end-to-end trên ít nhất hai trận broadcast không tham gia train, xuất
   timeline và phân rã lỗi localization/player/stroke/side.
6. Chỉ sau đó mới thêm court position và shuttle trajectory kiểu BST để xử lý
   các cặp khó như `clear/drop/smash`, `lift/net_shot`, `drive/net_attack`.

### Mục tiêu đo lường

- Không cam kết 90% trên video bất kỳ. Với miền broadcast, mục tiêu đầu tiên là
  hit recall trên 85%, stroke accuracy 70–80% và side accuracy 80–90%.
- Báo cáo thêm joint accuracy đúng đồng thời hit, người đánh, stroke và side;
  đây là metric gần chất lượng sản phẩm hơn accuracy của từng head riêng lẻ.
- Cho phép trả `unknown` khi confidence thấp thay vì ép mọi cửa sổ thành một
  trong tám cú đánh.

> Cập nhật: 16/09/2026  
> Quyết định: tiếp tục phát triển trên repository `AI_Classifier`; ưu tiên bài
> toán phân loại cú đánh trước, sau đó mới định vị cú đánh trong video dài và
> phân tích chiến thuật.

## 1. Quyết định tổng quát

Không viết lại dự án từ đầu. Repository hiện tại đã có nhiều thành phần sản
phẩm có giá trị: API upload video, job bất đồng bộ, pose estimation, kiểm tra
chất lượng, timeline/viewer, artifact JSON/video và 181 unit test đang vượt
qua. Viết lại sẽ mất phần ổn định này mà không giải quyết được vấn đề khó nhất
là dữ liệu và protocol đánh giá.

Đề tài được thu hẹp và đổi trọng tâm thành:

> **Hệ thống định vị thời gian và phân loại cú đánh trong video trận đấu cầu
> lông đơn, làm nền tảng cho phân tích chuỗi chiến thuật.**

Thứ tự bắt buộc:

1. Phân loại đúng một clip chứa một cú đánh.
2. Phát hiện các hit/action trong một rally chưa cắt.
3. Kết hợp thành pipeline video → stroke timeline.
4. Chỉ sau khi tầng nhận diện được đánh giá mới bổ sung phân tích chiến thuật.

Không tiếp tục tuyên bố hệ thống có thể tự động đánh giá lỗi kỹ thuật tổng quát
từ video điện thoại khi chưa có dataset lỗi do chuyên gia gán nhãn.

### Yêu cầu cứng về số lớp

- Sản phẩm và thực nghiệm cuối cùng phải phân loại **ít nhất 5 nhóm cú đánh**.
- Mục tiêu ưu tiên là 7 lớp chung hoặc 8 lớp Fine-Badminton.
- Không chấp nhận lùi xuống bài toán `stroke/background`, hai lớp hoặc ba lớp
  làm kết quả chính; các bài toán đó chỉ được dùng làm sanity check.
- Nếu audit không tạo được ít nhất 5 lớp có train/validation/test độc lập theo
  trận, phải bổ sung VideoBadminton/BFMD/ShuttleSet thay vì tiếp tục gộp lớp.

## 2. Phạm vi sản phẩm

### 2.1. Input được hỗ trợ

- Trận đấu đơn, hai người.
- Camera cố định ở cuối sân, nhìn thấy toàn bộ sân.
- Video ngang, tối thiểu 720p và FPS đủ ổn định.
- Phiên bản đầu nhận một rally hoặc một đoạn ngắn; video toàn trận là bước sau.
- Có quality gate để từ chối replay, camera lệch, video một người hoặc sân bị
  che quá nhiều.

### 2.2. Output bắt buộc

```json
{
  "events": [
    {
      "start_frame": 1210,
      "hit_frame": 1221,
      "end_frame": 1234,
      "player_side": "upper",
      "stroke": "net_shot",
      "stroke_side": "backhand",
      "confidence": 0.81
    }
  ]
}
```

Giao diện phải cho phép bấm vào từng event để phát đúng đoạn video và xem bằng
chứng pose/cầu nếu có.

### 2.3. Ngoài phạm vi phiên bản đầu

- Đánh đôi.
- Video điện thoại tự do nhiều góc.
- Phân biệt đầy đủ 29 lớp fine-grained ngay từ đầu.
- Đánh giá đúng/sai biomechanics hoặc chẩn đoán chấn thương.
- Tư vấn chiến thuật bằng LLM mà không có dữ liệu rally có cấu trúc.
- Real-time và tự động calibration sân trong mọi điều kiện.

## 3. Phân rã bài toán

```text
Video/clip
   │
   ├─ Phase A: clip classification (ground-truth boundary)
   │      └─ player side + stroke class + forehand/backhand/aroundhead
   │
   ├─ Phase B: hit/action localization
   │      └─ start/end/hit frame + player side
   │
   ├─ Phase C: end-to-end stroke timeline
   │      └─ localization + classification + confidence
   │
   └─ Phase D: tactical analysis
          └─ rally sequence, pattern, shot influence, heatmap
```

Phase A là ưu tiên hiện tại. Không bắt đầu Phase D khi chưa có baseline và
metric của Phase A.

## 4. Chiến lược dataset

### 4.1. Vai trò từng dataset

| Dataset | Nội dung chính | Dùng cho | Không dùng để kết luận |
| --- | --- | --- | --- |
| Fine-Badminton | 10 video, 9.817 interval, 29 lớp, start/end và upper/lower | Dataset chính cho clip classification và temporal action localization | Phân tích chiến thuật sâu vì thiếu score/vị trí cầu/rally outcome |
| ShuttleSet | 44 trận, 3.685 rally, 36.482 stroke, hit frame, hitter, shot type, `backhand`, `aroundhead`, vị trí và outcome | Head forehand/backhand/aroundhead, baseline BST, hit spotting và tactical model | Interval start/end chính xác nếu chỉ dùng annotation gốc |
| ShuttleSet NPY của BST | Pose hai người + court position + shuttle trajectory đã xử lý | Reproduce và đánh giá clip classifier BST | Production video upload; đây không phải MP4 |
| BFMD | Full match, rally, hit, shot type, court, bbox, pose, shuttle, caption | Dataset ưu tiên cho end-to-end/full-match và external validation | Dùng ngay toàn bộ khi chưa audit đồng bộ video/link |
| Dữ liệu 2 lớp hiện tại | Backhand drive và forehand clear, chủ yếu clip biểu diễn | Giữ làm regression test cho pipeline pose/ST-GCN++ | Khả năng tổng quát trên trận đấu |
| MultiSense 3 lớp | Background + hai kỹ thuật, split theo subject | Kiểm thử quality gate/biomechanics phụ | Stroke taxonomy của trận đấu |

Nguồn:

- Fine-Badminton: <https://zenodo.org/records/20292976>
- ShuttleSet/CoachAI: <https://github.com/wywyWang/CoachAI-Projects>
- BFMD: <https://github.com/Ning-D/BFMD>
- BST: <https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer>

#### Các dataset bổ sung cần audit

Không giới hạn nghiên cứu ở Fine-Badminton. Danh sách ưu tiên được mở rộng như
sau:

| Dataset | Quy mô/cấu trúc | Giá trị | Mức ưu tiên |
| --- | --- | --- | --- |
| VideoBadminton | 7.822 clip, 18 lớp, 19 VĐV, khoảng 145 phút, video tự quay 60 FPS | Chỉ dùng tham khảo/ablation; không trộn vào model chính nếu góc quay không tương thích trận đấu hai người | Thấp |
| BFMD | 19 trận, 1.687 rally, 16.751 hit; bản phát hành có 11.301 shot, court/pose/shuttle/shot type | Nguồn tốt nhất cho full-match, hit detection và external validation | Rất cao |
| Badminton Olympics | 10 trận broadcast, interval và khoảng 12 lớp | Bổ sung domain broadcast, dùng cho TemPose/external test | Cao |
| BadmintonDB | 9 trận, 9.671 stroke; chủ yếu hai VĐV Momota–Ginting | Interval/action và external test, nhưng player diversity thấp | Trung bình |
| ShuttleSet | 44 trận, 36.492 stroke, 18 lớp, hit frame và tactical fields | Hit spotting, BST và tactical analysis | Rất cao |
| S2-Labeling | 6 trận có shot-by-shot tactical annotations | Kiểm tra format và external hit/rally test | Trung bình |
| ShuttleSet22 | Mở rộng dữ liệu stroke-level cho forecasting | Tactical forecasting; không phải nguồn video classification đầu tiên | Thấp ở Phase A |
| BadPL | 5.566 sample nhưng là dữ liệu bảo mật | Không thể phụ thuộc vì không tải công khai | Không dùng |

Nguồn VideoBadminton:

- Paper: <https://arxiv.org/abs/2403.12385>
- Repository: <https://github.com/qilimk/VideoBadminton>

VideoBadminton đặc biệt phù hợp để bổ sung Phase A vì đã là clip action-centric,
có 18 lớp và benchmark báo cáo nhiều kiến trúc như SlowFast, ST-GCN và PoseC3D.
Tuy nhiên đây là video tự quay của đội tuyển trường, khác miền với broadcast
Fine-Badminton; không được trộn rồi split ngẫu nhiên.

#### Chiến lược multi-dataset

Không nối tất cả file vào một thư mục và train ngay. Thực hiện theo bốn thí
nghiệm tách biệt:

1. **Within-dataset:** train/test Fine-Badminton theo trận.
2. **Within-dataset:** train/test VideoBadminton theo subject/video nguồn.
3. **Cross-dataset:** train Fine-Badminton, test VideoBadminton và chiều ngược
   lại trên taxonomy giao nhau.
4. **Joint training:** train hai nguồn với `dataset_id`, source-balanced sampler
   và test trên trận/subject chưa thấy.

Taxonomy giao nhau an toàn ban đầu nên chỉ gồm 7 lớp:

```text
serve, clear, smash, drop, net_shot, lift/lob, drive
```

Chỉ thêm `net_attack` khi audit xác nhận cả hai dataset có đủ `push/net_kill`.

Nếu buộc phải dùng taxonomy tối thiểu, sử dụng ít nhất 6 lớp sau để vẫn vượt
yêu cầu năm động tác và giữ ý nghĩa chuyên môn:

```text
serve, clear, smash, drop, net_shot, lift
```

`drive` là lớp thứ bảy được giữ khi coverage theo trận đạt yêu cầu. Không gộp
smash/drop/net chỉ để làm đẹp metric.

Mỗi sample phải có thêm:

```text
dataset_id, domain (broadcast/match_fixed_camera), subject_id, match_id, camera_id
```

Mục tiêu của multi-dataset không chỉ là tăng số clip mà là đo và giảm domain
shift giữa các giải đấu, camera và chất lượng broadcast. Fine-Badminton vẫn là nguồn chính
cho localization; VideoBadminton là nguồn tăng độ đa dạng cho classifier; BFMD
và ShuttleSet là nguồn cho hit/rally/tactical layer.

#### Mở rộng forehand/backhand bằng ShuttleSet

Không nhân trực tiếp tám lớp coarse thành các lớp ghép như
`forehand_lift`, `backhand_lift`, ... vì phân bố tổ hợp rất lệch và một số tổ
hợp gần như không có mẫu. Giữ hai mục tiêu dự đoán tách biệt:

```text
stroke_class: serve | clear | smash | drop | net_shot | lift | drive | net_attack
stroke_side:  forehand | backhand | aroundhead | unknown
```

`aroundhead` được giữ thành nhãn riêng. Đây là cú forehand overhead thực hiện ở
phía backhand, không được âm thầm gộp vào `forehand`. Có thể báo cáo thêm metric
nhị phân với `aroundhead -> forehand`, nhưng dữ liệu và output chuẩn vẫn giữ ba
nhãn.

Audit annotation ShuttleSet local ngày 16/09/2026:

```text
104 set CSV, 36.482 stroke gốc
35.041 stroke sau khi loại 1.441 dòng có cờ flaw
forehand mặc định (backhand/aroundhead trống): 18.041
backhand=1.0:                                13.666
aroundhead=1.0:                               3.333
xung đột backhand=1.0 và aroundhead=1.0:           1
```

Quy tắc parse bắt buộc: ShuttleSet mã hóa giá trị dương bằng `1.0` và giá trị
âm bằng ô trống, không phải `0/1`. Dòng có cả hai cờ dương phải bị loại hoặc gán
`unknown`, không tự chọn một nhãn. Các cột nguồn phải được giữ nguyên để audit.

Fine-Badminton không có ground truth forehand/backhand. Vì vậy dùng backbone
chung và hai classification head:

```text
video -> shared backbone -> stroke head (8 lớp)
                         -> stroke-side head (3 lớp)

L = L_stroke + lambda_side * mask_side * L_side
```

- Sample Fine-Badminton chỉ đóng góp `L_stroke`.
- Sample ShuttleSet đóng góp `L_side` và chỉ đóng góp `L_stroke` sau khi mapping
  shot type sang taxonomy chung được audit bằng YAML và unit test.
- Không sinh pseudo-label forehand/backhand cho Fine-Badminton trong baseline.
- Dùng source-balanced sampler để số lượng ShuttleSet không lấn át
  Fine-Badminton.
- So sánh tuần tự: Fine-only 8 lớp; ShuttleSet-only stroke-side; joint
  multi-task; cuối cùng ablation bỏ head stroke-side.

Bản checkout hiện có đủ annotation CSV nhưng chưa có media/NPY ShuttleSet đã
giải nén trong `external/.../ShuttleSet_merged`. Do đó có thể xây manifest và
audit nhãn ngay, nhưng chưa được tuyên bố đã train head mới cho tới khi checksum
và liên kết giữa annotation với video/NPY được xác nhận.

### 4.2. Dataset chính theo phase

#### Phase A — clip classification

1. Fine-Badminton: dùng ground-truth `start/end` để cắt clip.
2. Chuẩn hóa 60 chuỗi label thô Anh/Trung thành 29 label chuẩn.
3. Gộp thành 8 lớp coarse cho MVP.
4. ShuttleSet NPY: dùng để reproduce BST 25 lớp và làm bằng chứng baseline
   multimodal, không trộn taxonomy với Fine-Badminton.
5. ShuttleSet annotation: huấn luyện head `stroke_side` ba lớp; sau khi audit
   media và mapping shot type, thử joint multi-task với Fine-Badminton.

#### Phase B — localization

1. Fine-Badminton cho interval start/end.
2. BFMD cho hit-frame, player side, rally và các modality đã trích xuất.
3. ShuttleSet cho hit-frame/event spotting và tactical table.

#### Phase C — external test

- Giữ riêng ít nhất 2 trận không tham gia train.
- Thêm các video trận đấu hai người ngoài dataset, ưu tiên broadcast hoặc camera
  cuối sân tương thích; gán nhãn thủ công bởi hai người độc lập.
- Không dùng clip một người tập để đánh giá hoặc chọn checkpoint của sản phẩm.
- Không điều chỉnh threshold sau khi đã xem kết quả external test.

### 4.3. Taxonomy MVP tám lớp

| Lớp MVP | Nhãn Fine-Badminton được gộp |
| --- | --- |
| `serve` | Short Serve, High Serve, Flat Serve |
| `clear` | High Clear, Flat Clear, Clear Return of Serve |
| `smash` | Smash (Full-length), Short Smash, Jump Smash, Smash Return of Serve |
| `drop` | Drop Shot và các biến thể return |
| `net_shot` | Net Shot, Soft Net Shot, cross-court net và các biến thể return |
| `lift` | Lift, Defensive Lift và các biến thể return |
| `drive` | Drive, Drive Return of Smash |
| `net_attack` | Push Shot, Net Kill |

Giữ song song hai cột `raw_label` và `coarse_label`; không xóa nhãn 29 lớp.
Mapping phải nằm trong YAML có version và unit test.

`stroke_side` là thuộc tính trực giao với taxonomy tám lớp, không phải lớp coarse
thứ chín. Giá trị chuẩn là `forehand`, `backhand`, `aroundhead`, `unknown`.

### 4.4. Quy tắc split chống leakage

- Split theo **trận**, không split ngẫu nhiên theo clip.
- Gợi ý ban đầu: 6 trận train, 2 validation, 2 test.
- Báo cáo thêm 5-fold cross-validation theo trận nếu tài nguyên cho phép.
- Không để clip temporal augmentation của cùng interval đi sang split khác.
- Theo dõi `match_id`, `player_id`, `source_video`, `start/end` trong manifest.
- Với BFMD/ShuttleSet, kiểm tra trùng vận động viên giữa split và ghi đây là
  giới hạn nếu không tạo được player-disjoint split.

Schema manifest thống nhất:

```text
sample_id, dataset, match_id, video_path, start_frame, hit_frame, end_frame,
player_side, raw_label, coarse_label, stroke_side, stroke_side_source,
split, fps, annotation_source
```

## 5. Model ladder: từ đơn giản đến hoàn chỉnh

Không chọn một model duy nhất ngay từ đầu. Mỗi model trả lời một câu hỏi khác
nhau và tạo baseline để chứng minh cải tiến.

### 5.1. Phase A — phân loại clip trước

#### A0. Baseline thống kê

- Majority class.
- Random theo phân bố train.
- Dùng để phát hiện metric hoặc split sai.

#### A1. ST-GCN++ skeleton baseline

- Tái sử dụng pipeline PySKL hiện tại.
- Trích pose người thực hiện theo `upper/lower` ground truth.
- Train lại head 8 lớp trên Fine-Badminton, không dùng checkpoint hai lớp làm
  kết quả mới.
- Thử thêm skeleton cả hai người nếu tracking ổn định.

Vai trò: đo xem chỉ chuyển động cơ thể cung cấp được bao nhiêu thông tin.

#### A2. RGB clip classifier

Model khuyến nghị: một backbone video pretrained nhỏ như X3D-S/VideoMAE-Small
hoặc model tương đương trong MMAction2.

- Input trực tiếp clip MP4.
- Không phụ thuộc court calibration hay TrackNet.
- Fine-tune trên 8 lớp.

Vai trò: baseline dễ triển khai cho video upload và đối chiếu skeleton-only.

#### A3. BST multimodal baseline

- Pose hai người + court position + shuttle trajectory.
- Đã reproduce checkpoint ShuttleSet trên 3.499 test samples:
  Accuracy 83,77%, Macro-F1 82,09%.
- Không dùng checkpoint 25 lớp trực tiếp cho Fine-Badminton; muốn so sánh cùng
  taxonomy phải thay head và train lại.

Vai trò: kiểm tra giá trị bổ sung của cầu và vị trí sân. BST không phải model
production chính vì cần clip boundary và preprocessing phức tạp.

#### Quyết định Phase A

Train và so sánh tối thiểu:

1. ST-GCN++ 8 lớp.
2. RGB pretrained classifier 8 lớp.
3. BST giữ làm benchmark ShuttleSet; chỉ retrain 8 lớp nếu còn tài nguyên.

Chỉ chuyển Phase B khi có ít nhất một classifier vượt baseline với Macro-F1
hợp lý trên test theo trận và lỗi từng lớp đã được phân tích.

### 5.2. Phase B — định vị cú đánh

#### B1. Heuristic/event baseline

- Motion peak cổ tay/khuỷu từ module segmentation hiện tại.
- TrackNet đổi hướng quỹ đạo cầu.
- Luật hai người đánh luân phiên.

Đây là baseline giải thích được, không phải model cuối.

#### B2. Hit-frame detector

- Ba lớp theo frame: `no_hit`, `upper_hit`, `lower_hit`.
- Input có thể gồm pose hai người + cầu.
- Đánh giá tại tolerance ±2, ±5 và ±15 frame.

#### B3. Temporal Action Localization

Model chính đề xuất: ActionFormer/OpenTAD hoặc kiến trúc TAL tương đương.

- Input video features.
- Output nhiều interval `(start, end, class, score)`.
- Train trên Fine-Badminton 8 lớp trước, 29 lớp là thí nghiệm mở rộng.

So sánh hai hướng:

```text
Hit detector → cắt clip → classifier
                versus
ActionFormer → interval + class trực tiếp
```

### 5.3. Phase C — end-to-end

Hợp nhất output thành `StrokeEvent` chuẩn, temporal NMS/smoothing, quality gate
và timeline. Đo riêng:

- lỗi localization;
- lỗi player side;
- lỗi stroke class;
- lỗi end-to-end đúng cả thời điểm + người + lớp.

### 5.4. Phase D — chiến thuật (sau cùng)

Khi event stream đủ ổn định mới thêm:

- thống kê stroke và court zone;
- chuỗi 2–4 stroke phổ biến, support/confidence/lift;
- tỷ lệ thắng theo pattern;
- Shot Influence/rally outcome model;
- report có link tới đúng rally/frame.

ShuttleNet/DyMF/LLM/RAG là mở rộng, không phải MVP.

## 6. Protocol đánh giá

### 6.1. Clip classification

- Macro-F1 là metric chính vì lớp mất cân bằng.
- Accuracy/balanced accuracy.
- Precision, recall, F1 từng lớp.
- Confusion matrix.
- Top-2 accuracy chỉ là metric phụ.
- Expected Calibration Error hoặc reliability plot nếu dùng confidence để từ
  chối dự đoán.
- Với head `stroke_side`: Macro-F1 ba lớp, F1 từng lớp và confusion matrix.
- Báo cáo thêm joint accuracy `coarse_label AND stroke_side` chỉ trên sample có
  ground truth cho cả hai; không tính sample Fine-Badminton nhãn `unknown` là
  dự đoán sai.
- Báo cáo theo từng dataset/domain, không chỉ một con số gộp, để phát hiện model
  học nền sân hoặc phong cách quay thay vì động tác tay.

### 6.2. Hit spotting

- Precision/Recall/F1 tại ±2, ±5, ±15 frame.
- Mean/median absolute frame error.
- Số hit thừa và thiếu trên mỗi rally.

### 6.3. Temporal Action Localization

- mAP@tIoU 0.3, 0.5, 0.7.
- Average mAP.
- AP từng lớp.
- Recall proposal theo số proposal/video.

### 6.4. End-to-end

Một event chỉ đúng khi:

```text
tIoU đạt ngưỡng
AND player_side đúng
AND coarse_label đúng
AND stroke_side đúng (chỉ ở benchmark có nhãn)
```

Báo cáo cả kết quả dùng ground-truth boundary và predicted boundary để đo lỗi
lan truyền giữa các tầng.

## 7. Tái sử dụng repository AI_Classifier

### 7.1. Giữ nguyên và mở rộng

| Thành phần | Quyết định |
| --- | --- |
| `src/ai_classifier/api` | Giữ: upload, job store, artifact serving |
| `src/ai_classifier/pose` | Giữ: baseline pose và overlay; mở rộng multi-player |
| `src/ai_classifier/preprocessing` | Giữ: smoothing, interpolation, quality |
| `src/ai_classifier/segmentation` | Giữ làm heuristic baseline; bổ sung hit events/TAL adapter |
| `src/ai_classifier/action_recognition` | Refactor thành interface model-agnostic, giữ ST-GCN++ adapter |
| viewer/export/web | Giữ và đổi timeline từ technique phases sang stroke events |
| `tests` | Giữ; mọi schema/mapping/pipeline mới phải có test |

### 7.2. Đóng băng thành nhánh phụ

| Thành phần | Quyết định |
| --- | --- |
| `biomechanics` | Không xóa; dùng visualization/nghiên cứu phụ, không nằm trong metric chính |
| `error_detection` | Không mở rộng nếu chưa có expert-labelled error dataset |
| `reference_analysis` | Giữ cho demo kỹ thuật cũ |
| `rag` | Chỉ dùng diễn đạt bằng chứng có cấu trúc; không cho tự kết luận chiến thuật |
| checkpoint 2 lớp | Giữ regression/demo cũ, không dùng chứng minh đề tài mới |

### 7.3. Không sửa trực tiếp code vendor

- `external/pyskl`, `external/BST-*`, `external/TrackNetV3` là dependency tham
  khảo.
- Adapter của dự án đặt trong `src/ai_classifier/integrations/` hoặc
  `scripts/`, không tiếp tục vá vendor nếu không cần.
- Dataset, checkpoint và output lớn phải được `.gitignore`; chỉ commit manifest,
  config, checksum và link nguồn.

### 7.4. Cấu trúc đề xuất

```text
src/ai_classifier/
  events/                 # StrokeEvent, Rally, schemas
  datasets/               # label mapping, manifest readers
  localization/           # hit detector, ActionFormer adapter, NMS
  classification/         # ST-GCN++, RGB, BST adapters
  tracking/               # two-player/shuttle/court interfaces
  tactics/                # pattern mining, shot influence (Phase D)
  api/
  pose/
  preprocessing/
  visualization/

configs/
  datasets/fine_badminton.yaml
  datasets/bfmd.yaml
  taxonomy/strokes_8class.yaml
  experiments/

data/
  manifests/
  raw/                    # ignored
  processed/              # ignored
```

Không cần di chuyển toàn bộ code ngay. Refactor theo chiều dọc khi triển khai
từng phase để tránh phá pipeline đang chạy.

## 8. Môi trường chạy

Các dependency PySKL/MMCV, MMPose, TrackNet và OpenTAD dễ xung đột. Không cố
nhét tất cả vào một Python environment.

Đề xuất:

```text
main/API environment       Python 3.11, FastAPI, OpenCV, Ultralytics
pyskl environment          Python 3.10, PySKL/MMCV
localization environment   OpenTAD/MMAction2
tracking environment       TrackNet/MMPose
```

Các model giao tiếp bằng file artifact hoặc subprocess có argument rõ ràng;
về sau mới đóng gói service/container nếu cần.

## 9. Milestone và cổng Go/No-Go

### M0 — Dataset audit

**Trạng thái: hoàn thành audit ban đầu ngày 16/09/2026.**

Kết quả Fine-Badminton 8 lớp:

```text
10 video, 9.817 actions
60 raw labels -> 29 canonical labels -> 8 coarse classes
train: 5.921 (6 matches)
val:   1.980 (2 matches)
test:  1.916 (2 matches)
out-of-bounds/unknown-label issues: 0
```

Mọi lớp đều có mặt ở cả ba split; lớp nhỏ nhất `net_attack` vẫn có 140 train,
55 validation và 41 test samples. Dataset đạt cổng Go cho Phase A. 553 interval
chồng thời gian và 3.755 shared boundaries được giữ nguyên vì complete stroke
motions của hai người có thể overlap; đây không phải lỗi bounds.

Deliverables:

- Manifest Fine-Badminton.
- Mapping raw 60 → canonical 29 → coarse 8.
- Thống kê lớp, trận, duration và split.
- Script xem ngẫu nhiên annotation trên video.

Go khi ít nhất 8 lớp có đủ dữ liệu ở train/val/test theo trận và annotation
đồng bộ video.

Nếu 8 lớp không đạt, cho phép chuyển sang taxonomy chung 7 lớp hoặc taxonomy
tối thiểu 6 lớp. **No-Go** nếu còn dưới 5 lớp có thể đánh giá độc lập; khi đó
phải bổ sung dataset trước khi train model chính.

### M1 — Clip classification

**Trạng thái: infrastructure và smoke test đã chạy.**

- Virtual-clip loader đọc trực tiếp full-match MP4 theo `start/end`.
- Đã xuất 96 contact sheet cân bằng theo split/lớp để review thủ công.
- R(2+1)D-18 pretrained Kinetics đã chạy end-to-end trên smoke subset 8 train
  + 8 validation samples và sinh checkpoint 8 lớp.
- Smoke metric không phải kết quả nghiên cứu; mục đích chỉ là xác nhận decode,
  transform, forward/backward, metric và checkpoint hoạt động.

Deliverables:

- Dataset clip từ ground-truth interval.
- ST-GCN++ baseline.
- RGB baseline.
- Report Macro-F1/confusion/per-match metrics.

Go khi model vượt majority baseline rõ ràng, không có leakage và test theo
trận chạy tái lập được.

### M1b — Forehand/backhand/aroundhead

**Trạng thái: annotation đã audit, media/NPY chưa xác nhận đầy đủ.**

Deliverables:

- Manifest ShuttleSet có `match_id`, `player_id`, `hit_frame`, `raw_label`,
  `coarse_label`, `stroke_side` và cờ `flaw`.
- YAML mapping shot type ShuttleSet sang taxonomy chung; nhãn không chắc chắn
  phải là `unknown`, không ép vào một trong tám lớp.
- Baseline ShuttleSet-only ba lớp `forehand/backhand/aroundhead`.
- Model joint multi-task có shared backbone, stroke head và stroke-side head.
- Report theo trận, vận động viên, shot type và domain; có ablation so với model
  Fine-Badminton-only.

Go khi media/NPY khớp annotation bằng kiểm tra tự động, ba lớp có mặt trong
train/validation/test theo trận, và Macro-F1 stroke-side vượt majority baseline
trên test chưa thấy. No-Go cho joint training nếu chỉ có annotation mà không có
input video/feature tương ứng, hoặc split làm cùng trận xuất hiện ở nhiều tập.

### M2 — Hit/localization

**Trạng thái ngày 17/09/2026: đã có motion candidate-frame pilot, chưa phải
full-video spotting.**

- Baseline ba lớp `no_hit/upper_hit/lower_hit` dùng chuyển động theo lưới quanh
  candidate frame và Random Forest.
- Pilot tách theo trận (train 13/14/15, validation 35, test 44), 50 positive mỗi
  trận: validation Macro-F1 `58,92%`, test Macro-F1 `68,52%`, test accuracy
  `75,20%`.
- Test pilot chỉ dùng candidate frame và negative sampling. Không được báo cáo
  như precision/recall hit spotting trên toàn bộ video.
- Bước kế tiếp là quét video tuần tự, temporal NMS và đánh giá event matching
  tại tolerance ±2/±5/±15 frame; sau đó mới quyết định giữ motion baseline hay
  thay bằng pose/RGB temporal model.

Deliverables:

- Heuristic hit baseline.
- Hit detector hoặc ActionFormer baseline.
- Metric tolerance/tIoU.

Go khi hit recall/precision đủ để tạo timeline có thể kiểm tra; threshold được
chọn trên validation, không trên test.

### M3 — End-to-end upload

Deliverables:

- Upload rally MP4.
- Stroke event JSON.
- Timeline trên web.
- Quality gate và trạng thái `insufficient_data`.
- Báo cáo predicted-boundary so với oracle-boundary.

### M4 — Tactical analysis tùy chọn

Deliverables:

- Rally sequence.
- Pattern statistics.
- Shot influence prototype.
- Mỗi insight có video/frame evidence.

## 10. Công việc tiếp theo theo thứ tự

1. Giữ R(2+1)D-18 Fine-Badminton hiện tại làm baseline tám lớp; đánh giá trên
   test split theo trận và các trận broadcast external, không chọn model bằng
   external test.
2. Xác minh và giải nén media/NPY ShuttleSet; lập checksum và kiểm tra mỗi row
   annotation có input tương ứng quanh `hit_frame`.
3. Viết converter ShuttleSet sang manifest thống nhất, parse ô trống/`1.0`, loại
   `flaw` và dòng xung đột; thêm unit test cho các trường hợp này.
4. Tạo YAML mapping 18 shot type ShuttleSet sang taxonomy chung, giữ
   `raw_label`; review thủ công các mapping mơ hồ.
5. Tạo split theo trận, kiểm tra overlap vận động viên và báo cáo phân bố
   `stroke_side x coarse_label` cho từng split.
6. Train baseline ShuttleSet-only ba lớp `forehand/backhand/aroundhead` trên
   cùng loại input dự kiến dùng khi deploy.
7. Thêm stroke-side head vào RGB backbone và train joint multi-task với masked
   loss, source-balanced sampler; so sánh với hai baseline độc lập.
8. Train ST-GCN++ tám lớp và/hoặc stroke-side để đo giá trị của pose so với RGB.
9. Chốt Phase A dựa trên Macro-F1 theo trận, joint accuracy và kiểm tra
   cross-domain; sau đó mới triển khai hit detector/ActionFormer.
10. Tải/audit BFMD khi cần full-match external validation; chỉ bắt đầu Shot
    Influence khi đã có event timeline hoặc dùng ground-truth event với ghi chú
    rõ ràng.

## 11. Câu mô tả đề tài dùng với mentor

> Nhóm giữ lại nền tảng upload, pose, xử lý video và viewer đã xây dựng, nhưng
> chuyển trọng tâm nghiên cứu sang nhận diện cú đánh trong trận đơn. Trước tiên
> nhóm sẽ dùng ground-truth interval của Fine-Badminton để xây và đánh giá bộ
> phân loại tám nhóm cú đánh theo split từng trận. ShuttleSet được dùng để bổ
> sung head nhận diện forehand/backhand/aroundhead bằng học multi-task, không
> nhân tám lớp thành các tổ hợp thưa dữ liệu. Sau đó nhóm huấn luyện mô
> hình định vị thời gian để biến video rally chưa cắt thành stroke timeline.
> Phân tích chiến thuật chỉ được xây trên event stream đã đánh giá, và mọi nhận
> xét đều liên kết tới bằng chứng video.

## 12. Tiêu chí hoàn thành đề tài

Đề tài được coi là hoàn thành khi:

- Dataset, taxonomy và split có thể tái tạo bằng script.
- Có ít nhất hai baseline clip classification trên test theo trận.
- Kết quả chính có ít nhất 5 lớp động tác; mục tiêu là 7–8 lớp.
- Nếu công bố forehand/backhand, có benchmark ShuttleSet tách theo trận, metric
  riêng cho ba lớp stroke-side và joint metric với loại cú đánh.
- Có model/heuristic tạo nhiều stroke events từ một rally chưa cắt.
- Có metric localization, classification và end-to-end độc lập.
- Web nhận video hợp lệ và hiển thị timeline có thể nhảy tới từng event.
- Hệ thống từ chối hoặc cảnh báo input ngoài phạm vi.
- Không dùng kết quả validation như test và không tuyên bố khả năng trên video
  điện thoại/đánh đôi khi chưa đánh giá.
