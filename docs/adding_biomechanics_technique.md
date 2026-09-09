# Thêm kỹ thuật và clip mẫu hình học

Nguồn cấu hình chung là `configs/biomechanics/techniques.yaml`. Script build
reference và script đánh giá đều đọc file này; không thêm tên kỹ thuật vào
Python.

## Thêm clip mẫu cho kỹ thuật đã có

1. Trích pose `.npz` vào thư mục `pose_dir` của kỹ thuật.
2. Kiểm tra trực quan clip và thêm **tên file** vào đúng danh sách
   `clip_list` (`front`, `side`, hoặc `generic`). Không dùng clip test của
   người dùng làm mẫu.
3. Build lại đúng góc quay:

```bash
python scripts/data/build_reference_profiles.py forehand_lift --view side
python scripts/data/build_reference_profiles.py forehand_lift --view front
```

Quality gate vẫn tự loại pose quá nhỏ, thiếu keypoint, spike lớn hoặc phase
không hợp lệ. `clips_used` trong reference cho biết chính xác clip nào thực sự
được dùng; việc có tên trong `clip_list` không đảm bảo clip sẽ vượt quality
gate.

## Thêm một kỹ thuật mới

Ví dụ thêm `smash`:

1. Tạo thư mục pose và các danh sách clip đã duyệt theo góc quay.
2. Thêm khóa `smash` vào `techniques.yaml`, gồm `pose_dir`, `views` và
   `rules`.
3. Mỗi rule khai báo `feature`, `phase`, `direction`, ngưỡng confidence/số
   frame và góc quay tương thích.
4. Build từng reference rồi chạy đánh giá:

```bash
python scripts/data/build_reference_profiles.py smash --view side
python scripts/evaluation/check_technique_rules.py outputs/smash_pose.npz smash --view side
```

Các feature hiện có có thể dùng ngay, ví dụ `elbow_angle`, `elbow_height`,
`wrist_shoulder_distance`, `stance_width`. Nếu kỹ thuật cần đại lượng hình học
mới hoặc cách chia phase khác, phải bổ sung phép đo Python và test trước khi
khai báo rule trong registry.

Một kỹ thuật nên có rule phủ các phase quan trọng thay vì chỉ một giá trị cho
cả động tác. `backhand_drive` hiện minh họa các nhóm preparation, backswing,
forward swing, contact và follow-through. Rule `two_sided` phát hiện tư thế
nằm ngoài cả hai phía của phân bố chuyên gia; `lower`/`upper` chỉ phát hiện
đúng phía mang ý nghĩa lỗi. Mọi range phải được build từ curated clips, không
nhập một khoảng rộng chỉ để video test vượt qua.

Classifier nhận dạng động tác vẫn là mô hình riêng: thêm registry không tự
thêm lớp mới cho ST-GCN++. Muốn classifier nhận ra kỹ thuật mới còn phải bổ
sung dữ liệu, nhãn và train lại model.
