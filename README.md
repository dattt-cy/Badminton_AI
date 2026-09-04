# AI Classifier

Pipeline AI phan tich ky thuat cau long tu video:

1. Pose estimation: trich xuat keypoint tu video.
2. Preprocessing: tracking, smoothing va chuan hoa chuoi keypoint.
3. Action recognition: ST-GCN/ST-GCN++ phan loai dong tac.
4. Biomechanics: tinh goc khop va dac trung chuyen dong.
5. Error detection: phat hien loi theo tung dong tac va tung pha.
6. RAG: truy xuat tai lieu va tao huong dan khac phuc.

## Cau truc

```text
configs/                 Cau hinh cho tung thanh phan
data/                    Du lieu tho, nhan va du lieu da xu ly
docs/                    Tai lieu thiet ke va quy uoc du lieu
external/pyskl/          Ma nguon PySKL doc lap
models/                  Checkpoint va model export
notebooks/               Thu nghiem, khao sat du lieu
outputs/                 Ket qua inference, log va visualization
scripts/                 Lenh chay theo tung cong doan
src/ai_classifier/       Ma nguon chinh cua he thong AI
tests/                   Kiem thu
```

PySKL trong `external/pyskl` la dependency tham khao/baseline. Code du an
khong nen sua truc tiep trong thu muc nay neu khong that su can thiet.

## Trich xuat pose bang YOLOv8

Cai dat project va dependency:

```bash
python -m pip install -e .
```

Chay pose estimation tren video:

```bash
python scripts/inference/extract_pose.py data/raw/sample.mp4 outputs/sample_pose.npz
```

Lan chay dau tien, Ultralytics se tai checkpoint `yolov8n-pose.pt`. File dau
ra chua tensor `keypoints` co shape `(T, 17, 3)` theo thu tu `(x, y,
confidence)` va metadata gom FPS, chieu rong, chieu cao video. Neu mot frame
khong phat hien duoc nguoi, pose cua frame do duoc dien bang 0.

Pipeline mac dinh theo doi nguoi gan pose cua frame truoc, noi suy cac diem
bi mat trong toi da 4 frame va lam muot toa do theo thoi gian. Co the dieu
chinh tracking/smoothing trong `configs/pose/yolov8.yaml`.

Tao video preview co skeleton:

```bash
python scripts/inference/render_pose.py data/raw/sample.mp4 outputs/sample_pose.npz outputs/sample_pose_preview.mp4
```
