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

## Chuan bi dataset ST-GCN

Trich xuat skeleton cho toan bo dataset. Video `match` tu dong chon nguoi o
san xa; video `single_player` chon nguoi duy nhat:

```bash
python scripts/data/extract_dataset_poses.py
```

Lenh co the chay lai de tiep tuc vi cac file da xu ly se duoc bo qua. Sau khi
extract xong, xuat annotation dung format PySKL:

```bash
python scripts/data/export_pyskl.py
```

Xoay va scale du lieu duoc cau hinh de augmentation skeleton trong luc train,
khong nhan ban video vat ly. Cau hinh dataset nam tai
`configs/action_recognition/dataset.yaml`.

Config train ST-GCN++ nam tai
`configs/action_recognition/stgcnpp_badminton.py`. Pipeline train tu dong tao
rotation khoang +/-6.9 do va scale +/-10% moi epoch; validation va test khong
augmentation.

Vi cac clip `match` hien tai deu duoc cat tu cung mot tran, split duoc tao theo
cac block lien tiep va chi phu hop lam baseline thu nghiem. File
`data/annotations/badminton_actions.csv` ghi `source_group` de theo doi nguy
co ro ri du lieu. Can them tran hoac nguoi choi moi truoc khi danh gia chinh
thuc.

Train tren GPU Google Colab bang notebook
`notebooks/train_stgcnpp_colab.ipynb`. Notebook tao moi truong PySKL Python
3.10 rieng tren Colab; chi can upload `badminton_actions.pkl`, khong can upload
video tho.

## Nhan dien dong tac

Dat checkpoint tot nhat tai:

```text
models/checkpoints/action_recognition/best_top1_acc_epoch_6.pth
```

Nhan dien truc tiep tu video tran dau (nguoi o san xa):

```bash
python scripts/inference/classify_action.py path/to/video.mp4 \
  --target far \
  --pose-output outputs/video_pose.npz \
  --json-output outputs/video_prediction.json
```

Voi video chi co mot nguoi, dung `--target single`. Neu da co pose NPZ thi co
the bo qua YOLOv8 va chay nhanh hon:

```bash
python scripts/inference/classify_action.py outputs/video_pose.npz
```

Ket qua gom nhan `backhand_drive` hoac `forehand_lift`, confidence, xac suat
tung lop va thong so chat luong pose. ST-GCN++ phai chay trong moi truong
Python 3.10 co PySKL/MMCV tuong thich nhu notebook Colab; neu chay tu video,
moi truong do cung can cai Ultralytics.
