# AI Classifier

Pipeline AI phan tich ky thuat cau long tu video:

1. Pose estimation: trich xuat keypoint tu video.
2. Preprocessing: tracking, smoothing va chuan hoa chuoi keypoint.
3. Action recognition: ST-GCN/ST-GCN++ phan loai dong tac.
4. Biomechanics: tinh goc khop va dac trung chuyen dong.
5. Error detection: phat hien loi theo tung dong tac va tung pha.
6. RAG: truy xuat tai lieu va tao huong dan khac phuc.

## Cau truc du an

```text
configs/
  action_recognition/
    datasets/            Cau hinh tao tung bo du lieu
    experiments/         Cau hinh train/evaluate ST-GCN++
  biomechanics/          Registry, rule va reference theo ky thuat
  pose/                   Preset trich xuat pose
data/                     Du lieu theo vong doi raw -> interim -> processed
docs/                     Thiet ke, quy uoc va huong dan mo rong
external/pyskl/           Git submodule cua dependency PySKL
models/checkpoints/       Trong so pose va action recognition
models/exports/           Model da dong goi de deploy
notebooks/                Thu nghiem va Colab entry point
outputs/                  Ket qua inference, log va visualization (generated)
scripts/
  data/augmentation/      Tao bien the va materialize mau
  data/datasets/          Xay dung, migrate va kiem ke dataset
  data/export/            Xuat PySKL va feature table
  data/pose/              Trich pose hang loat
  data/references/        Chon mau va xay reference profile
  evaluation/             Audit va cham ky thuat
  inference/              CLI suy luan
  training/               Wrapper huan luyen
src/ai_classifier/        Ma nguon tai su dung, chia theo domain
tests/unit/               Unit test mirror domain trong src va scripts
```

PySKL trong `external/pyskl` la dependency tham khao/baseline. Code du an
khong nen sua truc tiep trong thu muc nay neu khong that su can thiet.
Quy uoc chi tiet nam trong `docs/project_structure.md`; danh muc cac lenh chay
nam trong `scripts/README.md`.

## Trich xuat pose bang YOLOv8

Cai dat project va dependency:

```bash
python -m pip install -e .
```

Chay pose estimation tren video:

```bash
python scripts/inference/extract_pose.py data/raw/sample.mp4 outputs/sample_pose.npz
```

Checkpoint YOLO duoc luu trong `models/checkpoints/pose/`, khong dat o root.
File dau ra chua tensor `keypoints` co shape `(T, 17, 3)` theo thu tu `(x, y,
confidence)` va metadata gom FPS, chieu rong, chieu cao video. Neu mot frame
khong phat hien duoc nguoi, pose cua frame do duoc dien bang 0.

Pipeline mac dinh theo doi nguoi gan pose cua frame truoc, noi suy cac diem
bi mat trong toi da 4 frame va lam muot toa do theo thoi gian. Co the dieu
chinh tracking/smoothing trong `configs/pose/yolov8.yaml`.

Tao video preview co skeleton:

```bash
python scripts/inference/render_pose.py data/raw/sample.mp4 outputs/sample_pose.npz outputs/sample_pose_preview.mp4
```

Khi trich pose de xay reference hoac cham diem hinh hoc, co the dung preset do
chinh xac cao. Reference va video can cham phai duoc trich bang cung model va
cung cau hinh smoothing de tranh distribution shift:

```bash
python scripts/inference/extract_pose.py data/raw/sample.mp4 outputs/sample_pose.npz \
  --config configs/pose/yolov8_high_accuracy.yaml
```

Neu camera bi nghieng, do goc cua mot duong san nam ngang trong anh (duong
doc anh huong xuong ben phai la goc duong) va truyen cung quy uoc khi xay
reference va khi cham video. Vi du:

```bash
python scripts/data/references/build_reference_profiles.py forehand_lift --view front \
  --floor-angle-degrees 3.5
python scripts/evaluation/check_technique_rules.py outputs/sample_pose.npz \
  forehand_lift --floor-angle-degrees 3.5
```

Khong doi preprocessing cua checkpoint action-recognition neu chua train lai,
vi thay doi dau vao co the tao distribution shift.

Khi cham hinh hoc, bat buoc chon mot camera view co dinh cho ca clip. He thong
khong suy camera view tu do rong vai, vi van dong vien co the xoay than trong
luc danh. Co the dung frame swing peak da gan nhan thu cong cho clip mot stroke:

```bash
python scripts/evaluation/check_technique_rules.py outputs/sample_pose.npz \
  forehand_clear --view front --swing-peak-frame 46 \
  --output outputs/sample_geometry.json
```

Bao cao gom elbow-extension delta, wrist path/excursion, balance proxy, ty le
thoi gian tung pha va cac nhom `strengths`, `needs_review`,
`insufficient_data`. De danh gia tren bo nhan thu cong, sao che
`configs/biomechanics/geometry_validation_template.csv` va chay:

```bash
python scripts/evaluation/validate_geometry_reports.py labels.csv \
  --output outputs/geometry_validation.json
```

Sau khi them cac chi so tong hop, co the cap nhat rieng vung tham chieu moi ma
khong thay doi cac nguong rule da duoc duyet:

```bash
python scripts/data/references/build_reference_profiles.py forehand_clear \
  --view side --summary-only
```

## Chuan bi dataset ST-GCN

Trich xuat skeleton cho toan bo dataset. Video `match` tu dong chon nguoi o
san xa; video `single_player` chon nguoi duy nhat:

```bash
python scripts/data/pose/extract_dataset_poses.py
```

Lenh co the chay lai de tiep tuc vi cac file da xu ly se duoc bo qua. Sau khi
extract xong, xuat annotation dung format PySKL:

```bash
python scripts/data/export/export_pyskl.py
```

Xoay va scale du lieu duoc cau hinh de augmentation skeleton trong luc train,
khong nhan ban video vat ly. Cau hinh dataset nam tai
`configs/action_recognition/datasets/dataset.yaml`.

Config train ST-GCN++ 2 lop nam tai
`configs/action_recognition/experiments/stgcnpp_badminton.py`. Hai lop gom
`backhand_drive` va `forehand_lift`. Pipeline train tu dong tao
rotation khoang +/-6.9 do va scale +/-10% moi epoch; validation va test khong
augmentation.

Vi cac clip `match` hien tai deu duoc cat tu cung mot tran va tung lam lech
phan bo du lieu, cau hinh mac dinh chi xuat cac nguon `single_player`. Cac clip
match van duoc giu lai de bo sung sau khi co nhieu tran doc lap. File
`data/annotations/badminton_actions.csv` ghi `source_group` de theo doi nguy
co ro ri du lieu. Bo du lieu sach hien co 83 train, 17 validation va 19 test.

Co the kiem tra pipeline co hoc dung nhan bang bo sanity 10 mau:

```bash
python scripts/data/datasets/create_overfit_subset.py \
  data/annotations/badminton_actions_2class.pkl \
  data/annotations/badminton_overfit_10.pkl
```

Train tren GPU Google Colab bang notebook
`notebooks/train_stgcnpp_2class_colab.ipynb`. Notebook tao moi truong PySKL
Python 3.10 rieng tren Colab; upload `badminton_actions_2class.pkl`, khong can
upload video tho.

## Nhan dien dong tac

Dat checkpoint tot nhat tai:

```text
models/checkpoints/action_recognition/best_top1_acc_2class_clean.pth
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

Ket qua gom `backhand_drive` hoac `forehand_lift`, confidence, xac suat tung
lop va thong so chat luong pose.
ST-GCN++ phai chay trong moi truong
Python 3.10 co PySKL/MMCV tuong thich nhu notebook Colab; neu chay tu video,
moi truong do cung can cai Ultralytics.

### Chay inference local bang WSL2 + NVIDIA GPU

May da cai moi truong `badminton-pyskl` trong WSL2 Ubuntu. Tu PowerShell tai
thu muc du an, chay:

```powershell
.\scripts\inference\classify_action_wsl.ps1 "data\inference\video.mp4"
```

Video mot nguoi mac dinh dung `-Target single`. Voi nguoi o nua san xa:

```powershell
.\scripts\inference\classify_action_wsl.ps1 "data\inference\match.mp4" -Target far
```

Ket qua JSON va pose NPZ duoc luu trong `outputs/`. Wrapper tu dong dung Python
3.10, PySKL, MMCV va CUDA cua WSL; khong can activate Conda thu cong.
