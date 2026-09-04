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
