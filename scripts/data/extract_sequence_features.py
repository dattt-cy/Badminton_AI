"""Extract structured sequence features [T, D] aligned with contact frame for CR-Gated Fusion."""

from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_classifier.features import modality_quality_vector, resample


def target_indices(rows: list[dict], fps: int = 30) -> dict[str, int]:
    """Locate each contact inside its between-hit source sequence."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["match_id"], row["set_id"], row["rally"])].append(row)

    output: dict[str, int] = {}
    for group in grouped.values():
        ordered = sorted(group, key=lambda row: int(row["ball_round"]))
        for index, row in enumerate(ordered):
            current = int(float(row["hit_frame"]))
            previous = (
                int(float(ordered[index - 1]["hit_frame"]))
                if index
                else current - fps // 2
            )
            start = max(previous, current - fps * 3 // 2)
            output[row["sample_id"]] = current - start
    return output


def extract_sample_sequences(
    joints: np.ndarray,
    position: np.ndarray,
    shuttle: np.ndarray,
    target_index: int,
    length: int = 32,
    before: int = 15,
    after: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned sequences for pose, court, shuttle, contact_dist and compute quality metrics."""
    window_length = before + 1 + after
    safe_target = min(max(int(target_index), 0), max(len(shuttle) - 1, 0))

    # Slice or pad each array into window [-before, +after]
    windowed = []
    for array in (joints, position, shuttle):
        safe_t = min(max(int(target_index), 0), max(len(array) - 1, 0))
        output = np.zeros((window_length, *array.shape[1:]), dtype=np.float32)
        source_start = max(0, safe_t - before)
        source_end = min(len(array), safe_t + after + 1)
        dest_start = before - min(before, safe_t)
        span = min(source_end - source_start, window_length - dest_start)
        if span > 0:
            output[dest_start:dest_start + span] = array[source_start:source_start + span]
        windowed.append(output)

    w_joints, w_pos, w_shuttle = windowed

    # 1. Quality vector q (4 metrics)
    quality = modality_quality_vector(w_joints, w_shuttle)

    # 2. Resample to target length (32)
    s_joints = resample(w_joints, length).reshape(length, -1).astype(np.float32)  # (32, 68)
    s_pos = resample(w_pos, length).reshape(length, -1).astype(np.float32)        # (32, 4)
    s_shuttle = resample(w_shuttle, length).reshape(length, -1).astype(np.float32) # (32, 2)

    # 3. Contact distance array: signed distance from contact frame (before=15)
    raw_dist = np.linspace(-before, after, length, dtype=np.float32)[:, None] / float(before)
    s_contact = raw_dist  # (32, 1)

    return s_joints, s_pos, s_shuttle, s_contact, quality


def main():
    manifest_path = Path("data/manifests/shuttleset_npy.csv")
    npy_root = Path(r"C:\Users\ADMIN\Downloads\dataset_npy_between_2_hits_with_max_limits\dataset_npy_between_2_hits_with_max_limits")
    source_cache = Path("work_dirs/shuttleset_fusion_epoch20_full_b4/features.npz")
    output_dir = Path("work_dirs/cr_gated_fusion")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sequence_features.npz"

    print("=" * 65)
    print("TRICH XUAT CHUOI DAC TRUNG DA PHUONG THUC CHO CR-GATED FUSION")
    print(f"Nguon manifest: {manifest_path}")
    print(f"Nguon RGB:      {source_cache}")
    print(f"Dich xuat:      {output_path}")
    print("=" * 65)

    print("[1/4] Dang nap RGB features va metadata tu cache cu...")
    source_data = np.load(source_cache)
    sample_ids = [str(sid) for sid in source_data["sample_id"]]
    sample_id_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}
    N = len(sample_ids)
    print(f"  -> Tim thay {N} mau da co san RGB embedding (512-D).")

    print("[2/4] Dang nap thong tin hit_frame tu manifest...")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        all_rows = [row for row in csv.DictReader(f) if row["sample_id"] in sample_id_to_idx]
    
    row_by_id = {row["sample_id"]: row for row in all_rows}
    target_idx_map = target_indices(all_rows)
    print(f"  -> Da map xong {len(row_by_id)} mau hop le.")

    print("[3/4] Dang trích xuat chuoi thoi gian [T=32] cho tung modality...")
    length = 32
    pose_arr = np.zeros((N, length, 68), dtype=np.float32)
    court_arr = np.zeros((N, length, 4), dtype=np.float32)
    shuttle_arr = np.zeros((N, length, 2), dtype=np.float32)
    contact_arr = np.zeros((N, length, 1), dtype=np.float32)
    quality_arr = np.zeros((N, 4), dtype=np.float32)
    valid_arr = np.zeros(N, dtype=np.bool_)

    t0 = time.time()
    extracted = 0
    for idx, sid in enumerate(sample_ids):
        row = row_by_id.get(sid)
        if row is None:
            continue
        try:
            joints = np.load(npy_root / row["joints_path"])
            pos = np.load(npy_root / row["position_path"])
            shuttle = np.load(npy_root / row["shuttle_path"])
            t_idx = target_idx_map.get(sid, 15)

            p_seq, c_seq, s_seq, cont_seq, q_vec = extract_sample_sequences(
                joints, pos, shuttle, t_idx, length=length
            )
            pose_arr[idx] = p_seq
            court_arr[idx] = c_seq
            shuttle_arr[idx] = s_seq
            contact_arr[idx] = cont_seq
            quality_arr[idx] = q_vec
            valid_arr[idx] = True
            extracted += 1
        except Exception as e:
            print(f"  [WARN] Khong trich xuat duoc sample {sid}: {e}", flush=True)

        if (idx + 1) % 5000 == 0 or (idx + 1) == N:
            elapsed = time.time() - t0
            print(f"  -> Da xu ly: {idx + 1}/{N} mau ({elapsed:.1f}s, {(idx + 1)/elapsed:.1f} mau/s)")

    print(f"[4/4] Dang luu sequence_features.npz (Tong cong {extracted} mau)...")
    np.savez_compressed(
        output_path,
        rgb=source_data["rgb"],
        pose=pose_arr,
        court=court_arr,
        shuttle=shuttle_arr,
        contact_dist=contact_arr,
        quality=quality_arr,
        valid=valid_arr,
        stroke=source_data["stroke"],
        side=source_data["side"],
        split=source_data["split"],
        sample_id=source_data["sample_id"],
    )
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"HOAN TAT! Da luu thanh cong file: {output_path} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
