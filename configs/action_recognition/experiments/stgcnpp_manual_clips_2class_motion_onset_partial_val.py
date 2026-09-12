"""Evaluate a motion-onset checkpoint on partial validation clips only."""

_base_ = "./stgcnpp_manual_clips_2class_motion_onset.py"

data = dict(
    test=dict(
        ann_file="data/annotations/manual_clips_2class_motion_onset_val.pkl",
        split="val",
    )
)
