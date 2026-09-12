"""Evaluate the manual two-class checkpoint on the subject-separated val set."""

_base_ = "./stgcnpp_manual_clips_2class.py"

data = dict(test=dict(split="val"))
