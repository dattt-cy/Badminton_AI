"""Three-class ST-GCN++ config with an explicit other/no-action class."""

_base_ = "./stgcnpp_badminton.py"

model = dict(
    cls_head=dict(type="GCNHead", num_classes=3, in_channels=256),
)
work_dir = "./work_dirs/stgcnpp_badminton_3class"
