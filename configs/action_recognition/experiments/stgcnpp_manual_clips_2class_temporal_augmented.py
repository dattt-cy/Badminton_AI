"""ST-GCN++ for original manual clips plus isolated start-at-swing variants."""

_base_ = "./stgcnpp_manual_clips_2class.py"

ann_file = "data/annotations/manual_clips_2class_temporal_augmented.pkl"

train_pipeline = [
    dict(type="BadmintonRandomRot2D", theta=0.12),
    dict(type="RandomScale", scale=0.10),
    dict(type="BadmintonEdgePad", min_frames=64),
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSample", clip_len=64),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]

val_pipeline = [
    dict(type="BadmintonEdgePad", min_frames=64),
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSample", clip_len=64, num_clips=1),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]

test_pipeline = [
    dict(type="BadmintonEdgePad", min_frames=64),
    dict(type="PreNormalize2D"),
    dict(type="GenSkeFeat", dataset="coco", feats=["j"]),
    dict(type="UniformSample", clip_len=64, num_clips=10),
    dict(type="PoseDecode"),
    dict(type="FormatGCNInput", num_person=1),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]

data = dict(
    train=dict(dataset=dict(ann_file=ann_file, pipeline=train_pipeline)),
    val=dict(ann_file=ann_file, pipeline=val_pipeline),
    # This dataset has a held-out validation split but no separate test split.
    test=dict(ann_file=ann_file, pipeline=test_pipeline, split="val"),
)

work_dir = "./work_dirs/stgcnpp_manual_clips_2class_temporal_augmented"
