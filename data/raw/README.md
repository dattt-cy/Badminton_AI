# Raw video dataset

Copy each clip into the matching action and recording type:

```text
data/raw/
|-- backhand_drive/
|   |-- single_player/
|   `-- match/
|-- forehand_lift/
|   |-- single_player/
|   `-- match/
```

Use one complete action per clip. Recommended filenames:

```text
player01_backhand_drive_001.mp4
player01_forehand_lift_001.mp4
match01_forehand_lift_far_001.mp4
match01_backhand_drive_near_001.mp4
```

The current classifier uses only `backhand_drive` and `forehand_lift`.
Previously collected `other` videos are retained locally but excluded from
pose export and training annotations.

For match clips, include `near`, `far`, `left`, or `right` in the filename to
identify the target player. Raw videos are ignored by Git.

For the current dataset, the labeled player in every `match` clip is on the
far side of the court. The batch configuration therefore uses `target: far`.
