# Reference profile audit

## Scope

Reviewed the four-frame contact sheets for all 65 `forehand_lift` and 56
`backhand_drive` single-player videos. A clip is eligible only when pose
availability is at least 90%, median body scale is at least 30 px, scale CV is
at most 0.25, wrist-speed spike ratio is at most 3.0, and detected phases are
plausible.

The downloaded test video `videoplayback (2) (1).mp4` is byte-identical to
`forehand_lift (76).mp4` and is excluded from every reference profile to avoid
test/reference leakage.

## Profiles

| Technique | Projected view | Samples | Curated list | Reference |
|---|---|---:|---|---|
| Forehand lift | side-on | 5 | `forehand_lift_side_curated_clips.txt` | `forehand_lift_side_reference.yaml` |
| Forehand lift | front | 5 | `forehand_lift_front_curated_clips.txt` | `forehand_lift_front_reference.yaml` |
| Backhand drive | front | 12 | `backhand_drive_front_curated_clips.txt` | `backhand_drive_front_reference.yaml` |
| Backhand drive | side/oblique | 6 | `backhand_drive_side_curated_clips.txt` | `backhand_drive_side_reference.yaml` |

The evaluation CLI selects `front` or `side` automatically from the median
projected shoulder-width/torso-length ratio. `--view` can override this when
the automatic projection label is visibly wrong.

## Exclusions

- Very distant players and clips dominated by empty court.
- Incomplete phases, unstable body scale, or wrist-speed spikes.
- `005_backhand_dive5_05` through `005_backhand_dive5_10`: labelled in-video
  as receiving-shuttle demonstrations, not used as reference backhand drives.
- Mixed camera views are not pooled into the view-specific envelopes.

## Interpretation

`stance_too_narrow` is a one-sided 2D stance-width comparison. It does not
claim that the player's physical balance is unstable. Kinetic-chain timing is
diagnostic only and remains excluded from MVP error rules.

The forehand view profiles are provisional because each has only five clean,
homogeneous clips. Add expert-confirmed clips from the same views before using
them for coaching-grade conclusions.
