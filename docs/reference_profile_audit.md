# Reference profile audit

## Scope

The original audit reviewed the contact sheets for all 65 `forehand_lift` and
56 legacy `backhand_drive` single-player videos. The active backhand and
forehand-clear profiles now also include cropped MultiSense expert references
and visually reviewed Expert/train manual clips. A clip is eligible only when pose
availability is at least 90%, median body scale is at least 30 px, scale CV is
at most 0.25, wrist-speed spike ratio is at most 3.0, and detected phases are
plausible.

The downloaded test video `videoplayback (2) (1).mp4` is byte-identical to
`forehand_lift (76).mp4` and is excluded from every reference profile to avoid
test/reference leakage.

## Profiles

| Technique | Projected view | Samples | Curated list | Reference |
|---|---|---:|---|---|
| Forehand lift | generic | 9 | `forehand_lift/forehand_lift_curated_clips.txt` | `forehand_lift/forehand_lift_reference.yaml` |
| Forehand lift | side-on | 4 | `forehand_lift/forehand_lift_side_curated_clips.txt` | `forehand_lift/forehand_lift_side_reference.yaml` |
| Forehand lift | front | 4 | `forehand_lift/forehand_lift_front_curated_clips.txt` | `forehand_lift/forehand_lift_front_reference.yaml` |
| Backhand drive | generic | 5 | `backhand_drive/backhand_drive_curated_clips.txt` | `backhand_drive/backhand_drive_reference.yaml` |
| Backhand drive | front | 13 | `backhand_drive/backhand_drive_multisense_front_curated_clips.txt` | `backhand_drive/backhand_drive_multisense_front_reference.yaml` |
| Backhand drive | side/oblique | 21 | `backhand_drive/backhand_drive_combined_side_curated_clips.txt` | `backhand_drive/backhand_drive_combined_side_reference.yaml` |
| Forehand clear | front | 16 | `forehand_clear/forehand_clear_front_curated_clips.txt` | `forehand_clear/forehand_clear_front_reference.yaml` |
| Forehand clear | side/oblique | 16 | `forehand_clear/forehand_clear_combined_side_curated_clips.txt` | `forehand_clear/forehand_clear_combined_side_reference.yaml` |

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

All active profiles remain provisional. Sample count alone does not make a
profile coaching-grade; contact frames and rule outcomes still need independent
expert validation.

## Runtime evaluation policy

- Short uploads up to five seconds with at most one motion proposal use the
  full pose sequence so preparation and follow-through are not cut away.
- Camera view is supplied explicitly for the whole clip. Projected body
  orientation is measured per phase only to mark rules as review or
  insufficient when the athlete rotates out of the observable 2D plane.
- `forehand_clear` wrist-height rules are temporarily excluded because the
  current 2D reference envelopes conflict with visually valid overhead contact.
