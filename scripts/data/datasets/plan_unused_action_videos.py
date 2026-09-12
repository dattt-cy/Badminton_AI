"""Plan cleanup without removing training or geometric-reference inputs."""
import json
import pickle
from pathlib import Path


def main():
    root = Path.cwd().resolve()
    artifact = root / 'data/annotations/multisense_actions_3class_legacy_augmented.pkl'
    with artifact.open('rb') as source:
        dataset = pickle.load(source)
    keep = set()
    for item in dataset['annotations']:
        identifier = item['frame_dir']
        if '/LegacyTrain/' in identifier:
            name = Path(identifier).name.removeprefix('legacy_')
            keep.add((root / 'data/raw/backhand_drive/single_player' / (name + '.mp4')).resolve())
    # Legacy curated poses and reference YAMLs still depend on their raw videos.
    for path in (root / 'configs/biomechanics').glob('*curated_clips.txt'):
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                name = Path(line.strip()).stem.removesuffix('_pose') + '.mp4'
                candidate = root / 'data/raw/backhand_drive/single_player' / name
                if candidate.exists():
                    keep.add(candidate.resolve())
    candidates = []
    for folder in ('data/raw/forehand_clear', 'data/raw/backhand_drive/single_player'):
        for path in sorted((root / folder).rglob('*.mp4')):
            if path.resolve() not in keep:
                candidates.append({'path': str(path.resolve()),
                                   'relative': path.relative_to(root).as_posix(),
                                   'bytes': path.stat().st_size})
    report = {'artifact': str(artifact), 'kept_raw_backhand': len(keep),
              'move_candidates': candidates,
              'total_bytes': sum(item['bytes'] for item in candidates)}
    output = root / 'outputs/unused_action_videos_cleanup.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'{len(candidates)} unused videos; {report["total_bytes"]/1024**2:.2f} MiB; protected raw backhand={len(keep)}')


if __name__ == '__main__':
    main()
