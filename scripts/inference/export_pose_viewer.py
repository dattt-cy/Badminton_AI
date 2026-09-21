"""Export a pose NPZ as compact JSON for the interactive 3D viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_classifier.pose.viewer_export import export_pose_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = export_pose_viewer(args.pose, args.output)
    print(f"Viewer data: {result}")


if __name__ == "__main__":
    main()
