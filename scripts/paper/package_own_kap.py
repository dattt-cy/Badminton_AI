"""Bundle a fine-tuned KAP and the team's RGB encoder into one pipeline checkpoint.

The pipeline switches to the team-built RGB branch when a KAP checkpoint carries ``rgb_encoder``.

Example:
    python scripts/paper/package_own_kap.py --kap work_dirs/paper_ownrgb_ss22/x13/finetune/ownrgb_x13_ss22x13_base1/best.pth \
        --encoder work_dirs/paper_ownrgb_ss22/x13/rgb_encoder_best.pth --output work_dirs/paper_ownrgb_ss22/x13/kap_own_rgb.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kap", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kap = torch.load(args.kap, map_location="cpu", weights_only=False)
    encoder = torch.load(args.encoder, map_location="cpu", weights_only=False)
    payload = {
        "model": kap["model"], "model_type": "cross_v2", "gate_free": True, "rgb_dim": kap["rgb_dim"],
        "court_reference": kap["court_reference"], "rgb_encoder": encoder["model"], "rgb_encoder_epoch": encoder["epoch"],
        "kap_epoch": kap["epoch"], "val_metrics": kap["val_metrics"], "clean_val_metrics": kap["clean_val_metrics"],
        "per_match": kap["per_match"], "base_checkpoint": kap["base_checkpoint"],
    }
    torch.save(payload, args.output)
    print(f"[+] {args.output} rgb_dim={payload['rgb_dim']} kap_epoch={payload['kap_epoch']} encoder_epoch={payload['rgb_encoder_epoch']}")


if __name__ == "__main__":
    main()
