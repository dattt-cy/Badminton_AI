import json, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(r"E:\HK1-2026\PBL6\AI_Classifier")
PYTHON = sys.executable

CLIPS = [
    {
        "id": "Clip 1 (Match 44 - Antonsen vs Axelsen)",
        "video": REPO_ROOT / "work_dirs" / "eval_clips" / "clip1_match44_10min.mp4",
        "output": REPO_ROOT / "work_dirs" / "eval_clips" / "clip1_match44_10min_pipeline.json",
    },
    {
        "id": "Clip 2 (Match 39 - Ginting vs Lee Zii Jia)",
        "video": REPO_ROOT / "work_dirs" / "eval_clips" / "clip2_match39_10min.mp4",
        "output": REPO_ROOT / "work_dirs" / "eval_clips" / "clip2_match39_10min_pipeline.json",
    },
    {
        "id": "Clip 3 (Match 43 - Marin vs Chochuwong)",
        "video": REPO_ROOT / "work_dirs" / "eval_clips" / "clip3_match43_10min.mp4",
        "output": REPO_ROOT / "work_dirs" / "eval_clips" / "clip3_match43_10min_pipeline.json",
    },
]

def main():
    print("=" * 80)
    print("[*] STARTING 3-CLIP x 10-MIN FULL PIPELINE BENCHMARK (30 MINS TOTAL)")
    print("=" * 80)

    ckpt_hit = REPO_ROOT / "src" / "ai_classifier" / "models" / "checkpoints" / "badminton_hitnet_v2.pth"

    for idx, c in enumerate(CLIPS, 1):
        print(f"\n>>> [{idx}/3] RUNNING FULL PIPELINE ON {c['id']} <<<", flush=True)
        t0 = time.time()
        cmd = [
            PYTHON,
            str(REPO_ROOT / "run_pipeline.py"),
            "--video", str(c["video"]),
            "--output", str(c["output"]),
            "--hit-checkpoint", str(ckpt_hit),
            "--threshold", "0.72",
            "--nms-radius", "6",
            "--same-side-nms-radius", "20",
            "--serve-rescue-threshold", "0.55",
            "--bridge-rescue-threshold", "0.55",
            "--use-tactical-decoder",
            "--no-auto-ocr",
        ]
        ret = subprocess.run(cmd)
        elapsed = time.time() - t0
        if ret.returncode == 0:
            print(f"[+] Completed {c['id']} in {elapsed:.1f}s -> {c['output'].name}", flush=True)
        else:
            print(f"[!] Error running {c['id']} (code {ret.returncode})", flush=True)

    print("\n" + "=" * 80)
    print("[*] ALL 3 CLIPS PROCESSED! COMPUTING EVALUATION AGAINST GROUND TRUTH...")
    print("=" * 80)

    # Call benchmark evaluation
    bench_script = REPO_ROOT / "scripts" / "evaluation" / "benchmark_3clips_10min.py"
    subprocess.run([PYTHON, str(bench_script)])

if __name__ == "__main__":
    main()

