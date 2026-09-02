#!/usr/bin/env python3
"""AEMS Phase A pipeline orchestration script."""

import subprocess, sys, os, time, argparse
from datetime import datetime

PYTHON = sys.executable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_LOG = os.path.join(BASE_DIR, "outputs", "aems", "_pipeline_log.txt")
os.makedirs(os.path.dirname(PIPELINE_LOG), exist_ok=True)

STAGES = {
    "preprocess": [
        ("build_manifest", f"scripts/data/build_aems_manifest.py", []),
    ],
    "extract_frames": [
        ("extract_frames", f"scripts/data/extract_aems_frames.py", ["--resume"]),
    ],
    "extract_audio": [
        ("extract_audio", f"scripts/data/extract_aems_audio.py", ["--resume"]),
    ],
    "embed": [
        ("video_embeddings", f"scripts/embeddings/precompute_aems_video_embeddings.py", []),
        ("audio_embeddings", f"scripts/embeddings/precompute_aems_audio_embeddings.py", []),
        ("text_embeddings_desc_train", f"scripts/embeddings/precompute_aems_text_embeddings.py",
         ["--split", "train", "--fusion", "description"]),
        ("text_embeddings_desc_test", f"scripts/embeddings/precompute_aems_text_embeddings.py",
         ["--split", "test", "--fusion", "description"]),
        ("text_embeddings_trans_train", f"scripts/embeddings/precompute_aems_text_embeddings.py",
         ["--split", "train", "--fusion", "transcript"]),
        ("text_embeddings_trans_test", f"scripts/embeddings/precompute_aems_text_embeddings.py",
         ["--split", "test", "--fusion", "transcript"]),
        ("text_embeddings_fused_train", f"scripts/embeddings/precompute_aems_text_embeddings.py",
         ["--split", "train", "--fusion", "fused"]),
        ("text_embeddings_fused_test", f"scripts/embeddings/precompute_aems_text_embeddings.py",
         ["--split", "test", "--fusion", "fused"]),
    ],
    "train_transformer": [
        ("train_transformer", f"scripts/training/train_aems_temporal_transformer.py",
         ["--epochs", "12"]),
    ],
    "export_transformer": [
        ("export_transformer", f"scripts/training/export_aems_transformer_embeddings.py", []),
    ],
    "train_gating": [
        ("train_gating", f"scripts/training/train_aems_gating.py", ["--epochs", "15"]),
    ],
    "eval": [
        ("eval_canonical", f"scripts/evaluation/eval_aems_retrieval.py",
         ["--visual-variant", "meanpool", "--text-variant", "fused", "--bootstrap"]),
    ],
}

REQUIRED_INPUTS = {
    "extract_frames": ["data/processed/aems/metadata/aems_manifest_v1.json"],
    "extract_audio": ["data/processed/aems/metadata/aems_manifest_v1.json"],
    "embed": ["data/processed/aems/metadata/aems_manifest_v1.json"],
    "train_transformer": ["embeddings/aems_video_embeddings_v1.pt"],
    "export_transformer": ["models/aems_temporal_transformer_best_v1.pth"],
    "train_gating": ["embeddings/aems_video_embeddings_v1.pt",
                     "embeddings/aems_audio_embeddings_v1.pt",
                     "embeddings/aems_text_embeddings_fused_train.pt",
                     "embeddings/aems_text_embeddings_fused_test.pt"],
    "eval": ["embeddings/aems_video_embeddings_v1.pt",
             "embeddings/aems_audio_embeddings_v1.pt",
             "embeddings/aems_text_embeddings_fused_test.pt"],
}

STAGE_ORDER = ["preprocess", "extract_frames", "extract_audio", "embed",
               "train_transformer", "export_transformer", "train_gating", "eval"]


def log_message(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(PIPELINE_LOG, "a") as f:
        f.write(line + "\n")


def check_inputs(stage):
    missing = []
    for path in REQUIRED_INPUTS.get(stage, []):
        if not os.path.exists(os.path.join(BASE_DIR, path)):
            missing.append(path)
    return missing


def run_stage(stage, pilot=False):
    if stage not in STAGES:
        log_message(f"[ERROR] Unknown stage: {stage}")
        return False

    missing = check_inputs(stage)
    if missing:
        log_message(f"[ERROR] Stage '{stage}' missing required inputs:")
        for m in missing:
            log_message(f"  Missing: {m}")
        return False

    all_ok = True
    for name, script, extra_args in STAGES[stage]:
        cmd = [PYTHON, script] + extra_args
        if pilot:
            cmd.append("--pilot")
        log_message(f"[START] {name}: {' '.join(cmd)}")
        start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR, timeout=86400)
            elapsed = time.time() - start
            if result.returncode == 0:
                log_message(f"[OK] {name} completed in {elapsed:.1f}s")
                for line in result.stdout.strip().split("\n")[-3:]:
                    if line.strip():
                        log_message(f"  {line.strip()}")
            else:
                log_message(f"[FAIL] {name} exit code {result.returncode} after {elapsed:.1f}s")
                log_message(f"  stderr: {result.stderr.strip()[-500:]}")
                all_ok = False
        except subprocess.TimeoutExpired:
            log_message(f"[FAIL] {name} timed out after 24h")
            all_ok = False
        except Exception as e:
            log_message(f"[FAIL] {name} exception: {e}")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="AEMS Phase A Pipeline Orchestrator")
    parser.add_argument("--stage", default="all",
                        help=f"Stage to run: {', '.join(STAGE_ORDER)} or 'all'")
    parser.add_argument("--pilot", action="store_true", help="Run pilot mode")
    args = parser.parse_args()

    log_message("=" * 60)
    log_message(f"AEMS PIPELINE START (pilot={args.pilot})")
    log_message("=" * 60)

    if args.stage == "all":
        stages_to_run = STAGE_ORDER
    elif args.stage in STAGE_ORDER:
        stages_to_run = [args.stage]
    else:
        log_message(f"[ERROR] Unknown stage: {args.stage}. Valid: {', '.join(STAGE_ORDER)}")
        sys.exit(1)

    all_success = True
    for stage in stages_to_run:
        log_message(f"\n{'='*60}")
        log_message(f"STAGE: {stage}")
        log_message(f"{'='*60}")
        if not run_stage(stage, pilot=args.pilot):
            all_success = False
            log_message(f"[ABORT] Stage '{stage}' failed. Stopping pipeline.")
            break

    log_message(f"\n{'='*60}")
    if all_success:
        log_message("PIPELINE COMPLETE — all stages succeeded")
    else:
        log_message("PIPELINE FAILED — see log above")
    log_message(f"{'='*60}")

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
