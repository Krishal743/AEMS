#!/usr/bin/env python3
"""Clean up intermediate transformer checkpoints, keeping only best + last."""

import os, glob, sys, argparse
from src.config import AEMS_TRANSFORMER_CHECKPOINT_DIR, AEMS_TRANSFORMER_BEST_PATH

parser = argparse.ArgumentParser(description="Clean up intermediate AEMS checkpoints")
parser.add_argument("--keep-best", action="store_true", default=True)
parser.add_argument("--keep-last", action="store_true", default=True)
parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
args = parser.parse_args()

checkpoint_dir = AEMS_TRANSFORMER_CHECKPOINT_DIR
best_path = AEMS_TRANSFORMER_BEST_PATH

if not os.path.isdir(checkpoint_dir):
    print(f"No checkpoint directory found: {checkpoint_dir}")
    sys.exit(0)

all_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "temporal_transformer_epoch_*.pth")))
if not all_checkpoints:
    print(f"No checkpoints found in {checkpoint_dir}")
    sys.exit(0)

to_keep = set()
if args.keep_best and os.path.exists(best_path):
    to_keep.add(best_path)
if args.keep_last:
    to_keep.add(all_checkpoints[-1])

to_delete = [p for p in all_checkpoints if p not in to_keep]

print(f"Found {len(all_checkpoints)} checkpoints in {checkpoint_dir}")
print(f"  Keep best: {best_path if args.keep_best else 'no'}")
print(f"  Keep last: {all_checkpoints[-1] if args.keep_last else 'no'}")
print(f"  To delete: {len(to_delete)}")
print()

for p in sorted(to_delete):
    size_mb = os.path.getsize(p) / 1e6
    print(f"  {'[DRY RUN]' if args.dry_run else '[DELETE]'} {os.path.basename(p)} ({size_mb:.1f} MB)")

total_size = sum(os.path.getsize(p) for p in to_delete)
print(f"\nTotal reclaimable: {total_size / 1e6:.1f} MB")

if not args.dry_run and to_delete:
    confirm = input(f"Delete {len(to_delete)} intermediate checkpoints? [y/N] ")
    if confirm.lower() == "y":
        for p in to_delete:
            os.remove(p)
        print(f"Deleted {len(to_delete)} files, reclaimed {total_size / 1e6:.1f} MB")
    else:
        print("Aborted.")
