import torch
import torch.nn as nn
import torch.nn.functional as F
import json, os, random, numpy as np, time, gc, argparse, subprocess
from PIL import Image
import clip
from tqdm import tqdm
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.models.temporal_transformer import TemporalTransformer
from src.config import (AEMS_MANIFEST_PATH, AEMS_FRAMES_DIR, AEMS_TRANSFORMER_BEST_PATH,
                         AEMS_TRANSFORMER_CHECKPOINT_DIR, AEMS_VIDEO_EMBEDDINGS_TRANSFORMER_PATH,
                         NUM_FRAMES, DEVICE, set_seeds)
from src.data.metadata import load_metadata

set_seeds(42)

NUM_EPOCHS_DEFAULT = 12
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.07
NUM_HEADS = 4
NUM_LAYERS = 2
HIDDEN_DIM = 512
FFN_HIDDEN = 2048
DROPOUT = 0.1
GRAD_CLIP = 1.0
FRAME_CACHE = {}

parser = argparse.ArgumentParser(description="Train AEMS Temporal Transformer")
parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
parser.add_argument("--frames-dir", type=str, default=AEMS_FRAMES_DIR)
parser.add_argument("--epochs", type=int, default=NUM_EPOCHS_DEFAULT)
parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
parser.add_argument("--hard-cap", type=int, default=18, help="Hard cap on epochs")
args = parser.parse_args()

if args.epochs > args.hard_cap:
    print(f"[WARN] Requested {args.epochs} epochs exceeds hard cap of {args.hard_cap}. Capping at {args.hard_cap}.")
    args.epochs = args.hard_cap

os.makedirs(AEMS_TRANSFORMER_CHECKPOINT_DIR, exist_ok=True)
start_epoch = 0
best_r1 = 0.0
best_epoch = 0

print("[INIT] Loading CLIP model...")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model = clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad_(False)

print("[DATA] Loading metadata...")
metadata = load_metadata(args.manifest)

print("[DATA] Finding videos with frames...")
valid_video_ids = set()
for vid in os.listdir(args.frames_dir):
    frame_dir = os.path.join(args.frames_dir, vid)
    if os.path.isdir(frame_dir):
        frame_count = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        if frame_count == NUM_FRAMES:
            valid_video_ids.add(vid)

print(f"[DATA] Videos with {NUM_FRAMES} frames: {len(valid_video_ids)}")

full_data = [m for m in metadata if m["split"] == "train" and m["video_id"] in valid_video_ids]
print(f"[DATA] Total train+val items: {len(full_data)}")

all_vids = sorted({m["video_id"] for m in full_data})
random.shuffle(all_vids)
val_size = max(1, int(len(all_vids) * 0.15))
val_vids = set(all_vids[:val_size])
train_vids = set(all_vids[val_size:])
assert len(train_vids & val_vids) == 0, "Train/val video overlap detected!"

val_data = [m for m in full_data if m["video_id"] in val_vids]
train_data = [m for m in full_data if m["video_id"] in train_vids]

unique_train_vids = len(set(m["video_id"] for m in train_data))
unique_val_vids = len(set(m["video_id"] for m in val_data))
print(f"[DATA] Train: {len(train_data)} items ({unique_train_vids} unique videos)")
print(f"[DATA] Val:   {len(val_data)} items ({unique_val_vids} unique videos)")


def get_video_frames(video_id):
    if video_id in FRAME_CACHE:
        return FRAME_CACHE[video_id]
    frame_dir = os.path.join(args.frames_dir, video_id)
    if not os.path.exists(frame_dir):
        return None
    frames = []
    for i in range(NUM_FRAMES):
        frame_path = os.path.join(frame_dir, "frame_%04d.jpg" % i)
        if not os.path.exists(frame_path):
            return None
        img = Image.open(frame_path).convert("RGB")
        img = preprocess(img)
        frames.append(img)
    return torch.stack(frames)


print("[MODEL] Creating transformer...")
model = TemporalTransformer().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[MODEL] Total params: {total_params:,}, Trainable: {trainable_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler("cuda")

if args.resume:
    print(f"[RESUME] Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_r1 = checkpoint.get("best_r1", 0.0)
    best_epoch = checkpoint.get("best_epoch", 0)
    print(f"[RESUME] Resuming from epoch {start_epoch}/{args.epochs}")
    print(f"[RESUME] Previous best val R@1: {best_r1:.4f}")

remaining = args.epochs - start_epoch
if remaining > 0:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=1e-6
    )
else:
    scheduler = None


def info_nce_loss(video_embeds, text_embeds, temperature):
    logits = video_embeds @ text_embeds.T / temperature
    labels = torch.arange(len(logits), device=DEVICE)
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    return (loss_v2t + loss_t2v) / 2


def encode_clip_frames(frame_tensors):
    bsz = frame_tensors.shape[0]
    with torch.no_grad(), torch.amp.autocast("cuda"):
        flat = frame_tensors.flatten(0, 1)
        embeds = clip_model.encode_image(flat)
        embeds = embeds.view(bsz, NUM_FRAMES, HIDDEN_DIM).float()
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    return embeds


def encode_clip_text(texts):
    with torch.no_grad(), torch.amp.autocast("cuda"):
        tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
        embeds = clip_model.encode_text(tokens)
        embeds = embeds / embeds.norm(dim=1, keepdim=True)
    return embeds.float()


print("[CACHE] Preloading all frames into RAM...")
for vid in tqdm(sorted(valid_video_ids), desc="Preloading"):
    frames = get_video_frames(vid)
    if frames is not None:
        FRAME_CACHE[vid] = frames
cache_gb = sum(f.numel() for f in FRAME_CACHE.values()) * 4 / 1e9
print(f"[CACHE] Loaded {len(FRAME_CACHE)} videos ({cache_gb:.1f} GB)")


def train_epoch(model, data, epoch_num):
    model.train()
    random.shuffle(data)
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    total_videos = 0

    pbar = tqdm(range(0, len(data), BATCH_SIZE), desc=f"Epoch {epoch_num+1}/{args.epochs} [Train]")
    for i in pbar:
        batch = data[i:i + BATCH_SIZE]
        if len(batch) < 2:
            continue
        video_ids = [item["video_id"] for item in batch]
        captions = [item["text_description"] for item in batch]
        frame_tensors, valid_captions, valid_vids = [], [], []
        for j, vid in enumerate(video_ids):
            frames = get_video_frames(vid)
            if frames is not None:
                frame_tensors.append(frames)
                valid_captions.append(captions[j])
                valid_vids.append(vid)
        if len(frame_tensors) < 2:
            continue
        frames_batch = torch.stack(frame_tensors).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            frame_embeds = encode_clip_frames(frames_batch)
            text_embeds = encode_clip_text(valid_captions)
        with torch.amp.autocast("cuda"):
            video_embeds = model(frame_embeds)
            loss = info_nce_loss(video_embeds, text_embeds, TEMPERATURE)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += loss.item()
        num_batches += 1
        total_videos += len(valid_vids)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        del frames_batch, frame_embeds, text_embeds, video_embeds, loss
        gc.collect()
        torch.cuda.empty_cache()
    epoch_time = time.time() - start_time
    throughput = total_videos / epoch_time if epoch_time > 0 else 0
    return total_loss / max(num_batches, 1), epoch_time, throughput


@torch.no_grad()
def validate(model, data, epoch_num):
    model.eval()
    all_video_embeds, all_text_embeds, all_vids = [], [], []
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    pbar = tqdm(range(0, len(data), BATCH_SIZE), desc=f"Epoch {epoch_num+1}/{args.epochs} [Val]")
    for i in pbar:
        batch = data[i:i + BATCH_SIZE]
        if len(batch) < 2:
            continue
        video_ids = [item["video_id"] for item in batch]
        captions = [item["text_description"] for item in batch]
        frame_tensors, valid_captions, valid_vids = [], [], []
        for j, vid in enumerate(video_ids):
            frames = get_video_frames(vid)
            if frames is not None:
                frame_tensors.append(frames)
                valid_captions.append(captions[j])
                valid_vids.append(vid)
        if len(frame_tensors) < 2:
            continue
        frames_batch = torch.stack(frame_tensors).to(DEVICE, non_blocking=True)
        frame_embeds = encode_clip_frames(frames_batch)
        text_embeds = encode_clip_text(valid_captions)
        video_embeds = model(frame_embeds)
        video_embeds = F.normalize(video_embeds, dim=1)
        sim = video_embeds @ text_embeds.T
        loss = info_nce_loss(video_embeds, text_embeds, TEMPERATURE)
        all_video_embeds.append(video_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
        all_vids.extend(valid_vids)
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        del frames_batch, frame_embeds, text_embeds, video_embeds, sim, loss
        gc.collect()
        torch.cuda.empty_cache()
    if num_batches == 0:
        return 0.0, {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0}, 0.0
    video_embeds_all = torch.cat(all_video_embeds, dim=0).float()
    text_embeds_all = torch.cat(all_text_embeds, dim=0).float()
    video_embeds_all = F.normalize(video_embeds_all, dim=1)
    text_embeds_all = F.normalize(text_embeds_all, dim=1)
    sim_all = video_embeds_all @ text_embeds_all.T
    metrics = evaluate_retrieval(sim_all, all_vids, all_vids, ks=[1, 5, 10])
    val_time = time.time() - start_time
    return total_loss / num_batches, metrics, val_time


print("=" * 70)
print("STARTING AEMS TRANSFORMER TRAINING")
print(f"  Batch size: {BATCH_SIZE}, Epochs: {args.epochs}, LR: {LEARNING_RATE}")
print(f"  Hard cap: {args.hard_cap}")
print("=" * 70)

log = []
no_improve_count = 0
early_stop_patience = 3
early_stop_threshold = 0.005

for epoch in range(start_epoch, args.epochs):
    torch.cuda.reset_peak_memory_stats()
    epoch_start = time.time()

    train_loss, epoch_time_val, throughput = train_epoch(model, train_data, epoch)
    val_loss, val_metrics, val_time = validate(model, val_data, epoch)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    current_lr = optimizer.param_groups[0]["lr"]

    if torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(val_loss)):
        print("[ABORT] NaN loss detected. Stopping training.")
        break

    if epoch > start_epoch and train_loss > log[-1]["train_loss"] * 3:
        print(f"[WARN] Train loss spiked: {log[-1]['train_loss']:.4f} -> {train_loss:.4f}")
        break

    epoch_log = {
        "epoch": epoch + 1,
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "val_R@1": round(val_metrics["R@1"], 4),
        "val_R@5": round(val_metrics["R@5"], 4),
        "val_R@10": round(val_metrics["R@10"], 4),
        "lr": current_lr,
        "epoch_time_sec": round(epoch_time_val + val_time, 1),
        "peak_gpu_mem_gb": round(peak_mem, 2),
    }
    log.append(epoch_log)

    print(f"\nEpoch {epoch+1}/{args.epochs}")
    print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")
    print(f"  Val R@1: {val_metrics['R@1']:.4f}  R@5: {val_metrics['R@5']:.4f}  R@10: {val_metrics['R@10']:.4f}")

    checkpoint_path = os.path.join(AEMS_TRANSFORMER_CHECKPOINT_DIR, f"temporal_transformer_epoch_{epoch+1}.pth")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "best_r1": best_r1,
        "best_epoch": best_epoch,
    }, checkpoint_path)
    print(f"  [CHECKPOINT] Saved {checkpoint_path}")

    if val_metrics["R@1"] > best_r1:
        improvement = val_metrics["R@1"] - best_r1
        best_r1 = val_metrics["R@1"]
        best_epoch = epoch + 1
        torch.save(model.state_dict(), AEMS_TRANSFORMER_BEST_PATH)
        print(f"  [BEST] New best R@1: {best_r1:.4f} (improved by {improvement:.4f})")
        no_improve_count = 0
    else:
        no_improve_count += 1
        print(f"  [PATIENCE] No improvement for {no_improve_count}/{early_stop_patience} epochs")

    if no_improve_count >= early_stop_patience:
        print(f"[EARLY STOP] No R@1 improvement >= {early_stop_threshold} for {early_stop_patience} epochs. Stopping.")
        break

    if scheduler is not None:
        scheduler.step()

print(f"\nTraining complete. Best val R@1: {best_r1:.4f} at epoch {best_epoch}")
print(f"Best checkpoint: {AEMS_TRANSFORMER_BEST_PATH}")

print("\n[OPTIONAL] Extend training? (use --resume with last checkpoint and higher --epochs)")
print(f"  python {__file__} --resume {checkpoint_path} --epochs {min(args.epochs + 6, args.hard_cap)}")
