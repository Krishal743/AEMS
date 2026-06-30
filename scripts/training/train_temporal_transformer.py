import torch
import torch.nn as nn
import torch.nn.functional as F
import json, os, random, numpy as np, time, gc, argparse
from PIL import Image
import clip
from tqdm import tqdm
from src.evaluation.evaluate_retrieval import evaluate_retrieval
from src.models.temporal_transformer import TemporalTransformer

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
DEVICE = "cuda"

NUM_FRAMES = 16
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

METADATA_PATH = "data/processed/metadata/msrvtt_metadata.json"
FRAMES_DIR = "data/processed/video/frames_uniform"
OUTPUT_PATH = "embeddings/video_embeddings_transformer.pt"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Train Temporal Transformer on MSR-VTT")
parser.add_argument("--resume", type=str, default=None,
                    help="Resume from checkpoint path (e.g. checkpoints/temporal_transformer_epoch_4.pth)")
parser.add_argument("--epochs", type=int, default=4,
                    help="Total number of epochs to run (default: 4)")
args = parser.parse_args()

NUM_EPOCHS = args.epochs
start_epoch = 0
best_r1 = 0.0

print("[INIT] Loading CLIP model...")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model = clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad_(False)

print("[DATA] Loading metadata...")
with open(METADATA_PATH) as f:
    metadata = json.load(f)

print("[DATA] Finding videos with frames...")
valid_video_ids = set()
for vid in os.listdir(FRAMES_DIR):
    frame_dir = os.path.join(FRAMES_DIR, vid)
    if os.path.isdir(frame_dir):
        frame_count = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        if frame_count == NUM_FRAMES:
            valid_video_ids.add(vid)

print(f"[DATA] Videos with {NUM_FRAMES} frames: {len(valid_video_ids)}")

full_data = [m for m in metadata if m["split"] == "train" and m["video_id"] in valid_video_ids]
print(f"[DATA] Total train+val items: {len(full_data)}")

random.shuffle(full_data)
val_size = int(len(full_data) * 0.15)
val_data = full_data[:val_size]
train_data = full_data[val_size:]

unique_train_vids = len(set(m["video_id"] for m in train_data))
unique_val_vids = len(set(m["video_id"] for m in val_data))
print(f"[DATA] Train: {len(train_data)} items ({unique_train_vids} unique videos)")
print(f"[DATA] Val:   {len(val_data)} items ({unique_val_vids} unique videos)")

def get_video_frames(video_id):
    if video_id in FRAME_CACHE:
        return FRAME_CACHE[video_id]
    frame_dir = os.path.join(FRAMES_DIR, video_id)
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
    best_r1 = checkpoint["val_metrics"]["R@1"]
    print(f"[RESUME] Resuming from epoch {start_epoch}/{NUM_EPOCHS}")
    print(f"[RESUME] Previous val R@1: {best_r1:.4f}")

remaining = NUM_EPOCHS - start_epoch
if remaining > 0:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=1e-6
    )
    print(f"[SCHED] CosineAnnealingLR over {remaining} epochs, eta_min=1e-6")
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
        tokens = clip.tokenize(texts).to(DEVICE)
        embeds = clip_model.encode_text(tokens)
        embeds = embeds / embeds.norm(dim=1, keepdim=True)
    return embeds.float()

def compute_metrics(video_embeds, text_embeds, vids, temperature):
    sim = video_embeds @ text_embeds.T
    loss = info_nce_loss(video_embeds, text_embeds, temperature)
    metrics = evaluate_retrieval(sim, vids, vids, ks=[1, 5, 10])
    return loss.item(), metrics

print("[CACHE] Preloading all frames into RAM...")
for vid in tqdm(sorted(valid_video_ids), desc="Preloading"):
    frames = get_video_frames(vid)
    if frames is not None:
        FRAME_CACHE[vid] = frames
cache_gb = sum(f.numel() for f in FRAME_CACHE.values()) * 4 / 1e9
print(f"[CACHE] Loaded {len(FRAME_CACHE)} videos ({cache_gb:.1f} GB)")

def train_epoch(model, data, epoch):
    model.train()
    random.shuffle(data)
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    total_videos = 0
    scaler = torch.amp.GradScaler("cuda")

    pbar = tqdm(range(0, len(data), BATCH_SIZE), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for i in pbar:
        batch = data[i:i + BATCH_SIZE]
        if len(batch) < 2:
            continue

        video_ids = [item["video_id"] for item in batch]
        captions = [item["text"] for item in batch]

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
    throughput = total_videos / epoch_time
    return total_loss / max(num_batches, 1), epoch_time, throughput

@torch.no_grad()
def validate(model, data, epoch):
    model.eval()
    all_video_embeds, all_text_embeds, all_vids = [], [], []
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    pbar = tqdm(range(0, len(data), BATCH_SIZE), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
    for i in pbar:
        batch = data[i:i + BATCH_SIZE]
        if len(batch) < 2:
            continue

        video_ids = [item["video_id"] for item in batch]
        captions = [item["text"] for item in batch]

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
print("STARTING TRAINING")
print(f"  Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
print(f"  Weight decay: {WEIGHT_DECAY}, Temperature: {TEMPERATURE}")
print(f"  Gradient clip: {GRAD_CLIP}, Mixed precision: enabled")
print(f"  Train items: {len(train_data)}, Val items: {len(val_data)}")
print(f"  Model params: {total_params:,} ({trainable_params:,} trainable)")
if args.resume:
    print(f"  Resume from epoch {start_epoch}, remaining: {remaining}")
print("=" * 70)

log = []

for epoch in range(start_epoch, NUM_EPOCHS):
    torch.cuda.reset_peak_memory_stats()
    epoch_start = time.time()

    train_loss, epoch_time, throughput = train_epoch(model, train_data, epoch)
    val_loss, val_metrics, val_time = validate(model, val_data, epoch)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    current_lr = optimizer.param_groups[0]["lr"]

    epoch_log = {
        "epoch": epoch + 1,
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "val_R@1": round(val_metrics["R@1"], 4),
        "val_R@5": round(val_metrics["R@5"], 4),
        "val_R@10": round(val_metrics["R@10"], 4),
        "lr": current_lr,
        "epoch_time_sec": round(epoch_time + val_time, 1),
        "train_time_sec": round(epoch_time, 1),
        "val_time_sec": round(val_time, 1),
        "throughput_videos_per_sec": round(throughput, 1),
        "peak_gpu_mem_gb": round(peak_mem, 2),
    }
    log.append(epoch_log)

    print(f"\n{'─'*70}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")
    print(f"  Val R@1: {val_metrics['R@1']:.4f}  R@5: {val_metrics['R@5']:.4f}  R@10: {val_metrics['R@10']:.4f}")
    print(f"  Time: train={epoch_time:.1f}s  val={val_time:.1f}s  total={epoch_time+val_time:.1f}s")
    print(f"  Throughput: {throughput:.1f} videos/sec  |  Peak GPU: {peak_mem:.2f}GB")
    print(f"{'─'*70}\n")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"temporal_transformer_epoch_{epoch+1}.pth")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
    }, checkpoint_path)
    print(f"  [CHECKPOINT] Saved {checkpoint_path}")

    if val_metrics["R@1"] > best_r1:
        best_r1 = val_metrics["R@1"]
        torch.save(model.state_dict(), "models/temporal_transformer_best.pth")
        print(f"  [BEST] New best R@1: {best_r1:.4f}")

    if torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(val_loss)):
        print("[ABORT] NaN loss detected. Stopping training.")
        break

    if epoch > start_epoch and train_loss > log[-2]["train_loss"] * 3:
        print(f"[WARN] Train loss spiked: {log[-2]['train_loss']:.4f} -> {train_loss:.4f}")
        break

    if scheduler is not None:
        scheduler.step()
        print(f"  [SCHED] LR updated to: {optimizer.param_groups[0]['lr']:.2e}")

print(f"\n{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("FINAL EVALUATION ON FULL VALIDATION SET")
print(f"{'='*70}")
model.load_state_dict(torch.load("models/temporal_transformer_best.pth"))
model.eval()
final_loss, final_metrics, final_time = validate(model, val_data, NUM_EPOCHS - 1)
print(f"  Final Val loss: {final_loss:.4f}")
print(f"  Final R@1: {final_metrics['R@1']:.4f}")
print(f"  Final R@5: {final_metrics['R@5']:.4f}")
print(f"  Final R@10: {final_metrics['R@10']:.4f}")
print(f"  Time: {final_time:.1f}s")

print(f"\n{'='*70}")
print("EPOCH LOG")
print(f"{'='*70}")
print(f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>8} {'R@1':>6} {'R@5':>6} {'R@10':>7} {'Time(s)':>8} {'Vid/s':>7} {'GPU(GB)':>8}")
for entry in log:
    print(f"{entry['epoch']:>6} {entry['train_loss']:>10.4f} {entry['val_loss']:>8.4f} {entry['val_R@1']:>6.4f} {entry['val_R@5']:>6.4f} {entry['val_R@10']:>7.4f} {entry['epoch_time_sec']:>8.1f} {entry['throughput_videos_per_sec']:>7.1f} {entry['peak_gpu_mem_gb']:>8.2f}")

print(f"\n{'='*70}")
print("EXPORTING TRANSFORMER EMBEDDINGS FOR ALL VIDEOS")
print(f"{'='*70}")
model.load_state_dict(torch.load("models/temporal_transformer_best.pth"))
model.eval()

all_video_embeds = {}
unique_videos = sorted(valid_video_ids)
for video_id in tqdm(unique_videos, desc="Encoding"):
    frames = get_video_frames(video_id)
    if frames is None:
        continue
    frames = frames.to(DEVICE, non_blocking=True)
    with torch.no_grad():
        frame_embeds = encode_clip_frames(frames.unsqueeze(0))
        video_embed = model(frame_embeds)
    all_video_embeds[video_id] = video_embed.squeeze(0).cpu()

    if (len(all_video_embeds)) % 1000 == 0:
        torch.save(all_video_embeds, OUTPUT_PATH)

    del frames, frame_embeds, video_embed
    gc.collect()
    torch.cuda.empty_cache()

torch.save(all_video_embeds, OUTPUT_PATH)
print(f"[EXPORT] Saved {len(all_video_embeds)} embeddings to {OUTPUT_PATH}")

print("\n[DONE] Training complete.")
