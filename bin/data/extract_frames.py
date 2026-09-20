import json, os, subprocess, glob, argparse, time, gc, re
from tqdm import tqdm
from src.config import AEMS_MANIFEST_PATH, AEMS_FRAMES_DIR, NUM_FRAMES

parser = argparse.ArgumentParser(description="Extract 16 uniform frames from AEMS videos")
parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
parser.add_argument("--frames-root", type=str, default=AEMS_FRAMES_DIR)
parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
parser.add_argument("--resume", action="store_true", help="Skip videos with frames already extracted")
args = parser.parse_args()

FRAMES_ROOT = args.frames_root
os.makedirs(FRAMES_ROOT, exist_ok=True)

FFMPEG_BIN = None
try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

with open(args.manifest, encoding="utf-8") as f:
    metadata = json.load(f)

videos = {}
for item in metadata:
    vid = item["video_id"]
    if vid not in videos:
        videos[vid] = item["video_path"]

print(f"Total unique videos: {len(videos)}")


def get_video_duration(video_path):
    ffmpeg_exe = FFMPEG_BIN or "ffmpeg"
    cmd = [ffmpeg_exe, "-i", video_path, "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
        if match:
            h, m, s = match.groups()
            return int(h) * 3600 + int(m) * 60 + float(s)
        return None
    except:
        return None


def extract_uniform_frames(video_path, output_dir, num_frames):
    video_path = str(video_path).replace('\\', '/')
    duration = get_video_duration(video_path)
    if duration is None or duration <= 0:
        return False
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_cmd = [FFMPEG_BIN] if FFMPEG_BIN else ["ffmpeg"]
    for i in range(num_frames):
        timestamp = (i + 0.5) * duration / num_frames
        output_file = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        if os.path.exists(output_file):
            continue
        cmd = ffmpeg_cmd + ["-y", "-loglevel", "error",
               "-ss", str(timestamp), "-i", video_path,
               "-vframes", "1", "-q:v", "2", output_file]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        except:
            return False
    return True


failed = []
success = 0
skipped_existing = 0
video_items = list(videos.items())

for video_id, video_path in tqdm(video_items, desc="Extracting frames"):
    output_dir = os.path.join(FRAMES_ROOT, video_id)

    if args.resume and os.path.exists(output_dir):
        existing = len(glob.glob(os.path.join(output_dir, "frame_*.jpg")))
        if existing == args.num_frames:
            success += 1
            skipped_existing += 1
            continue
        elif existing > 0:
            for f in glob.glob(os.path.join(output_dir, "frame_*.jpg")):
                os.remove(f)

    if extract_uniform_frames(video_path, output_dir, args.num_frames):
        success += 1
    else:
        failed.append(video_id)

failed_path = os.path.join(os.path.dirname(FRAMES_ROOT), "_frames_failed.txt")
with open(failed_path, "w") as f:
    for vid in failed:
        f.write(vid + "\n")

print(f"\nExtraction complete:")
print(f"  Success: {success} (new: {success - skipped_existing}, skipped existing: {skipped_existing})")
print(f"  Failed: {len(failed)}")
print(f"  Failed list: {failed_path}")
