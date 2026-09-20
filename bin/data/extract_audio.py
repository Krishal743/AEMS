import json, os, argparse, subprocess, gc, re
from tqdm import tqdm
from src.config import AEMS_MANIFEST_PATH, AEMS_AUDIO_DIR, AEMS_AUDIO_SR

parser = argparse.ArgumentParser(description="Extract 3x10s segments from AEMS videos")
parser.add_argument("--manifest", type=str, default=AEMS_MANIFEST_PATH)
parser.add_argument("--audio-dir", type=str, default=AEMS_AUDIO_DIR)
parser.add_argument("--resume", action="store_true", help="Skip videos with audio already extracted")
parser.add_argument("--segment-sec", type=int, default=10)
parser.add_argument("--num-segments", type=int, default=3)
args = parser.parse_args()

SEG = args.segment_sec
NUM = args.num_segments
MAX_WAV_SEC = SEG * NUM

AUDIO_DIR = args.audio_dir
os.makedirs(AUDIO_DIR, exist_ok=True)

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

ffmpeg_exe = FFMPEG_BIN or "ffmpeg"
failed = []
success = 0
skipped_existing = 0

for video_id, video_path in tqdm(list(videos.items()), desc="Extracting audio"):
    output_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")
    if args.resume and os.path.exists(output_path):
        success += 1
        skipped_existing += 1
        continue

    duration = None
    try:
        probe = subprocess.run(
            [ffmpeg_exe, "-i", video_path, "-f", "null", "-"],
            capture_output=True, timeout=60
        )
        stderr = probe.stderr.decode()
        m = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", stderr)
        if m:
            h, mn, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            duration = h * 3600 + mn * 60 + s + ms / 100.0
    except Exception:
        pass

    if duration is None:
        failed.append((video_id, "probe failed"))
        continue

    if duration <= MAX_WAV_SEC:
        cmd = [ffmpeg_exe, "-y", "-loglevel", "error", "-i", video_path,
               "-vn", "-acodec", "pcm_s16le", "-ar", str(AEMS_AUDIO_SR), "-ac", "1",
               output_path]
    else:
        mid = duration / 2.0
        seg_starts = [0.0, mid - SEG / 2.0, duration - SEG]
        filter_parts = []
        concat_inputs = []
        for i, ss in enumerate(seg_starts):
            filter_parts.append(f"[0:a]atrim=start={ss}:duration={SEG}[a{i}]")
            concat_inputs.append(f"[a{i}]")
        filter_chain = ";".join(filter_parts)
        concat_desc = "".join(concat_inputs)
        filter_complex = f"{filter_chain};{concat_desc}concat=n={NUM}:v=0:a=1"
        cmd = [ffmpeg_exe, "-y", "-loglevel", "error", "-i", video_path,
               "-filter_complex", filter_complex,
               "-acodec", "pcm_s16le", "-ar", str(AEMS_AUDIO_SR), "-ac", "1",
               output_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            success += 1
        else:
            failed.append((video_id, "empty output"))
    except subprocess.TimeoutExpired:
        failed.append((video_id, "timeout"))
    except subprocess.CalledProcessError as e:
        failed.append((video_id, e.stderr.decode()[:100] if e.stderr else "ffmpeg error"))

failed_path = os.path.join(os.path.dirname(AUDIO_DIR), "_audio_failed.txt")
with open(failed_path, "w") as f:
    for vid, reason in failed:
        f.write(f"{vid},{reason}\n")

print(f"\nAudio extraction complete:")
print(f"  Success: {success} (new: {success - skipped_existing}, skipped existing: {skipped_existing})")
print(f"  Failed: {len(failed)}")
print(f"  Failed list: {failed_path}")
