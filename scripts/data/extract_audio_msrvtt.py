import os
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip

VIDEO_DIR = "data/raw/msrvtt/videos"
OUTPUT_DIR = "data/processed/audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 48000
CLIP_DURATION = 10


def extract_audio(video_path, output_path):
    try:
        video = VideoFileClip(video_path)

        # skip videos without audio
        if video.audio is None:
            print(f"Skipping (no audio): {video_path}")
            return

        duration = video.duration

        start = max(0, (duration - CLIP_DURATION) / 2)
        end = start + CLIP_DURATION

        audio = video.audio.subclip(start, end)

        temp_path = "temp.wav"
        audio.write_audiofile(temp_path, fps=TARGET_SR, verbose=False, logger=None)

        y, sr = librosa.load(temp_path, sr=TARGET_SR, mono=True)

        if len(y) < TARGET_SR * CLIP_DURATION:
            y = librosa.util.fix_length(y, size=TARGET_SR * CLIP_DURATION)

        sf.write(output_path, y, TARGET_SR)
        os.remove(temp_path)

    except Exception as e:
        print(f"Error processing {video_path}: {e}")


def main():
    print(f"Total videos: {len(os.listdir(VIDEO_DIR))}")
    for vid in os.listdir(VIDEO_DIR):
        # only process valid mp4 files
        if not vid.endswith(".mp4"):
            continue

        # skip macOS junk files
        if vid.startswith("._") or vid.startswith("."):
            continue

        video_path = os.path.join(VIDEO_DIR, vid)
        output_path = os.path.join(OUTPUT_DIR, vid.replace(".mp4", ".wav"))

        # skip if already processed
        if os.path.exists(output_path):
            continue

        print(f"Processing {vid}")
        extract_audio(video_path, output_path)


if __name__ == "__main__":
    main()