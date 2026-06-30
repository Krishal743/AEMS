from huggingface_hub import hf_hub_download
import zipfile
import os

repo_id = "friedrichor/MSR-VTT"

print("Downloading MSR-VTT video archive...")

zip_path = hf_hub_download(
    repo_id=repo_id,
    filename="MSRVTT_Videos.zip",
    repo_type="dataset"
)

print("Downloaded ZIP to:", zip_path)

# Extract videos
extract_dir = "data/raw/msrvtt/videos"
os.makedirs(extract_dir, exist_ok=True)

print("Extracting videos (this may take a few minutes)...")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_dir)

print("✅ Video extraction complete")
