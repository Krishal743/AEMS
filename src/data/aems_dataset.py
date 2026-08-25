import os, glob
from PIL import Image
from torch.utils.data import Dataset
from src.data.metadata import load_metadata, filter_by_split


class AEMSDataset(Dataset):
    def __init__(self, manifest_path, split="train", num_frames=16):
        self.metadata = filter_by_split(load_metadata(manifest_path), split=split)
        self.num_frames = num_frames

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        frames_dir = item["frames_dir"]
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        assert len(frame_files) == self.num_frames, \
            f"Expected {self.num_frames} frames for {item['video_id']}, got {len(frame_files)}"
        images = [Image.open(f).convert("RGB") for f in frame_files]
        return {
            "images": images,
            "video_id": item["video_id"],
            "text_description": item["text_description"],
            "text_transcript": item["text_transcript"],
            "qa_questions": item["qa_questions"],
            "qa_answers": item["qa_answers"],
            "audio_path": item["audio_path"],
        }
