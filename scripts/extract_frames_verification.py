import os
import random

root = "data/processed/video/frames"
videos = os.listdir(root)

print("Total frame folders:", len(videos))

sample = random.choice(videos)
frames = os.listdir(os.path.join(root, sample))

print("Sample video:", sample)
print("Frames:", frames)
