import os

print("Videos:", len(os.listdir("data/raw/msrvtt/videos")))
print("Annotations:", os.listdir("data/raw/msrvtt/annotations"))

with open("data/raw/msrvtt/annotations/train_list.txt") as f:
    print("Train split sample:", f.readline().strip())

with open("data/raw/msrvtt/annotations/test_list.txt") as f:
    print("Test split sample:", f.readline().strip())
