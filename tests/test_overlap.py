import random

def check_overlap(full_data):
    # This is the NEW logic
    all_vids = sorted({m["video_id"] for m in full_data})
    random.seed(42)
    random.shuffle(all_vids)
    val_size = max(1, int(len(all_vids) * 0.15))
    val_vids = set(all_vids[:val_size])
    train_vids = set(all_vids[val_size:])
    
    # Check overlap
    return train_vids & val_vids

def test_train_val_overlap():
    # Mock data: 100 videos, 20 captions each
    metadata = []
    for vid in range(100):
        for cap_id in range(20):
            metadata.append({"video_id": f"vid_{vid}", "text": "...", "split": "train"})
            
    overlap = check_overlap(metadata)
    
    # Assert there is NO overlap (this should fail on original code)
    assert len(overlap) == 0, f"Leakage detected! {len(overlap)} overlapping videos"

if __name__ == "__main__":
    try:
        test_train_val_overlap()
        print("Test Passed!")
    except AssertionError as e:
        print(f"Test Failed: {e}")
        exit(1)
