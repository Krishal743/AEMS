import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import gc
import clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Keyword dictionaries for routing override
VISUAL_KEYWORDS = {"car", "person", "scene", "red", "blue", "building", "sky", "dog", "cat", "water", "face", "road", "tree", "mountain", "indoor", "outdoor", "city", "street", "room", "beach", "field", "forest", "desk", "chair", "table", "window", "floor", "wall", "car", "truck", "bus", "bike", "motorcycle", "boat", "airplane", "helicopter", "plane", "bird", "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "fish", "snake", "lizard", "frog", "butterfly", "bee", "ant", "spider", "crab", "shark", "whale", "dolphin", "child", "adult", "man", "woman", "boy", "girl", "people", "crowd", "soldier", "police", "doctor", "nurse", "chef", "driver", "pilot", "singer", "dancer", "actor", "athlete", "student", "teacher"}

AUDIO_KEYWORDS = {"music", "song", "sound", "loud", "quiet", "explosion", "crash", "bang", "fire", "water", "rain", "thunder", "wind", "voice", "speech", "talking", "singing", "laughing", "crying", "scream", "shout", "whisper", "applause", "cheering", "music", "beat", "rhythm", "drum", "guitar", "piano", "violin", "trumpet", "horn", "bell", "chime", "horn", "alarm", "siren", "bell", "gun", "shot", "firework", "engine", "motor", "tire", "footstep", "running", "walking", "jumping", "climbing", "swimming", "diving", "flying", "driving", "riding", "landing", "taking off", "breaking", "crashing", "hitting", "kicking", "punching", "slapping", "clapping", "snapping", "clicking", "ticking", "clock", "timer", "bell"}

TEXT_KEYWORDS = {"talk", "lecture", "explain", "describe", "tell", "show", "how to", "what is", "about", "regarding", "concerning", "tutorial", "lesson", "class", "course", "teaching", "learning", "study", "reading", "writing", "speaking", "discuss", "analysis", "review", "summary", "explanation", "demonstration", "instruction", "guide", "introduction", "conclusion", "result", "finding", "discovery", "knowledge", "information", "fact", "concept", "theory", "principle", "method", "approach", "technique", "strategy", "process", "procedure", "step", "stage", "phase", "level", "degree", "extent", "amount", "number", "point", "issue", "problem", "question", "answer", "solution"}

# Test queries manually selected
TEST_QUERIES = [
    # Visual-heavy (3)
    "red car driving on a highway",
    "blue sky with clouds",
    "person walking in a room",
    
    # Audio-related (3)
    "loud explosion sound",
    "music playing with drums",
    "rain falling sound",
    
    # Text/semantic-heavy (3)
    "explaining how to cook recipe",
    "lecture on physics concepts",
    "tutorial about coding",
]

# Labels for analysis
QUERY_LABELS = [
    "red car - VISUAL",
    "blue sky - VISUAL", 
    "person walking - VISUAL",
    "explosion - AUDIO",
    "music - AUDIO",
    "rain - AUDIO",
    "explaining - TEXT",
    "lecture - TEXT",
    "tutorial - TEXT",
]

EXPECTED_DOMINANT = [
    0,  # visual
    0,  # visual
    0,  # visual
    2,  # audio
    2,  # audio
    2,  # audio
    1,  # text
    1,  # text
    1,  # text
]


class GatingNetwork(nn.Module):
    def __init__(self, text_dim=512, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, text_embed):
        x = text_embed.float() if text_embed.dtype == torch.float16 else text_embed
        return self.net(x)


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x).float()


def main():
    print("[INIT] Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    
    print("[DATA] Loading embeddings...")
    video_db_v = torch.load("embeddings/video_embeddings.pt", weights_only=False)
    video_db_a = torch.load("embeddings/audio_embeddings.pt", weights_only=False)
    caption_db = torch.load("embeddings/caption_embeddings.pt", weights_only=False)
    
    # Get common videos
    test_vids = list(video_db_v.keys())
    common_vids = [vid for vid in test_vids if vid in video_db_a and vid in caption_db][:100]
    
    video_embeds_v = torch.stack([to_tensor(video_db_v[vid]) for vid in common_vids])
    video_embeds_a = torch.stack([to_tensor(video_db_a[vid]) for vid in common_vids])
    caption_embeds = torch.stack([
        to_tensor(caption_db[vid]).max(dim=0)[0] for vid in common_vids
    ])
    
    video_embeds_v = F.normalize(video_embeds_v, p=2, dim=1)
    video_embeds_a = F.normalize(video_embeds_a, p=2, dim=1)
    caption_embeds = F.normalize(caption_embeds, p=2, dim=1)
    
    print("[GATE] Loading gating network weights...")
    gating_net = GatingNetwork(text_dim=512, hidden_dim=128).to(DEVICE)
    gating_net.load_state_dict(torch.load("models/gating_weights.pth"))
    gating_net.eval()
    
    num_videos = len(common_vids)
    
    print("\n" + "="*70)
    print("PER-QUERY GATING BEHAVIOR TEST")
    print("="*70)
    
    results = []
    
    for i, query in enumerate(TEST_QUERIES):
        tokens = clip.tokenize([query]).to(DEVICE)
        query_embed = clip_model.encode_text(tokens)
        query_embed = query_embed / query_embed.norm(dim=1, keepdim=True)
        
        # Keyword override DISABLED - use learned gating only
        # query_lower = query.lower()
        # if any(word in query_lower for word in VISUAL_KEYWORDS):
        #     weights = torch.tensor([[0.6, 0.2, 0.2]]).to(DEVICE)
        # elif any(word in query_lower for word in AUDIO_KEYWORDS):
        #     weights = torch.tensor([[0.2, 0.2, 0.6]]).to(DEVICE)
        # elif any(word in query_lower for word in TEXT_KEYWORDS):
        #     weights = torch.tensor([[0.2, 0.6, 0.2]]).to(DEVICE)
        # else:
        with torch.no_grad():
            weights = gating_net(query_embed.float().to(DEVICE))
        
        w_v = weights[0, 0].item()
        w_t = weights[0, 1].item()
        w_a = weights[0, 2].item()
        
        # Similarity computation
        sim_v = query_embed.float().to(DEVICE) @ video_embeds_v.float().to(DEVICE).T
        sim_t = query_embed.float().to(DEVICE) @ caption_embeds.float().to(DEVICE).T
        sim_a = query_embed.float().to(DEVICE) @ video_embeds_a.float().to(DEVICE).T
        
        # Get dominant modality
        weights_list = [w_v, w_t, w_a]
        dominant = weights_list.index(max(weights_list))
        dominant_name = ["visual", "caption", "audio"][dominant]
        
        results.append({
            'query': query,
            'label': QUERY_LABELS[i],
            'w_v': w_v,
            'w_t': w_t,
            'w_a': w_a,
            'dominant': dominant,
            'expected': EXPECTED_DOMINANT[i]
        })
        
        print(f"\n{i+1}. Query: \"{query}\"")
        print(f"   Expected: {['visual', 'caption', 'audio'][EXPECTED_DOMINANT[i]]}")
        print(f"   Weights: w_v={w_v:.4f}, w_t={w_t:.4f}, w_a={w_a:.4f}")
        print(f"   Dominant: {dominant_name}")
        
        diff = max(weights_list) - min(weights_list)
        print(f"   Spread: {diff:.4f}")
    
    # Acceptance criteria check
    print("\n" + "="*70)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*70)
    
    # Condition A: At least some queries must show clear dominant modality (diff >= 0.15)
    condition_a = any(
        max(r['w_v'], r['w_t'], r['w_a']) - min(r['w_v'], r['w_t'], r['w_a']) >= 0.15
        for r in results
    )
    print(f"Condition A (spread >= 0.15): {'PASS' if condition_a else 'FAIL'}")
    
    # Condition B: Different queries produce different dominant modalities
    dominants = set(r['dominant'] for r in results)
    condition_b = len(dominants) > 1
    print(f"Condition B (vary dominant): {'PASS' if condition_b else 'FAIL'} - {dominants}")
    
    # Condition C: Dominant modality aligns with query meaning
    correct = sum(1 for r in results if r['dominant'] == r['expected'])
    condition_c = correct >= len(results) * 0.5
    print(f"Condition C (aligns meaning): {'PASS' if condition_c else 'FAIL'} - {correct}/{len(results)} correct")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_pass = condition_a and condition_b and condition_c
    print(f"Overall: {'ALL CONDITIONS PASS' if all_pass else 'SOME CONDITIONS FAILED'}")
    
    if all_pass:
        print("\n>> Proceed to next step: scaling up training")
    else:
        print("\n>> Gating is still weak - needs more training or fix")


if __name__ == "__main__":
    main()