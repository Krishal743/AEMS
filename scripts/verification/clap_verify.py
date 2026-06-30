import sys

print("[INFO] Python executable being used:")
print(sys.executable)
print("-" * 50)

# Step 1: Check CLAP import
try:
    import laion_clap
    print("laion_clap imported successfully")
except Exception as e:
    print("Failed to import laion_clap")
    print(e)
    exit()

# Step 2: Check model loading
try:
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    print("CLAP model loaded successfully")
except Exception as e:
    print("Failed to load CLAP model")
    print(e)
    exit()

# Step 3: Test text encoding
try:
    text = ["a dog barking"]
    emb = model.get_text_embedding(text)
    print("Text encoding works")
    print("Embedding shape:", emb.shape)
except Exception as e:
    print("Text encoding failed")
    print(e)
    exit()

print("-" * 50)
print("CLAP SETUP FULLY WORKING")