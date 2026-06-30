import torch
import laion_clap
import librosa

class CLAPEncoder:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()
        self.model.to(device)
        self.model.eval()

    def encode_audio(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=48000, mono=True)

        # ensure numpy float32
        audio = audio.astype("float32")

        # batch dimension
        audio = audio.reshape(1, -1)

        with torch.no_grad():
            emb = self.model.get_audio_embedding_from_data(x=audio)

        return emb.squeeze(0)   # NO .cpu()

    def encode_text(self, texts):
        with torch.no_grad():
            emb = self.model.get_text_embedding(texts)

        return emb   # NO .cpu()