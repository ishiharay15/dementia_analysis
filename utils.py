import torch
import opensmile
import librosa

def extract_wave2vec_embeddings(wav2vec2_model, waveform, device):
    waveform = waveform.to(device)
    if waveform.dim() == 2:  # Convert stereo to mono
        waveform = waveform.mean(dim=0)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    with torch.no_grad():
        embedding, _ = wav2vec2_model.extract_features(waveform)
    return embedding[-1].squeeze(0)

def extract_text_embeddings(tokenizer, bert_model, text_paths, device):
    embeddings = []
    with torch.no_grad():
        for text_path in text_paths:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state.squeeze(0)
            embeddings.append(embedding)
    max_seq_len = max(e.shape[0] for e in embeddings)
    padded_embeddings = [F.pad(e, (0, 0, 0, max_seq_len - e.shape[0])) for e in embeddings]
    return torch.stack(padded_embeddings).squeeze(0)

def extract_egemaps_features(audio_path, device):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    features = smile.process_file(audio_path)
    return torch.tensor(features.values[0], dtype=torch.float32).to(device)

def extract_librosa_features(audio_path, device):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return torch.tensor(mfcc.mean(axis=1), dtype=torch.float32).to(device)