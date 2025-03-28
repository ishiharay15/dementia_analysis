import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.pipelines as pipelines
import torch.nn.functional as F
from utils import *
import torch.nn.utils.rnn as rnn_utils
from transformers import BertTokenizer, BertModel

class AudioTextDataset(Dataset):
    def __init__(self, csv_file, model, feature_set, tokenizer, text_encoder, audio_encoder, lang="eng", device='cuda'):
        self.data = pd.read_csv(csv_file)

        if lang == "eng":
            self.data = self.data[self.data["lang"] == "eng_Latn"]
        elif lang == "chin":
            self.data = self.data[self.data["lang"].isin(["zho_Hant", "yue_Hant"])]
        elif lang == "all":
            pass

        self.data.reset_index(drop=True, inplace=True)
        self.feature_set = feature_set
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row["audio"]
        text_path = row["text"]
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        # Extract audio features based on selection
        audio_features = []

        if "wave2vec" in self.feature_set:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            wave2vec_feat = extract_wave2vec_embeddings(self.audio_encoder, waveform).to(self.device)
            audio_features.append(wave2vec_feat.mean(dim=0))  # Mean Pooling

        if "egemaps" in self.feature_set:
            egemaps_feat = extract_egemaps_features(audio_path).to(self.device)
            audio_features.append(egemaps_feat)

        if "librosa" in self.feature_set:
            librosa_feat = extract_librosa_features(audio_path).to(self.device)
            audio_features.append(librosa_feat)

        if audio_features:
            audio_embedding = torch.cat(audio_features, dim=-1)
        else:
            audio_embedding = None

        if "BERT" in self.feature_set:
            text_embedding = extract_text_embeddings(self.tokenizer, self.text_encoder, [text_path]).to(self.device)  
        else:
            text_embedding = None

        # Normalize
        if audio_embedding is not None:
            audio_embedding = normalize_embeddings(audio_embedding)
        if text_embedding is not None:
            text_embedding = normalize_embeddings(text_embedding)

        return audio_embedding, text_embedding, label.to(self.device)

def normalize_embeddings(embeddings):
    return F.layer_norm(embeddings, embeddings.shape[-1:])

def collate_fn(batch):
    audio_embeddings, text_embeddings, labels = zip(*batch)

    # Remove None values
    audio_embeddings = [e for e in audio_embeddings if e is not None]
    text_embeddings = [e for e in text_embeddings if e is not None]

    # Handle case where all values are None
    if not audio_embeddings and not text_embeddings:
        raise ValueError("Error: No valid feature embeddings found in batch!")

    # Pad sequences if needed
    padded_audio = rnn_utils.pad_sequence(audio_embeddings, batch_first=True, padding_value=0) if audio_embeddings else None
    padded_text = rnn_utils.pad_sequence(text_embeddings, batch_first=True, padding_value=0) if text_embeddings else None

    labels = torch.tensor(labels, dtype=torch.long)

    return padded_audio, padded_text, labels