import torch
import torch.nn as nn    

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        if self.use_attention:
            x, _ = self.self_attention(x, x, x)
            x = x.mean(dim=1)
        return self.fc(x)


class LSTMMLPClassifier(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        x = torch.cat((h[-2], h[-1]), dim=-1)  # Last forward & backward hidden states
        return self.fc(x)


class CrossAttention(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, dropout_rate=0.5, num_heads=8):
        super().__init__()
        self.attention_audio = nn.MultiheadAttention(embed_dim=input_dim_audio, num_heads=num_heads, batch_first=True)
        self.attention_text = nn.MultiheadAttention(embed_dim=input_dim_text, num_heads=num_heads, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(input_dim_audio + input_dim_text, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, audio_embedding, text_embedding):
        if audio_embedding is None or text_embedding is None:
            raise ValueError("Cross-attention requires both audio and text embeddings.")

        attn_audio, _ = self.attention_audio(audio_embedding, text_embedding, text_embedding)
        attn_text, _ = self.attention_text(text_embedding, audio_embedding, audio_embedding)

        fused_representation = torch.cat([attn_audio.mean(dim=1), attn_text.mean(dim=1)], dim=-1)
        return self.fc(fused_representation)
