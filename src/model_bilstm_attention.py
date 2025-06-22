import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):  # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq_len, hidden*2)

        # Attention weights
        attn_weights = F.softmax(self.attention_fc(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        logits = self.classifier(context)  # (batch, num_classes)
        return logits
