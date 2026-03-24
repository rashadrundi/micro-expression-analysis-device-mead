import torch
import torch.nn as nn

class TransformerVA(nn.Module):
    def __init__(self, input_dim, hidden=256, num_layers=4, n_heads=4):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            nhead=n_heads,
            d_model=hidden,
            batch_first=True,
            dropout=0.1
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.output_proj = nn.Linear(hidden, 2)
    
    def forward(self, x):
        x = self.input_proj(x)
        enc = self.encoder(x)

        pooled = enc.mean(dim=1)

        return self.output_proj(pooled)