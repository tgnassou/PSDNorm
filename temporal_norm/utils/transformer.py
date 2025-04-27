import torch
import torch.nn as nn


class CNNTransformer(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        sfreq=100,
        n_epochs=35,
        transformer_layers=4,
        nhead=8,
        d_model=1024,
        dropout=0.1,
    ):
        super().__init__()
        kernel_size = 9
        cnn_plan = [
            (d_model // 16, 3 * kernel_size, 8),
            (d_model // 16, kernel_size, 1),
            (d_model // 8, kernel_size, 4),
            (d_model // 8, kernel_size, 1),
            (d_model // 4, kernel_size, 4),
            (d_model // 4, kernel_size, 1),
            (d_model // 2, kernel_size, 4),
            (d_model // 2, kernel_size, 1),
            (d_model, kernel_size, 4),
            (d_model, kernel_size, 1),
        ]

        layers, in_c = [], n_channels
        for out_c, k, s in cnn_plan:
            layers += [
                nn.Conv1d(
                    in_c, out_c,
                    kernel_size=k,
                    stride=s,
                    padding=k // 2
                ),
                nn.BatchNorm1d(out_c),
                nn.ELU(),
            ]
            in_c = out_c
        self.cnn = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(n_epochs)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, transformer_layers)

        self.classifier = nn.Conv1d(d_model, n_classes, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape input
        if x.ndim == 4:  # input x has shape (B, S, C, T)
            x = x.permute(0, 2, 1, 3)  # (B, C, S, T)
            x = x.flatten(start_dim=2)  # (B, C, S * T)

        x = self.cnn(x)
        x = self.pool(x)
        x = self.transformer(x.transpose(1, 2)).transpose(1, 2)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    from torchscan import summary

    n_channels = 2
    n_classes = 5
    n_epochs = 35
    sfreq = 100
    seconds = 30
    seq_len = n_epochs * seconds * sfreq

    model = CNNTransformer(
        n_channels=n_channels,
        n_classes=n_classes,
        sfreq=sfreq,
        n_epochs=n_epochs,
    ).eval().cuda()

    print("─ CNN summary ─")
    summary(model.cnn, (n_channels, seq_len), receptive_field=True)

    print("\n─ Transformer summary ─")
    # summary(model.transformer, (n_epochs, 1024))
