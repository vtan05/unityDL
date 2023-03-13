import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dims, n_hidden):
        super(Encoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=160, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        #self.flatten = nn.Flatten(start_dim=1)

        #self.linear = nn.Sequential(
        #    nn.Linear(128 * 3 * 3, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, 256)
        #)

        self.maxpool = nn.MaxPool1d(3, stride=2, return_indices=True)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x, indices = self.maxpool(x)
        #x = self.flatten(x)
        #x = self.linear(x)
        return x, indices


class Decoder(nn.Module):
    def __init__(self, latent_dims, n_hidden):
        super(Decoder, self).__init__()
        #self.linear = nn.Sequential(
        #    nn.Linear(256, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, 1792),
        #    nn.ReLU()
        #)

        #self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.unpool = nn.MaxUnpool1d(3, stride=2)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=160, kernel_size=3, stride=1)
        )

    def forward(self, x, indices):
        #x = self.linear(x)
        #x = self.unflatten(x)
        x = self.unpool(x, indices)
        x = self.decoder_cnn(x)
        x = torch.sigmoid(x)
        return x
