import random
import numpy as np
import torch


def train(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []

    for image_batch in dataloader:
        image_batch = image_batch.to(device)
        encoded_data, indices = encoder(image_batch)
        decoded_data = decoder(encoded_data, indices)

        loss = loss_fn(decoded_data, image_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)


def test(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        out = []
        label = []
        for image_batch in dataloader:
            image_batch = image_batch.to(device)
            encoded_data, indices = encoder(image_batch)
            decoded_data = decoder(encoded_data, indices)

            out.append(decoded_data.cpu())
            label.append(image_batch.cpu())

        out = torch.cat(out)
        label = torch.cat(label)

        val_loss = loss_fn(out, label)
    return val_loss.data
