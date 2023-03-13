import torch
from torchsummary import summary
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from model import Encoder, Decoder
from train import train, test
from utils import SaveBestModel


def run():
    #data_dir = 'dataset'
    #train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    # test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    #train_transform = transforms.Compose([transforms.ToTensor()])
    # test_transform = transforms.Compose([transforms.ToTensor()])

    #train_dataset.transform = train_transform
    # test_dataset.transform = test_transform

    train_dataset = np.load('dataset/LAFAN1/processed/train.npy')
    train_data = torch.Tensor(train_dataset)

    val_dataset = np.load('dataset/LAFAN1/processed/valid.npy')
    val_data = torch.Tensor(val_dataset)

    #m = len(train_dataset)
    #train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
    batch_size = 256

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.MSELoss()
    lr = 0.001
    torch.manual_seed(0)
    latent_dims = 4
    n_hidden = 256

    encoder = Encoder(latent_dims=latent_dims, n_hidden=n_hidden)
    decoder = Decoder(latent_dims=latent_dims, n_hidden=n_hidden)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    encoder.to(device)
    decoder.to(device)

    num_epochs = 10
    losses = {'train_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        train_loss = train(encoder, decoder, device, train_loader, loss_fn, optim)
        val_loss = test(encoder, decoder, device, valid_loader, loss_fn)
        save_best_model = SaveBestModel()
        save_best_model(val_loss, epoch, encoder, optim, loss_fn, 'encoder')
        save_best_model(val_loss, epoch, decoder, optim, loss_fn, 'decoder')
        print('EPOCH {}/{} : train loss {:.3e}, val loss {:.3e}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        losses['train_loss'].append(train_loss)
        losses['val_loss'].append(val_loss)

    plt.figure(figsize=(10, 8))
    plt.semilogy(losses['train_loss'], label='Train')
    plt.semilogy(losses['val_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
