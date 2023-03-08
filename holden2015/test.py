import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import plotly.express

from model import Encoder, Decoder


def viz_latent():
    encoded_samples = []
    for sample in tqdm(test_dataset):
        label = sample[1]

        encoder.eval()
        with torch.no_grad():
            encoded_img = encoder(img)

        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
        encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)


def show_image(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))


def run():
    data_dir = 'dataset'
    batch_size = 256
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset.transform = test_transform
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # loss_fn = torch.nn.MSELoss()
    lr = 0.001
    torch.manual_seed(0)
    latent_dims = 4
    n_hidden = 128

    encoder = Encoder(latent_dims=latent_dims, n_hidden=n_hidden)
    decoder = Decoder(latent_dims=latent_dims, n_hidden=n_hidden)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr)

    cpt_encoder = torch.load("outputs/encoder.pth")
    encoder.load_state_dict(cpt_encoder['model_state_dict'])
    optim.load_state_dict(cpt_encoder['optimizer_state_dict'])

    cpt_decoder = torch.load("outputs/decoder.pth")
    decoder.load_state_dict(cpt_decoder['model_state_dict'])

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        images, labels = iter(test_loader).next()
        latent = encoder(images)

        mean = latent.mean(dim=0)
        std = (latent - mean).pow(2).mean(dim=0).sqrt()

        latent = torch.randn(128, latent_dims) * std + mean
        img_recon = decoder(latent)

        fig, ax = plt.subplots(figsize=(20, 8.5))
        show_image(torchvision.utils.make_grid(img_recon[:100], 10, 5))
        plt.show()


if __name__ == "__main__":
    run()
