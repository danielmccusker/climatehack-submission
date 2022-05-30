import numpy as np

import torch
import argparse

from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim

from dataset_wae import TrainDataset, TestDataset
from wae_gan import Encoder, Decoder


def transform_back(tens):
    return (tens/2 + 0.5)


def main():

    parser = argparse.ArgumentParser(description="PyTorch MNIST WAE-GAN")
    parser.add_argument(
        "-batch_size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "-epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "-lr",
        type=float,
        default=0.0001,
        help="learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "-dim_h", type=int, default=128, help="hidden dimension (default: 128)"
    )
    parser.add_argument(
        "-n_z", type=int, default=128, help="hidden dimension of z (default: 8)"
    )
    parser.add_argument(
        "-LAMBDA",
        type=float,
        default=0.01,
        help="regularization coef MMD term (default: 10)",
    )
    parser.add_argument(
        "-n_channel", type=int, default=1, help="input channels (default: 1)"
    )
    parser.add_argument(
        "-sigma",
        type=float,
        default=0.5,
        help="variance of hidden dimension (default: 1)",
    )
    args = parser.parse_args()

    ground_truth_data = np.load('./ground_truth_images.npy')

    encoder = Encoder(args)
    decoder = Decoder(args)

    encoder.load_state_dict(torch.load('./encoder.pt', map_location='cpu'))
    encoder.eval()
    decoder.load_state_dict(torch.load('./decoder.pt', map_location='cpu'))
    decoder.eval()
  
    reconstructions = []
    for image in enumerate(ground_truth_data):
        inv_mapping = decoder(encoder(image))
        reconstructions.append(inv_mapping)

    ground_truth_data = Tensor(ground_truth_data)
    reconstructions = Tensor(reconstructions)

    save_image(transform_back(ground_truth_data.data),
               "./ground_truth_data.png",
               )
    save_image(transform_back(reconstructions.data),
               "./reconstructed_data.png",
              )

if __name__ == "__main__":
    main()

