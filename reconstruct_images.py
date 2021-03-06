import numpy as np

import torch
import argparse

from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim

#from torchsummary import summary

from dataset_wae import TrainDataset, TestDataset
from wae_gan import Encoder, Decoder


def transform_back(tens):
    return (tens/2 + 0.5)


def main():

    parser = argparse.ArgumentParser(description="PyTorch WAE-GAN")
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
        "-n_z", type=int, default=256, help="hidden dimension of z (default: 8)"
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

    ground_truth_data = torch.tensor(np.load('./ground_truth_images.npy'))
    print(ground_truth_data.shape)

    encoder = Encoder(args)
    decoder = Decoder(args)

    print('model definition')
    print(encoder)

    state_dict = torch.load('./encoder.pt', map_location='cpu')
    print('loaded model')
    for k, v in state_dict.items():
        if 'fc' in k:
            print(k)
            print(v.shape)
            print()

    encoder.load_state_dict(torch.load('./encoder.pt', map_location='cpu'))
    encoder.eval()
    decoder.load_state_dict(torch.load('./decoder.pt', map_location='cpu'))
    decoder.eval()

   
    mapping = encoder(ground_truth_data)
    print(torch.mean(mapping, axis=1), torch.std(mapping, axis=1))
    inv_mapping = decoder(mapping)
    

    save_image(transform_back(ground_truth_data.data),
               "./ground_truth_data.png",
               )
    save_image(transform_back(inv_mapping.data),
               "./reconstructed_data.png",
              )

if __name__ == "__main__":
    main()

