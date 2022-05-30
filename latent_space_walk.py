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


def interpolate_points(p1, p2, n_steps=20):
	# interpolate ratios between the points
	ratios = np.linspace(0, 1, num=n_steps)
	# linear interpolate vectors
	vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
		vectors.append(v)
	return np.asarray(vectors)


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

    encoder = Encoder(args)
    encoder.load_state_dict(torch.load('./encoder.pt', map_location='cpu'))
    encoder.eval()

    decoder = Decoder(args)
    decoder.load_state_dict(torch.load('./decoder.pt', map_location='cpu'))
    decoder.eval()

    ### Draw random points in the latent space and map to pixel space
    
    rand_inits = torch.stack([torch.normal(mean=torch.zeros(256), std=torch.ones(256)*30) for _ in range(10)])
    inv_mapping = decoder(rand_inits)
    
    save_image(transform_back(inv_mapping.data),
               "./random_initializations.png",
              )

    ground_truth_data = torch.tensor(np.load('./ground_truth_images.npy'))[[0, 4]]
    mapping = encoder(ground_truth_data)

    ### Interpolated walk
    walk = torch.tensor(interpolate_points(mapping[0].detach().numpy(), mapping[1].detach().numpy(), 20))

    ### Random walk
    # walk[0] = ground_truth_data[0].numpy()
    # end = ground_truth_data[1].numpy()
    # for i in range(99):
    #     norm = np.linalg.norm(walk[i])
    #     step = walk[i]
    #     for j, s in enumerate(step):
    #         step[j] += np.random.normal(scale=10)
    #     walk[i+1] = step / norm

    inv_mapping = decoder(walk)

    save_image(transform_back(inv_mapping.data),
               "./walk.png",
              )

if __name__ == "__main__":
    main()

