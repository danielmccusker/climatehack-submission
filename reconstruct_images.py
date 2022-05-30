import torch

from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim

from dataset_wae import TrainDataset, TestDataset
from wae_gan import Encoder, Decoder


def transform_back(tens):
    return (tens/2 + 0.5)


def main():

    torch.manual_seed(111)

    data_loc = "/scratch/mdatascience_team_root/mdatascience_team/shared_data/climatehack/preprocessed_data_all.npz"

    test_dataset = TestDataset(data_path=data_loc)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    encoder = torch.load('./encoder.pt', map_location='cpu')
    decoder = torch.load('./encoder.pt', map_location='cpu')

    random_selection = (torch.rand(10)*len(test_loader)).type(torch.IntTensor)

    ground_truth = []
    reconstructions = []
    for images in test_loader[random_selection]:
        image = images[0]
        inv_mapping = decoder(encoder(image))
        ground_truth.append(image)
        reconstructions.append(inv_mapping)

    ground_truth = Tensor(ground_truth)
    reconstructions = Tensor(reconstructions)

    save_image(transform_back(ground_truth.data),
               "./ground_truth_data.png",
               )
    save_image(transform_back(reconstructions.data),
               "./reconstructed_data.png",
              )

if __name__ == "__main__":
    main()

