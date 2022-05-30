import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

from dataset_wae import TrainDataset, TestDataset

#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class ConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity=activation)
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBNRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()
        torch.nn.init.kaiming_normal_(
            self.conv.weight, nonlinearity="leaky_relu"
        )
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))


class ConvTBNRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            in_channels, out_channels, *args, **kwargs
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()
        torch.nn.init.kaiming_normal_(
            self.conv.weight, nonlinearity="leaky_relu"
        )
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        # output should be 1x1
        self.main = nn.Sequential(
            ConvBNRelu(
                1, self.dim_h, stride=2, kernel_size=4, padding=1
            ),  # 64 x 64
            ConvBNRelu(
                self.dim_h, self.dim_h * 2, stride=2, kernel_size=4, padding=1
            ),  # 32 x 32
            ConvBNRelu(
                self.dim_h * 2,
                self.dim_h * 2,
                stride=4,
                kernel_size=5,
                padding=1,
            ),  # 8 x 8
            ConvBNRelu(
                self.dim_h * 2,
                self.dim_h * 4,
                stride=4,
                kernel_size=5,
                padding=1,
            ),  # 2 x 2
            ConvBNRelu(
                self.dim_h * 4,
                self.dim_h * 4,
                stride=2,
                kernel_size=2,
            ),  # 1 x 1
        )
        self.fc = nn.Linear(self.dim_h * 4, self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.proj = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * (2**3)), nn.ReLU()
        )

        self.main = nn.Sequential(
            ConvTBNRelu(
                self.dim_h * 2,
                self.dim_h * 2,
                stride=2,
                kernel_size=4,
                padding=1,
            ),  # 4 x 4
            ConvTBNRelu(
                self.dim_h * 2,
                self.dim_h * 2,
                stride=2,
                kernel_size=4,
                padding=1,
            ),  # 8 x 8
            ConvTBNRelu(
                self.dim_h * 2, self.dim_h, stride=2, kernel_size=4, padding=1
            ),  # 16 x 16
            ConvTBNRelu(
                self.dim_h, self.dim_h // 2, stride=2, kernel_size=4, padding=1
            ),  # 32 x 32
            ConvTBNRelu(
                self.dim_h // 2,
                self.dim_h // 4,
                stride=2,
                kernel_size=4,
                padding=1,
            ),  # 64 x 64
            ConvTBNRelu(
                self.dim_h // 4, 1, stride=2, kernel_size=4, padding=1
            ),  # 128 x 128
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 2, 2, 2)
        x = self.main(x)
        return x


class LDR(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LDR, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Identity()
        self.relu = nn.LeakyReLU()
        self._init_weights()

    def _init_weights(self):
        weights = (self.linear.weight,)
        biases = (self.linear.bias,)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight)
        for bias in biases:
            torch.nn.init.zeros_(bias)

    def forward(self, x):
        return self.dropout(self.relu(self.linear(x)))


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            LDR(self.n_z, self.dim_h * 4),
            LDR(self.dim_h * 4, self.dim_h * 4),
            LDR(self.dim_h * 4, self.dim_h * 4),
            LDR(self.dim_h * 4, self.dim_h * 4),
            LDR(self.dim_h * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x)
        return x


def main():

    torch.manual_seed(123)

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

    data_loc = "/scratch/mdatascience_team_root/mdatascience_team/shared_data/climatehack/preprocessed_data_all.npz"

    print('loading data')
    train_dataset = TrainDataset(data_path=data_loc)
    test_dataset = TestDataset(data_path=data_loc)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    print('done loading')

    if not os.path.isdir("./data/reconst_images"):
        os.makedirs("data/reconst_images")

    if not os.path.isdir("./checkpoint/wae/encoder"):
        os.makedirs("checkpoint/wae/encoder")

    if not os.path.isdir("./checkpoint/wae/decoder"):
        os.makedirs("checkpoint/wae/decoder")

    encoder, decoder, discriminator = (
        Encoder(args),
        Decoder(args),
        Discriminator(args),
    )

    #criterion = nn.MSELoss()

    encoder.train()
    decoder.train()
    discriminator.train()

    # Optimizers
    enc_optim = optim.Adam(encoder.parameters(), lr=args.lr)
    dec_optim = optim.Adam(decoder.parameters(), lr=args.lr)
    dis_optim = optim.Adam(discriminator.parameters(), lr=0.5 * args.lr)

    enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
    dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
    dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.5)

    ms_ssim_module = MS_SSIM(data_range=2, size_average=True, channel=1, win_size=7)

    if torch.cuda.is_available():
        encoder, decoder, discriminator = (
            encoder.cuda(),
            decoder.cuda(),
            discriminator.cuda(),
        )

    one = torch.tensor(1, dtype=torch.float32)
    mone = one * -1

    if torch.cuda.is_available():
        one = one.cuda()
        mone = mone.cuda()

    for epoch in range(args.epochs):
        step = 0

        for images in train_loader:

            if torch.cuda.is_available():
                images = images.cuda()

            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()

            # ======== Train Discriminator ======== #

            frozen_params(decoder)
            frozen_params(encoder)
            free_params(discriminator)

            z_fake = torch.randn(images.size()[0], args.n_z) * args.sigma

            if torch.cuda.is_available():
                z_fake = z_fake.cuda()

            d_fake = discriminator(z_fake)

            z_real = encoder(images)
            d_real = discriminator(z_real)

            torch.mean(torch.log(d_fake)).backward(mone)
            torch.mean(torch.log(1 - d_real)).backward(mone)

            dis_optim.step()

            # ======== Train Generator ======== #

            free_params(decoder)
            free_params(encoder)
            frozen_params(discriminator)

            batch_size = images.size()[0]

            z_real = encoder(images)
            x_recon = decoder(z_real)
            d_real = discriminator(encoder(Variable(images.data)))

            recon_loss = 1 - ms_ssim_module(x_recon, images)

            #recon_loss = criterion(x_recon, images)
            d_loss = args.LAMBDA * (torch.log(d_real)).mean()

            recon_loss.backward(one)
            d_loss.backward(mone)

            enc_optim.step()
            dec_optim.step()

            step += 1

            if (step + 1) % 30 == 0:
                print(
                    "Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f, Discrim Loss: %.4f"
                    % (
                        epoch + 1,
                        args.epochs,
                        step + 1,
                        len(train_dataset.cached_items)//args.epochs,
                        recon_loss.item(),
                        d_loss.item()
                    ),
                    flush=True
                )

        if (epoch + 1) % 1 == 0:
            batch_size = args.batch_size
            test_iter = iter(test_loader)
            test_data = next(test_iter)

            torch.save(
                encoder.state_dict(),
                f"./checkpoint/wae/encoder/epoch{epoch}.pt",
            )
            torch.save(
                decoder.state_dict(),
                f"./checkpoint/wae/decoder/epoch{epoch}.pt",
            )

            z_real = encoder(Variable(test_data.cuda()))

            reconst = (
                decoder(torch.randn_like(z_real))
                .cpu()
                .view(batch_size, 1, 128, 128)
            )

            save_image(
                transform_back(test_data.view(batch_size, 1, 128, 128)),
                "./data/reconst_images/wae_gan_input.png",
            )
            save_image(
                transform_back(reconst.data),
                "./data/reconst_images/wae_gan_images_%d.png" % (epoch + 1),
            )


def transform_back(tens):
    return (tens/2 + 0.5)


if __name__ == "__main__":
    main()

