# -*-coding:utf-8-*-
# author lyl
import paddle
import paddle.nn as nn


class Unflatten(nn.Layer):
    def __init__(self, input_channels, height, width):
        super(Unflatten, self).__init__()
        self.input_channels = input_channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.reshape([x.shape[0], self.input_channels, self.height, self.width])


class ConvVAE(nn.Layer):
    def __init__(self, input_channels=1, z_dim=20):
        super(ConvVAE, self).__init__()

        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2D(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2D(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*7*7, 1024),
            nn.ReLU()
        )

        # hidden -> mu
        self.fc1 = nn.Linear(1024, self.z_dim)
        # hidden -> log_sigma
        self.fc2 = nn.Linear(1024, self.z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128*7*7),
            Unflatten(128, 7, 7),
            nn.ReLU(),
            nn.Conv2DTranspose(128, 64, 4, 2, 1), # [bz, 64, 14, 14]
            nn.ReLU(),
            nn.Conv2DTranspose(64, 1, 4, 2, 1), # [bz, 1, 28, 28]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, log_sigma = self.fc1(h), self.fc2(h)
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        std = paddle.exp(log_sigma * 0.5)
        epislon = paddle.randn(shape=std.shape, dtype=std.dtype)
        return mu + std * epislon

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        sample_z = self.reparameterize(mu, log_sigma)
        return self.decode(sample_z), mu, log_sigma


if __name__ == '__main__':
    x = paddle.randn((64, 1, 28, 28))
    myvae = ConvVAE(input_channels=1, z_dim=20)
    res, mu, sigma = myvae(x)
    print(res.shape, mu.shape, sigma.shape)