# -*-coding:utf-8-*-
# author lyl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class VAE(nn.Layer):
    def __init__(self, input_dim=784, hidden_dim=400, z_dim=20):
        super(VAE, self).__init__()
        # ecoder part
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim) # mu
        self.fc3 = nn.Linear(hidden_dim, z_dim) # log_sigma

        # decoder part
        self.fc4 = nn.Linear(z_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_sigma = self.fc3(h)
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        sigma = paddle.exp(log_sigma * 0.5)
        epsilon = paddle.randn(shape=sigma.shape, dtype=sigma.dtype)
        return mu + sigma * epsilon

    def decode(self, x):
        h = F.relu(self.fc4(x))
        res = F.sigmoid(self.fc5(h))
        return res

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        sample_z = self.reparameterize(mu, log_sigma)
        res = self.decode(sample_z)

        return res, mu, log_sigma