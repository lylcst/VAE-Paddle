# -*-coding:utf-8-*-
# author lyl
import paddle
from convVAE import ConvVAE
from VAE import VAE
import argparse
import os
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser('generate mnist')
parser.add_argument('--mode', type=str, default='convVAE')
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--result_dir', type=str, default='generate_result')

args = parser.parse_args()


def save_image(random_res, save_path):
    ndarr = ((random_res * 255) + 0.5).clip(0, 255).transpose([0, 2, 3, 1]).squeeze(-1).numpy()
    img_blocks = []
    for i in range(8):
        img_blocks.append(np.hstack(ndarr[i * 8:(i + 1) * 8]))
    for i in range(8):
        res = np.vstack(img_blocks[:8])

    img = Image.fromarray(np.uint8(res))
    img.save(save_path)


def generate(model):
    z = paddle.randn((64, 20))
    random_res = model.decode(z).reshape([-1, 1, 28, 28])
    return random_res


def main():
    if args.mode == 'convVAE':
        model = ConvVAE(input_channels=1, z_dim=20)
    elif args.mode == 'VAE':
        model = VAE(input_dim=28*28, hidden_dim=400, z_dim=20)
    else:
        raise ValueError('arg mode must be convVAE or VAE')

    if not os.path.exists(args.ckpt):
        raise ValueError('checkpoint not found at {}'.format(args.ckpt))
    model.set_state_dict(paddle.load(args.ckpt)['state_dict'])
    random_res = generate(model)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    save_image(random_res, './%s/random_sampled_result.png' % args.result_dir)


if __name__ == '__main__':
    main()

