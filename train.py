# -*-coding:utf-8-*-
# author lyl
import argparse
import shutil

import paddle
from VAE import VAE
from convVAE import ConvVAE
from utils import mnist_loader
import os
import paddle.nn.functional as F
from loguru import logger
import numpy as np
from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser(description='Convolution VAE-Paddle')
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True, help='model save directory')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--test_epoch', default=10, type=int)
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--z_dim', type=int, default=20)
    parser.add_argument('--input_dim', type=int, default=28*28)
    parser.add_argument('--input_channels', type=int, default=1)

    parser.add_argument('--mode', type=str, default='convVAE', help='choose convVAE or VAE')

    args = parser.parse_args()
    return args


def loss_func(res, images, mu, log_sigma):
    # 1. the reconstruction loss.
    reconstruction_loss = F.binary_cross_entropy(res, images, reduction='sum')

    # 2. KL-divergence
    divergence = 0.5 * paddle.sum(paddle.exp(log_sigma) + paddle.pow(mu, 2) - 1. - log_sigma)

    loss = reconstruction_loss + divergence
    return loss, reconstruction_loss, divergence


def save_image(random_res, save_path):
    ndarr = ((random_res * 255) + 0.5).clip(0, 255).transpose([0, 2, 3, 1]).squeeze(-1).numpy()
    img_blocks = []
    for i in range(8):
        img_blocks.append(np.hstack(ndarr[i*8:(i+1)*8]))
    for i in range(8):
        res = np.vstack(img_blocks[:8])
            
    img = Image.fromarray(np.uint8(res))
    img.save(save_path)


def save_checkpoint(state, is_best, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_file = os.path.join(save_dir, 'checkpoint_{}.pdparams'.format(state.get('epoch')))
    best_file = os.path.join(save_dir, 'model_best.pdparams')
    paddle.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


def train(args):
    if args.mode == 'VAE':
        myVAE = VAE(input_dim=28*28, hidden_dim=400, z_dim=args.z_dim)
    elif args.mode == 'convVAE':
        myVAE = ConvVAE(input_channels=args.input_channels, z_dim=args.z_dim)
    else:
        raise ValueError('arg mode must be VAE or convVAE. but now is %s' % args.mode)
    optimizer = paddle.optimizer.Adam(parameters=myVAE.parameters(), learning_rate=args.lr)

    start_epoch = 0
    best_test_loss = np.finfo('f').max

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info('loading checkpoint at %s' % args.resume)
            checkpoint = paddle.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            myVAE.set_state_dict(checkpoint['state_dict'])
            optimizer.set_state_dict(checkpoint['optimizer'])
        else:
            logger.info('no checkpoint found at {}'.format(args.resume))

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    train_loader, test_loader, classes = mnist_loader(args.batch_size, args.num_workers)

    logger.info('start training...')
    for epoch_id in range(start_epoch, args.epoches):

        for batch_id, data in enumerate(train_loader):
            images = data[0]
            if args.mode == 'VAE':
                images = images.reshape([images.shape[0], -1])
            res, mu, log_sigma = myVAE(images)
            loss, recon_loss, kld = loss_func(res, images, mu, log_sigma)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

            if (batch_id + 1) % 100 == 0:
                info = 'epoch: {}, batch: {}, recon_loss: {:.4f}, KL-div: {:.4f}, total_loss: {:.4f}'.format(
                    epoch_id+1, batch_id+1, recon_loss.item(), kld.item(), loss.item()
                )
                logger.info(info)

        # testing
        if (epoch_id+1) % args.test_epoch == 0:
            test_avg_loss = 0.0
            with paddle.no_grad():
                for idx, test_data in enumerate(test_loader):
                    test_images = test_data[0]
                    if args.mode == 'VAE':
                        test_images = test_images.reshape([test_images.shape[0], -1])
                    test_res, test_mu, test_log_sigma = myVAE(test_images)
                    test_loss, test_recon_loss, test_kld = loss_func(test_res, test_images, test_mu, test_log_sigma)
                    test_avg_loss += test_loss

                test_avg_loss /= len(test_loader.dataset)

                z = paddle.randn((args.batch_size, args.z_dim))
                random_res = myVAE.decode(z).reshape([-1, 1, 28, 28])
                save_image(random_res, './%s/random_sampled_%d.png' % (args.result_dir, epoch_id))

                # save model
                is_best = test_avg_loss < best_test_loss
                best_test_loss = min(test_avg_loss, best_test_loss)
                save_checkpoint({
                    'epoch': epoch_id,
                    'best_test_loss': best_test_loss,
                    'state_dict': myVAE.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, is_best, args.save_dir)


if __name__ == '__main__':
    args = get_parser()
    train(args)