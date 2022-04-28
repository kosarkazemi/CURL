# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

Instructions:

To get this code working on your system / problem please see the README.

*** BATCH SIZE: Note this code is designed for a batch size of 1. The code needs re-engineered to support higher batch sizes. Using higher batch sizes is not supported currently and could lead to artefacts. To replicate our reported results 
please use a batch size of 1 only ***

'''

from data import Adobe5kDataLoader, Dataset
import time
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import logging
import argparse
import torch.optim as optim
import numpy as np
import datetime
import os.path
import os
import metric
import model
import config
import sys
from torch.utils.tensorboard import SummaryWriter
np.set_printoptions(threshold=sys.maxsize)


def make_loaders():

    training_data_loader = Adobe5kDataLoader(data_dirpath=config.training_img_dirpath,
                                            img_ids_filepath=config.training_img_dirpath+"/images_train.txt")
    training_data_dict = training_data_loader.load_data()

    training_dataset = Dataset(data_dict=training_data_dict, normaliser=1, is_valid=False)

    validation_data_loader = Adobe5kDataLoader(data_dirpath=config.training_img_dirpath,
                                            img_ids_filepath=config.training_img_dirpath+"/images_valid.txt")
    validation_data_dict = validation_data_loader.load_data()
    validation_dataset = Dataset(data_dict=validation_data_dict, normaliser=1, is_valid=True)

    testing_data_loader = Adobe5kDataLoader(data_dirpath=config.training_img_dirpath,
                                        img_ids_filepath=config.training_img_dirpath+"/images_test.txt")
    testing_data_dict = testing_data_loader.load_data()
    testing_dataset = Dataset(data_dict=testing_data_dict, normaliser=1,is_valid=True)

    training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True,
                                                    num_workers=6)
    testing_data_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=config.batch_size, shuffle=False,
                                                    num_workers=6)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size,
                                                    shuffle=False,
                                                    num_workers=6)

    return training_data_loader, validation_data_loader, testing_data_loader


def main():


    # make loggers
    writer = SummaryWriter()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = f"./log_{timestamp}"
    os.mkdir(log_dirpath)
    handlers = [logging.FileHandler(f"./log_{timestamp}/curl.log"), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    # get data loaders
    training_data_loader, validation_data_loader, testing_data_loader = make_loaders()

    # make model
    net = model.CURLNet()
    net.to(config.device) 

    # make criterion
    criterion = model.CURLLoss(ssim_window_size=5)

    # make evaluators
    validation_evaluator = metric.Evaluator(
        criterion, validation_data_loader, "valid", log_dirpath)
    testing_evaluator = metric.Evaluator(
        criterion, testing_data_loader, "test", log_dirpath)

    if config.start_epoch > 0 :
        # load model and optimizer
        checkpoint = torch.load(config.checkpoint_filepath, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                    net.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-10)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for g in optimizer.param_groups:
            g['lr'] = 1e-5

        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        net.to(config.device)

    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                    net.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-10)

    # update model and optimizer status
    net.train()
    optimizer.zero_grad()

    best_valid_psnr = 0.0
    psnr_avg, ssim_avg = 0.0, 0.0
    total_examples = 0


    ## training loop
    for epoch in range(start_epoch, config.num_epoch):

        # train loss
        examples = 0.0
        running_loss = 0.0
        
        for batch_num, data in enumerate(training_data_loader, 0):

            input_img_batch = data['input_img'].to(config.device)
            gt_img_batch = data['output_img'].to(config.device)
            category = data['name']

            # pass image to the network
            net_img_batch, gradient_regulariser = net(input_img_batch)
            net_img_batch = torch.clamp(net_img_batch, 0.0, 1.0)

            loss = criterion(net_img_batch,
                            gt_img_batch, gradient_regulariser)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            examples += config.batch_size
            total_examples+=config.batch_size

            writer.add_scalar('Loss/train', loss.data[0], total_examples)

        logging.info('[%d] train loss: %.15f' %
                    (epoch + 1, running_loss / examples))
        writer.add_scalar('Loss/train_smooth', running_loss / examples, epoch + 1)

        # # Valid loss
        # examples = 0.0
        # running_loss = 0.0

        # for batch_num, data in enumerate(validation_data_loader, 0):

        #     net.eval()

        #     input_img_batch, gt_img_batch, category = Variable(
        #         data['input_img'],
        #         requires_grad=True).cuda(), Variable(data['output_img'],
        #                                             requires_grad=False).cuda(), data['name']

        #     net_img_batch, gradient_regulariser = net(
        #         input_img_batch)
        #     net_img_batch = torch.clamp(
        #         net_img_batch, 0.0, 1.0)

        #     optimizer.zero_grad()

        #     loss = criterion(net_img_batch,
        #                     gt_img_batch, gradient_regulariser)

        #     running_loss += loss.data[0]
        #     examples += config.batch_size
        #     total_examples+=config.batch_size
        #     writer.add_scalar('Loss/train', loss.data[0], total_examples)


        # logging.info('[%d] valid loss: %.15f' %
        #             (epoch + 1, running_loss / examples))
        # writer.add_scalar('Loss/valid_smooth', running_loss / examples, epoch + 1)
        # net.train()


        if (epoch + 1) % config.valid_every == 0:

            logging.info("Evaluating model on validation dataset")

            valid_loss, valid_psnr, valid_ssim = validation_evaluator.evaluate(net, epoch)
            test_loss, test_psnr, test_ssim = testing_evaluator.evaluate(net, epoch)

            # update best validation set psnr
            if valid_psnr > best_valid_psnr:

                logging.info(
                    "Validation PSNR has increased. Saving the more accurate model to file: " + 
                    'curl_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(valid_psnr,
                                                                                            valid_loss.tolist()[0],
                                                                                            test_psnr, test_loss.tolist()[0],
                                                                                            epoch))

                best_valid_psnr = valid_psnr
                snapshot_prefix = os.path.join(log_dirpath, 'curl')
                snapshot_path = snapshot_prefix + '_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(valid_psnr,
                                                                                                                        valid_loss.tolist()[0],
                                                                                                                        test_psnr, test_loss.tolist()[0],
                                                                                                                        epoch +1)

                # save checkpoint
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, snapshot_path)

            net.train()

    '''
    Run the network over the testing dataset split
    '''
    snapshot_prefix = os.path.join(log_dirpath, 'curl')

    valid_loss, valid_psnr, valid_ssim = validation_evaluator.evaluate(net, epoch)
    test_loss, test_psnr, test_ssim = testing_evaluator.evaluate(net, epoch)

    snapshot_path = snapshot_prefix + '_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(valid_psnr,
                                                                                                            valid_loss.tolist()[0],
                                                                                                            test_psnr, test_loss.tolist()[0],
                                                                                                            epoch +1)
    snapshot_prefix = os.path.join(log_dirpath, 'curl')
    torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, snapshot_path)

if __name__ == "__main__":
    main()
