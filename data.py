# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

'''
import matplotlib
matplotlib.use('agg')
import numpy as np
import sys
import torch
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging
import os
import util
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
from PIL import Image 

np.set_printoptions(threshold=sys.maxsize)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dict, transform=None, normaliser=2 ** 8 - 1, is_valid=False, is_inference=False):
        """Initialisation for the Dataset object

        :param data_dict: dictionary of dictionaries containing images
        :param transform: PyTorch image transformations to apply to the images
        :returns: N/A
        :rtype: N/A

        """
        self.transform = transform
        self.data_dict = data_dict
        self.normaliser = normaliser
        self.is_valid = is_valid
        self.is_inference = is_inference

    def __len__(self):
        """Returns the number of images in the dataset

        :returns: number of images in the dataset
        :rtype: Integer

        """
        return (len(self.data_dict.keys()))

    def __getitem__(self, idx):
        """Returns a pair of images with the given identifier. This is lazy loading
        of data into memory. Only those image pairs needed for the current batch
        are loaded.

        :param idx: image pair identifier
        :returns: dictionary containing input and output images and their identifier
        :rtype: dictionary

        """
        while True:

            if (self.is_inference) or (self.is_valid):

                input_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['input_img'], normaliser=self.normaliser)
                output_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['output_img'], normaliser=self.normaliser)

                if self.normaliser==1:
                    input_img = input_img.astype(np.uint8)
                    output_img = output_img.astype(np.uint8)

                input_img = TF.to_pil_image(input_img)
                input_img = TF.to_tensor(input_img)
                output_img = TF.to_pil_image(output_img)
                output_img = TF.to_tensor(output_img)

                if input_img.shape[1]==output_img.shape[2]:
                    output_img=output_img.permute(0,2,1)

                return {'input_img': input_img, 'output_img': output_img,
                        'name': self.data_dict[idx]['input_img'].split("/")[-1]}

            else:

                output_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['output_img'], normaliser=self.normaliser)
                input_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['input_img'], normaliser=self.normaliser)

                if self.normaliser==1:
                    input_img = input_img.astype(np.uint8)
                    output_img = output_img.astype(np.uint8)

                input_img = TF.to_pil_image(input_img)
                output_img = TF.to_pil_image(output_img)

                if not self.is_valid:
                        
                        # Random horizontal flipping
                        if random.random() > 0.5:
                            input_img = TF.hflip(input_img)
                            output_img = TF.hflip(output_img)

                        # Random vertical flipping
                        if random.random() > 0.5:
                            input_img = TF.vflip(input_img)
                            output_img = TF.vflip(output_img)

                        # Random rotation +90
                        if random.random() > 0.5:
                            input_img=TF.rotate(input_img,90,expand=True)
                            output_img=TF.rotate(output_img,90,expand=True)
                            #input_img.save("./"+self.data_dict[idx]['input_img'].split("/")[-1]+"1.png")
                            #output_img.save("./"+self.data_dict[idx]['output_img'].split("/")[-1]+"2.png")

                        # Random rotation -90
                        if random.random() > 0.5:
                            input_img=TF.rotate(input_img,-90, expand=True)
                            output_img=TF.rotate(output_img,-90, expand=True)

                        # Random rotation -90
                        if random.random() > 0.5:
                            input_img=TF.rotate(input_img, 180, expand=True)
                            output_img=TF.rotate(output_img, 180, expand=True)

                        #output_img.save("./"+self.data_dict[idx]['output_img'].split("/")[-1]+"2.png")

                # Transform to tensor
                #print(output_img.shape)
                #plt.imsave("./"+self.data_dict[idx]['input_img'].split("/")[-1]+".png", output_img,format='png')
                input_img = TF.to_tensor(input_img)
                output_img = TF.to_tensor(output_img)
                
                return {'input_img': input_img, 'output_img': output_img,
                        'name': self.data_dict[idx]['input_img'].split("/")[-1]}



class Adobe5kDataLoader():

    def __init__(self, data_dirpath, img_ids_filepath):
        """Initialisation function for the data loader

        :param data_dirpath: directory containing the data
        :param img_ids_filepath: file containing the ids of the images to load
        :returns: N/A
        :rtype: N/A

        """
        self.data_dirpath = data_dirpath
        self.img_ids_filepath = img_ids_filepath
        self.data_dict = defaultdict(dict)

    def load_data(self):
        """ Loads the Samsung image data into a Python dictionary

        :returns: Python two-level dictionary containing the images
        :rtype: Dictionary of dictionaries

        """

        logging.info("Loading Adobe5k dataset ...")

        with open(self.img_ids_filepath) as f:
            '''
            Load the image ids into a list data structure
            '''
            image_ids = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            # image_ids_list = [x.rstrip() for x in image_ids]
            image_ids_list = image_ids

        idx = 0
        idx_tmp = 0
        img_id_to_idx_dict = {}

        for root, dirs, files in os.walk(self.data_dirpath):

            for file in files:

                img_id = file.split("-")[0]

                is_id_in_list = False
                for img_id_test in image_ids_list:
                    if img_id_test == img_id:
                        is_id_in_list = True
                        break

                if is_id_in_list:  # check that the image is a member of the appropriate training/test/validation split

                    if not img_id in img_id_to_idx_dict.keys():
                        img_id_to_idx_dict[img_id] = idx
                        self.data_dict[idx] = {}
                        self.data_dict[idx]['input_img'] = None
                        self.data_dict[idx]['output_img'] = None
                        idx_tmp = idx
                        idx += 1
                    else:
                        idx_tmp = img_id_to_idx_dict[img_id]

                    if "input" in root:  # change this to the name of your
                                        # input data folder

                        input_img_filepath = file

                        self.data_dict[idx_tmp]['input_img'] = root + \
                            "/" + input_img_filepath

                    elif ("output" in root):  # change this to the name of your

                        output_img_filepath = file

                        self.data_dict[idx_tmp]['output_img'] = root + \
                            "/" + output_img_filepath

                else:

                    logging.debug("Excluding file with id: " + str(img_id))

        for idx, imgs in self.data_dict.items():
            assert ('input_img' in imgs)
            assert ('output_img' in imgs)

        return self.data_dict
