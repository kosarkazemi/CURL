import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import os
from .util import ImageProcessing
import cv2 as cv
import torchvision.transforms.functional as TF
from . import rgb_ted

RESIZE_WIDTH = 200
RESIZE_INFERENCE = True

def apply_curves(img, L, R, H, device="cpu"):

    img = img[:, 0:3, :, :]
    img_clamped = torch.clamp(img, 0, 1)

    img_clamped = img_clamped.to(device)

    img_clamped = torch.clamp(img, 0, 1)

    # lab
    img_lab = torch.clamp(ImageProcessing.rgb_to_lab(
        img_clamped.squeeze(0), device=device), 0, 1)

    img_lab, _ = ImageProcessing.adjust_lab(
        img_lab.squeeze(0), L[0, 0:48], device=device)

    # rgb
    img_rgb = ImageProcessing.lab_to_rgb(img_lab.squeeze(0), device=device)
    img_rgb = torch.clamp(img_rgb, 0, 1)

    img_rgb, _ = ImageProcessing.adjust_rgb(
        img_rgb.squeeze(0), R[0, 0:48], device=device)
    img_rgb = torch.clamp(img_rgb, 0, 1)

    # hsv
    img_hsv = ImageProcessing.rgb_to_hsv(img_rgb.squeeze(0), device=device)
    img_hsv = torch.clamp(img_hsv, 0, 1)

    img_hsv, _ = ImageProcessing.adjust_hsv(
        img_hsv, H[0, 0:64], device=device)
    img_hsv = torch.clamp(img_hsv, 0, 1)

    # back to rgb
    img_residual = torch.clamp(ImageProcessing.hsv_to_rgb(
        img_hsv.squeeze(0)), 0, 1)
    img = torch.clamp(img + img_residual.unsqueeze(0), 0, 1)

    return img



def curl_enhance(imgInputPath, model, device):

    input_img_downscaled, input_img = ImageProcessing.load_image(
                                                            imgInputPath,
                                                            normaliser=1,
                                                            resize=RESIZE_INFERENCE,
                                                            resize_width=RESIZE_WIDTH
                                                            )
    input_img_downscaled, input_img = input_img_downscaled.astype(np.uint8), input_img.astype(np.uint8)

    input_img_downscaled, input_img = TF.to_pil_image(input_img_downscaled), TF.to_pil_image(input_img)
    input_img_downscaled, input_img = TF.to_tensor(input_img_downscaled), TF.to_tensor(input_img)

    with torch.no_grad():

        input_img_downscaled = torch.unsqueeze(input_img_downscaled, 0)
        input_img_downscaled = torch.clamp(input_img_downscaled, 0, 1)
        input_img_downscaled = input_img_downscaled.to(device)
        input_img = torch.unsqueeze(input_img, 0)
        input_img = torch.clamp(input_img, 0, 1)
        input_img = input_img.to(device)

        # get the curves from applying thr forward on the down scaled image
        _ , _, L, R, H = model(input_img_downscaled)
        tn = model.tednet(input_img_downscaled)
        net_output_img_example = apply_curves(tn, L, R, H, device)

        net_output_img_example_numpy = net_output_img_example.squeeze(0).cpu().data.numpy()

        net_output_img_example_numpy = ImageProcessing.swapimdims_3HW_HW3(net_output_img_example_numpy)

        net_output_img_example_rgb = net_output_img_example_numpy
        net_output_img_example_rgb = ImageProcessing.swapimdims_HW3_3HW(net_output_img_example_rgb)
        net_output_img_example_rgb = np.expand_dims(net_output_img_example_rgb, axis=0)
        net_output_img_example_rgb = np.clip(net_output_img_example_rgb, 0, 1)

        net_output_img_example = (net_output_img_example_rgb[0, 0:3, :, :] * 255).astype('uint8')
        net_output_img_example = ImageProcessing.swapimdims_3HW_HW3(net_output_img_example)

        return net_output_img_example
