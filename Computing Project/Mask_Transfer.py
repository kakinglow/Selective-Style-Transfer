# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:52:36 2020

@author: Kaking
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np
from imageio import imread, imsave

from skimage.transform import resize
import matplotlib.pyplot as plt


# Loads the mask and performs binarization where the mask is either 0 or 1
def load_mask(mask_path, shape):
    mask = imread(mask_path, as_gray=True)
    width, height , _ = shape.shape
    
    mask = resize(mask, (width, height), order=3).astype('float32')
    
    mask[mask <= 127] = 0
    mask[mask > 128] = 255
    
    max = np.amax(mask)
    mask /= max
    
    
    return mask

# The mask covers a region of the product image and replaces it with the same region of the content image
def mask_content(content, generated, mask):
    width, height, channels = generated.shape
    
    for i in range(width):
        for j in range(height):
            if mask[i, j] == 0.:
                generated[i, j, :] = content[i, j, :]
    
    return generated

# It puts the 2 functions together to make it easier to import into the main program.
def run_mask_transfer(generated_path, content_path, mask_path):
    
    generated_image = imread(generated_path, pilmode='RGB')
    img_width, img_height, channels = generated_image.shape
    
    content_image = imread(content_path, pilmode='RGB')
    content_image = resize(content_image, (img_width, img_height), preserve_range=True)
    
    mask = load_mask(mask_path, generated_image)
    
    img = mask_content(content_image, generated_image, mask)
    imsave('product.jpg', img)
    
    return img


