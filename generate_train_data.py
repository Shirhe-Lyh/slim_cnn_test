#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 13:43:34 2018

@author: shirhe-lyh
"""

import cv2
import numpy as np

from captcha.image import ImageCaptcha


def generate_captcha(text='1'):
    capt = ImageCaptcha(width=28, height=28, font_sizes=[24])
    image = capt.generate_image(text)
    image = np.array(image, dtype=np.uint8)
    return image
    
    
if __name__ == '__main__':
    output_dir = './datasets/images/'
    for i in range(50000):
        label = np.random.randint(0, 10)
        image = generate_captcha(str(label))
        image_name = 'image{}_{}.jpg'.format(i+1, label)
        output_path = output_dir + image_name
        cv2.imwrite(output_path, image)
        
