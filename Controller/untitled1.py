#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:42:19 2017

@author: danberenberg
"""

from PIL import Image
import pyscreenshot

im = pyscreenshot.grab(bbox=(0,0,600,600))
#im_grey = Image.open("test_shot.png").convert('LA')
im_grey = im.convert('LA')
im_resized = im_grey.resize((64,64),Image.ANTIALIAS)
im_resized.save("test-64_greyscale.png", dpi=(64,64))
