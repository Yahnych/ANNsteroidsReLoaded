#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:42:19 2017

@author: danberenberg
"""

from PIL import Image


im = Image.open("test_shot.png").convert('LA')
im_resized = im.resize((64,64),Image.ANTIALIAS)
im_resized.save("test-64_greyscale.png", dpi=(64,64))