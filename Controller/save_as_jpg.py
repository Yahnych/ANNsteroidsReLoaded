#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 01:47:16 2017

@author: danberenberg
"""

import matplotlib.image as im
import argparse
import sys, os
import glob
import numpy as np
import controller_utils as ut

def setup_args():
    parser = argparse.ArgumentParser(prog="save_jpg")
    parser.add_argument("directory",help="filepath to save")
    
    return parser

if __name__ == '__main__':
    args = setup_args().parse_args()
    
    directory = args.directory
    
    if not os.path.isdir(directory):
        sys.exit("[!] Not a directory: {}".format(directory))
        
    regex = directory.strip("/") + "/*pgm"
    files = glob.glob(regex)
    
    if len(files) < 1:
        sys.exit("[!] No .pgm files found :(")
        
    
    for file in files:
        data = ut.pgm2pil(file)
        data = np.array(data,dtype=np.float32)
        print(data)
        filen = file.strip(".pgm")
        im.imsave(data,filen + ".jpg")
        
    for file in files:
        os.remove(file)
        
        
    print("deleted {} items".format(len(files)))