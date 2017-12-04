#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:42:19 2017

@author: danberenberg
"""

#from PIL import Image
#import PIL
#import time
import sys,os
import argparse
import glob
from scipy.misc import imread
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import warnings

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE    = '\x1b[2K'
CURSOR_DOWN_ONE = "\x1b[B"
BUFFER_LINE   = CURSOR_UP_ONE + ERASE_LINE 

def pgm2pil(filename):

    try:
        inFile = open(filename)
        header = None
        size = None
        maxGray = None
        data = []

        for line in inFile:
            stripped = line.strip()
           # print (stripped)
            if stripped[0] == '#': 
                continue
            elif header == None: 
                if stripped != 'P2': return None
                header = stripped
            elif size == None:
                size = [int(elt) for elt in stripped.split()]#map(int, stripped.split())
            elif maxGray == None:
                maxGray = int(stripped)
            else:
                for item in stripped.split():
                    data.append(int(item.strip()))

        data = np.reshape(data, (size[1],size[0]))/255.0#/float(maxGray)*255
        return np.flipud(data)

    except Exception as e:
        print(e, "OH SHIT");
        pass

    return None

def load_bar(curr,full):
    # 100 bars
    ratio = curr/float(full)
    num_bars = int (100*ratio)/2
 
    load = "[ " + ":"*int(num_bars) + " "*(50-int(num_bars)) + "]"
    
    return load
def resize_and_save(filename,out_path,xsize,ysize,curr,full):
    """
    resize an image given by the [filename], by the specified [xsize] and [ysize] dimensions
    writing the new image to the [out_path].
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #im_resized.save(out_path + filename+'_rsz{}_{}.png'.format(xsize,ysize))
        
        if curr != 0:
            print (BUFFER_LINE + "resizing {:<25} --> {}/{} {}".format(filename,curr,full,load_bar(curr,full)))
        
        else:
            print ("resizing {:<25} --> {}/{} {}".format(filename,curr,full,load_bar(curr,full)))
        #im = Image.open(filename)
        pgm = pgm2pil(filename)
    
        #im2 = imread(filename)
        #print(im2)
        #m_grayscale = im.convert('LA')
        resized = resize(pgm,(xsize,ysize))
        filename = filename[:-4]
        filename = filename.split("/")[-1]
    
        imsave(out_path+filename+'_rsz{}_{}.pgm'.format(xsize,ysize),resized) 

def parse_args():
    """
    lay out the arguments that will be passed into the script
    """
    arg_parser = argparse.ArgumentParser(description="resize images")
    arg_parser.add_argument("image_path",help="director[y | ies] in which the images "+
                            "to be resized are located",nargs="+")
    arg_parser.add_argument("x",help="desired x size of the image",type=int)
    arg_parser.add_argument("y",help="desired y size of the image",type=int)
    arg_parser.add_argument("-out_dir",help="output directory for resized images, defaults to "+
                                            "creating output directories in each of the image paths",type=str)
    return arg_parser

if __name__ == '__main__':
    

    # parse otu args
    args = parse_args().parse_args()
    
    # get a list of image paths, unpack x and y sizes
    im_paths = args.image_path
    x_size = args.x
    y_size = args.y
    
    # out_dir is optional, determine whether it was specified
    out_dir = None
    if args.out_dir:
        out_dir = args.out_dir
    
    # iterate through image paths
    for directory_stem in im_paths:
        directory_stem = directory_stem.strip("/")
        
        
        if len(directory_stem) < 1:
            directory_stem= "."
        
        # get all of the ppm files in this directory
        files = glob.glob(directory_stem + "/*.pgm")
        full_num = len(files)
        # not output directory specified, create one inside 
        # of this directory
        if not out_dir:
            # make a rsz ( resize ) directory if necessary
            if not os.path.exists(directory_stem + "/rsz"):
                print ("making a dir: {}".format(directory_stem + "/rsz"))
                os.makedirs(directory_stem + "/rsz")
            
            out_dir = directory_stem + "/rsz/"
        
        # post setup -- resize and save those pups
        for i,f in enumerate(files):
            resize_and_save(f,out_dir,x_size,y_size,i,full_num)

