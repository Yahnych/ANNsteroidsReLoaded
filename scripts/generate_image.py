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

def resize_and_save(filename,out_path,xsize,ysize):
    print ("opening {}".format(filename))
    #im = Image.open(filename)
    im2 = imread(filename)
    #m_grayscale = im.convert('LA')
    resized = resize(im2,(xsize,ysize))
    filename = filename[:-4]
    filename = filename.split("/")[-1]

    imsave(out_path+filename+'_rsz{}_{}.ppm'.format(xsize,ysize),resized)        
    #im_resized.save(out_path + filename+'_rsz{}_{}.png'.format(xsize,ysize))


def parse_args():
    arg_parser = argparse.ArgumentParser(description="resize images")
    arg_parser.add_argument("image_path",help="director[y|ies] in which the images "+
                            "to be resized are located",nargs="+")
    arg_parser.add_argument("x",help="desired x size of the image",type=int)
    arg_parser.add_argument("y",help="desired y size of the image",type=int)
    arg_parser.add_argument("-out_dir",help="output directory for resized images, defaults to "+
                                            "creating output directories in each of the image paths",type=str)
    return arg_parser

if __name__ == '__main__':
    args = parse_args().parse_args()
    im_paths = args.image_path
    x_size = args.x
    y_size = args.y
    
    out_dir = None
    if args.out_dir:
        out_dir = args.out_dir
    
    
    for directory_stem in im_paths:
        directory_stem = directory_stem.strip("/")
        
        if len(directory_stem) < 1:
            directory_stem= "."
        files = glob.glob(directory_stem + "/*.ppm")
        # not output directory specified, create one inside 
        # of this directory
        if not out_dir:
            
            if not os.path.exists(directory_stem + "/rsz"):
                print ("making a dir: {}".format(directory_stem + "/rsz"))
                os.makedirs(directory_stem + "/rsz")
            
            out_dir = directory_stem + "/rsz/"
        
        for f in files:
            resize_and_save(f,out_dir,x_size,y_size)

