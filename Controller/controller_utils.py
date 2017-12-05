#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:12:15 2017

@author: danberenberg
"""

"""
train a controller using a deep q network

Adapted primarily from Arthur Juliani

[https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb]
"""

import numpy as np
#import random
#import sys, os
import argparse
import warnings
from skimage.transform import resize
from skimage.io import imsave

X_SIZE = 84
Y_SIZE = 84

def setup_args():
    """
    Set up arguments for the scripts
    
    IN:
        None
        
    OUT:
        parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run DRL agent')
    parser.add_argument('ast_path', type=str,
                    help='Path of asteroids executable')
    parser.add_argument('picture_path', type=str,
                    help='Directory where screenshots will be stored')
    parser.add_argument("-out_dir",help="output directory for resized images, defaults to "+
                            "creating output directories in each of the image paths",type=str)
    
    args = parser.parse_args()
    
    return args

def write_to_game(process,value):
    """
    Write an action to the game
    
    IN:
        process -- the game to write to
        value   -- the value to write to the game
                   
    OUT:
        None
    """

    value = bytes(value,'UTF-8') # python 3
    process.stdin.write(value)
    process.stdin.flush()

def read_from_game(process):
    """
    Read a response from the game
    
    IN:
        process -- the game to read from
        
    OUT:
        result -- the response from the game
    """
    
    result = process.stdout.readline().strip()
    result = result.decode("utf-8")
    
    return result

def pgm2pil(filename):
    """
    Convert a pgm file to Python Imaging Library format
    """
    try:
        
        # header contents
        inFile = open(filename)
        header = None
        size = None
        maxGray = None
        data = []
        
        
        for line in inFile:
            # take of /n
            stripped = line.strip()
            
            # ignore comments
            if stripped[0] == '#': 
                continue
            
            # detect header (first line in pgm file)
            elif header == None: 
                if stripped != 'P2': return None
                header = stripped
            
            # detect size - a 2d vector height,width
            elif size == None:
                size = [int(elt) for elt in stripped.split()]
                
            # maximum gray value
            elif maxGray == None:
                maxGray = int(stripped)
                
            # going inside this block means the reader
            # has reached the file's pixel body
            else:
                for item in stripped.split():
                    data.append(int(item.strip()))
        
        # resize the data and make it a ratio
        data = np.reshape(data, (size[1],size[0]))/255.0#/float(maxGray)*255
        
        # flip array upside down and return it
        return np.flipud(data)

    except Exception as e:
        print(e);
        pass

    return None

def resize_and_save(filename,out_path,xsize,ysize,curr,full):
    """
    resize an image given by the [filename], by the specified [xsize] and [ysize] dimensions
    writing the new image to the [out_path].
    """
    with warnings.catch_warnings():
        #warnings.simplefilter('ignore')
        #im_resized.save(out_path + filename+'_rsz{}_{}.png'.format(xsize,ysize))
        
        #if curr != 0:
         #   print (BUFFER_LINE + "resizing {:<25} --> {}/{} {}".format(filename,curr,full,load_bar(curr,full)))
        
       # else:
        #    print ("resizing {:<25} --> {}/{} {}".format(filename,curr,full,load_bar(curr,full)))
        #im = Image.open(filename)
        pgm = pgm2pil(filename)
    
        #im2 = imread(filename)
        #print(im2)
        #m_grayscale = im.convert('LA')
        resized = resize(pgm,(xsize,ysize))
        filename = filename[:-4]
        filename = filename.split("/")[-1]
    
        imsave(out_path+filename+'_rsz{}_{}.pgm'.format(xsize,ysize),resized) 
