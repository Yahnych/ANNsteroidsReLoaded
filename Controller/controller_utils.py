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
import argparse
import warnings
from skimage.transform import resize
from skimage.io import imsave
from scipy.misc import imread
import glob
import time

X_SIZE = 84
Y_SIZE = 84
N_CHANNELS = 4
VOLUME = 1

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE    = '\x1b[2K'
CURSOR_DOWN_ONE = "\x1b[B"
BUFFER_LINE   = CURSOR_UP_ONE + ERASE_LINE 

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
    parser.add_argument("--out_dir",help="output directory for resized images, defaults to "+
                            "creating output directories in each of the image paths",type=str)
    
    parser.add_argument("--model_path",help="load path of existing model",type=str)
    
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

def resize_and_save(filename,out_path,xsize,ysize):
    """
    resize an image given by the [filename], by the specified [xsize] and [ysize] dimensions
    writing the new image to the [out_path].
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #im_resized.save(out_path + filename+'_rsz{}_{}.png'.format(xsize,ysize))
        
        #if curr != 0:
         #   print (BUFFER_LINE + "resizing {:<25} --> {}/{} {}".format(filename,curr,full,load_bar(curr,full)))
        
       # else:
        #    print ("resizing {:<25} --> {}/{} {}".format(filename,curr,full,load_bar(curr,full)))
        pgm = pgm2pil(filename)

        resized = resize(pgm,(xsize,ysize))
        filename = filename[:-4]
        filename = filename.split("/")[-1]
    
        imsave(out_path+filename+'_rsz.pgm',resized) 
        
        return out_path+filename+'_rsz.pgm'
    
def get_flat_image(filename):
    img = imread(filename)
    return np.reshape(img,[img.shape[0]**2])

def next_state_and_reward(curr_photos,old_photos,img_pth,screenshot_location,out_dir,buffer_string,epsilon,total_steps,action): #logger):
    """
    Get the next image and reward
    """
    while (len(curr_photos - old_photos) < 1):
        curr_photos |= set(glob.glob(img_pth))
        
    # there are new photos, wait for them to be written to file
    new_photos = list(curr_photos - old_photos)
    print (buffer_string) #+ " found new photos ... sleeping for wrap-up [1 SEC] ")
    time.sleep(1)
    
    # yup ... unpack reward and photo id in same line and shove it 
    # in a list
    new_photos_rewards = [[int(photo_name.split("/")[1].split("_")[0]),int(photo_name.split("/")[1].split("_")[1].split(".")[0])] for photo_name in new_photos]
    
    new_photos = [tup[0] for tup in new_photos_rewards]
    new_rewards = [tup[1] for tup in new_photos_rewards]
    
    # score at this timestep
    score_t = new_rewards[new_photos.index(max(new_photos))]
    
    most_recent_regex = screenshot_location.strip("/") +"/%03d*pgm" % max(new_photos)
    most_recent_photo = glob.glob(most_recent_regex)[0]
    
    buffer_string = "[+] e = %.03f " % epsilon + " | total_steps =  %05d" %total_steps  +" | action: {}".format(action) + " | img: " + most_recent_photo
    #logger.info(buffer_string)
    # image to process
    rsz_photo = resize_and_save(most_recent_photo,out_dir,X_SIZE,Y_SIZE)
    
    return get_flat_image(rsz_photo),score_t,buffer_string,curr_photos,old_photos

def next_frame_thats_it(curr_photos,old_photos,img_pth,screenshot_location,out_dir):
    """
    Get the next state, THATS IT
    """
    
    while (len(curr_photos - old_photos) < 1):
        curr_photos |= set(glob.glob(img_pth))
    
    time.sleep(1)
    
    new_photos = list(curr_photos - old_photos)
    new_photos_rewards = [[int(photo_name.split("/")[1].split("_")[0]),int(photo_name.split("/")[1].split("_")[1].split(".")[0])] for photo_name in new_photos]
    new_photos = [tup[0] for tup in new_photos_rewards]
    
    most_recent_regex = screenshot_location.strip("/") +"/%03d*pgm" % max(new_photos)
    most_recent_photo = glob.glob(most_recent_regex)[0]
    
    rsz_photo = resize_and_save(most_recent_photo,out_dir,X_SIZE,Y_SIZE)
    
    return get_flat_image(rsz_photo)
    
    

class FrameBuffer:
    def __init__(self,size=4):
        self.stack = []
        self.size  = size
    
    def deep_copy(self):
        #print(type(self.stack))
        temp = FrameBuffer()
        temp.stack.extend(self.stack)
       
        return temp
    
    def add(self, frame):
        self.stack.append(frame)
        N_too_many = len(self.stack) - self.size
        
        if N_too_many > 0:
            #print ("N too many: {}".format(N_too_many))
            for i in range(N_too_many):
                self.stack.pop(0)