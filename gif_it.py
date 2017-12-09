#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:42:08 2017

@author: danberenberg
"""

import sys
import argparse

try:
    import imageio
except ImportError:
    sys.exit("idiotically, you chose the wrong python")

VALID_EXTENSIONS = ('png', 'jpg')

def create_gif(filenames,out_dir,name):

    with imageio.get_writer(out_dir.strip("/") + "/" + name.strip(".gif")  + ".gif", mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images",help="images that will be compiled into a gif"
                        ,nargs="+",type=str)

    parser.add_argument("out_dir",help="out directory for gif",type=str)
    parser.add_argument("gif_name",help="name of gif",type=str)

    return parser
if __name__ == '__main__': 
    
    #script = sys.argv[0]
    
    #if len(sys.argv) < 3:
    #    print('Usage: python {} <p> <path to images separated by space>'.format(script))
    #    sys.exit(1)
    
    #p = float(sys.argv[1])
    args = parse_args().parse_args()

    filenames = args.images
    out_dir   = args.out_dir
    name      = args.gif_name

    if not all(f.lower().endswith(VALID_EXTENSIONS) for f in filenames):
        print('Only png and jpg files allowed')
        sys.exit(1)
  
          
    create_gif(filenames,out_dir,name)
