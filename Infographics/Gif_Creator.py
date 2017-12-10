import os
import imageio
import sys
import warnings
import argparse

def create_gif(png_dir, out_dir, sort=False):
    images = []
    for subdir, dirs, files in os.walk(png_dir):
        if sort == True:
            files.sort(key=get_frame_number)
        print(files)
        for file in files:
            file_path = os.path.join(subdir, file)
            if file_path.endswith(".png"):
                images.append(imageio.imread(file_path))
            else:
                print("Ignoring:", file_path)
    imageio.mimsave(out_dir, images, duration=0.05)
    

def get_frame_number(frame):
    """
    Utility used to sort a list of frames based on frame number.
    Assumes frame is in the format: <frame_number>_<reward>.png
    Frames not in this format return a value of -1
    """
    if frame.endswith('.png'):
        return int(frame.split('_')[0])
    else:
        return -1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images",help="images that will be compiled into a gif"
                        ,type=str)

    parser.add_argument("out_dir",help="out directory for gif",type=str)
    parser.add_argument("gif_name",help="name of gif",type=str)

    return parser

if __name__ == '__main__': 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        args = parse_args().parse_args()

        filenames = args.images
        out_dir   = args.out_dir
        name      = args.gif_name
        if not os.path.exists(filenames):
            print("images folder does not exist")
            sys.exit(-1)
        if not os.path.exists(out_dir):
            print("out directory does not exist")
            sys.exit(-1)

        if not out_dir.endswith('/'):
            out_dir += '/'
        if not name.endswith('.gif'):
            if '.' in name:
                print("Invalid gif name!")
                sys.exit(-1)
            else:
                name +='.gif'
        print(filenames, out_dir, name)
        create_gif(filenames, out_dir + name, sort=True)
