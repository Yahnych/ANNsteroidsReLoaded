import sys
import argparse
import os
import pickle
import matplotlib.pyplot as plt
# plot average reward

def plot_average_reward(path, label=""):
    try:
        with open(path, 'rb') as p:
            data = pickle.load(p)
            plt.plot(data)
            plt.ylabel(label)
            plt.xlabel('Episodes')
            plt.show()
    except IOError as e:
        print("Couldn't open file:", e)

def main(argv):
    parser = argparse.ArgumentParser(description='Plot average reward over n episodes')
    parser.add_argument('pkl_dir', type=str,
                    help='Pickel file to be parsed')
    parser.add_argument("--y_label", help="y-axis-label")
    args = parser.parse_args()
    pickel_path = args.pkl_dir

    if args.y_label:
        plot_average_reward(pickel_path, args.y_label)
    else:
        plot_average_reward(pickel_path)
    

if __name__ == '__main__':
    main(sys.argv)
