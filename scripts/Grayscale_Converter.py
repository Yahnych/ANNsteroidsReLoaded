import os
import sys

PATH = os.getcwd() + "/"

R_CONSTANT = 0.2126
G_CONSTANT = 0.7152
B_CONSTANT = 0.0722

def make_grayscale(file_path, file, folder_out):
    '''
    Convert a r g b ppm file to a grayscale pgm file
    '''
    try:
        in_file  = open(file_path, 'r')
        ext = file.split('.')[-1]
        if (ext != 'ppm'):
            print("Error: Invalid file format:", file)
            return
        out_file = open(folder_out + '/' + file.split('.')[0] +'_grayscale.pgm', 'w')
    except IOError as e:
        print(file, "not found:",e)
    else:
        out_file.write('P2\n600 600\n255\n')
        all_lines = in_file.readlines()
        all_lines = [l.strip() for l in all_lines]
        all_lines = [l.split() for l in all_lines]
        all_lines = all_lines[3:]

        for row in all_lines:
            for i in range(0,len(row),3):
                r = int(R_CONSTANT * int(row[i]))
                g = int(G_CONSTANT * int(row[i+1]))
                b = int(B_CONSTANT * int(row[i+2]))
                grayscale = str(r + g + b)
                out_file.write(grayscale + '\n')
            out_file.write('\n')
        in_file.close()
        out_file.close()

def main():
    folder_name = input("Folder name:")
    folder_out = input("Output folder name:")

    total_files = len(os.listdir(folder_name))
    i = 0
    for file_name in os.listdir(folder_name):
        make_grayscale(folder_name + '/' + file_name, file_name, folder_out)
        i += 1
        if i % 10 == 0:
            print("Processed: ", str(i), "/", total_files, sep='')
    print("Processed: ", total_files, "/", total_files, sep='')
main()
