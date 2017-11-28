from subprocess import Popen, PIPE
import random
import sys
import argparse
import os

def play_game(p):
    
    
    i = 0

    result = ""
    
    left = random.randint(0, 1)
    right = random.randint(0, 1)
    up = random.randint(0, 1)
    space = random.randint(0, 1)
    while (i < 300):
        value = str(left) + str(right) + str(up) + str(space) + '\n'
        value = bytes(value, 'UTF-8')  # Needed in Python 3.
        p.stdin.write(value)
        p.stdin.flush()
        result = p.stdout.readline().strip()
        result = result.decode("utf-8")
        print(str(result))
        
        if i % 2 == 0:
            left = random.randint(0, 1)
            right = random.randint(0, 1)
            up = random.randint(0, 1)
            space = random.randint(0, 1)
        
        i+=1

    p.kill()
    print("Done")

def main(argv):
    parser = argparse.ArgumentParser(description='Run DRL agent')
    parser.add_argument('ast_path', type=str,
                    help='Path of asteroids executable')
    parser.add_argument('picture_path', type=str,
                    help='Directory where screenshots will be stored')
    args = parser.parse_args()
    
    asteroids_location = args.ast_path
    screenshot_location = args.picture_path

    #if (not os.path.exists(screenshot_location)):
#        print("You done fucked up")
#        sys.exit(1)
    
    process = Popen([asteroids_location, screenshot_location],  stdout=PIPE, stdin=PIPE) 
    play_game(process)

if __name__ == "__main__":
    main(sys.argv)
