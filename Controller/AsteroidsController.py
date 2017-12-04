from subprocess import Popen, PIPE
import random
import sys
import argparse
import os
from time import sleep
def play_game(p):
    
    
    i = 0

    result = ""
    
    left = random.randint(0, 1)
    right = random.randint(0, 1)
    #up = random.randint(0, 1)
    up = 0
    space = random.randint(0, 1)
    
    while (i < 200 and 'e' not in str(result)):
        value = str(left) + str(right) + str(up) + str(space) + '\n'
        value = bytes(value, 'UTF-8')  # Needed in Python 3.
        p.stdin.write(value)
        p.stdin.flush()
        result = p.stdout.readline().strip()
        result = result.decode("utf-8")
        print(i, str(result))

        if i % 2 == 0:
            left = random.randint(0, 1)
            right = random.randint(0, 1)
            #up = random.randint(0, 1)
            up = 0
            space = random.randint(0, 1)
        i+=1
    print('Game ended.')
    value = str('d') + str('d') + str('d') + str('d') + '\n'
    value = bytes(value, 'UTF-8')  # Needed in Python 3.
    p.stdin.write(value)
    
    p.stdin.flush()
    result = p.stdout.readline().strip()
    result = result.decode("utf-8")
    print(str(result))


    wait_time = 0
    response = ""
    while (response != 'd'):
        p.stdin.write(bytes('k\n', 'UTF-8'))
        p.stdin.flush()
        result = p.stdout.readline().strip()
        result = result.decode("utf-8")
        response = str(result)
        print(response)
        wait_time+=1
    
    p.kill()
    '''
    response = ""
    wait_time = 0
    while (reponse != 'd' and wait_time < 10):
        #p.stdin.write(value)
        #p.stdin.flush()
        #result = p.stdout.readline().strip()
        #result = result.decode("utf-8")
        #response = str(result)
        wait_time+=1

    print ("sleeping for 10")
    sleep(10)
    p.kill()
    print("Done")
    '''

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
    try:
        play_game(process)
    except KeyboardInterrupt:
        process.kill()
        sys.exit("Caught keyboard interrupt")

if __name__ == "__main__":
    main(sys.argv)
