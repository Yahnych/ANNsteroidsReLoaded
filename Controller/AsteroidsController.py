from subprocess import Popen, PIPE
import random
import sys


def play_game(p):
    
    
    i = 0

    result = ""
    
    left = random.randint(0, 1)
    right = random.randint(0, 1)
    up = random.randint(0, 1)
    space = random.randint(0, 1)
    while (result != 'over'):
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
    asteroids_location = ""
    if (len(argv) == 1):
        asteroids_location = input("Enter the file path of Asteroids:\n")
    elif (len(argv) == 2):
        asteroids_location = sys.argv[1]
    process = Popen([asteroids_location],  stdout=PIPE, stdin=PIPE)
    play_game(process)

if __name__ == "__main__":
    main(sys.argv)
