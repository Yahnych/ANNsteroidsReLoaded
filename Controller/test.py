from subprocess import Popen, PIPE
import random
def play_game():
    p = Popen(['./Asteroids'],  stdout=PIPE, stdin=PIPE)
    
    i = 0
    
    left = random.randint(0, 1)
    right = random.randint(0, 1)
    up = random.randint(0, 1)
    space = random.randint(0, 1)
    while (True):
        
        
        
        
        value = str(left) + str(right) + str(up) + str(space) + '\n'
        value = bytes(value, 'UTF-8')  # Needed in Python 3.
        p.stdin.write(value)
        p.stdin.flush()
        result = p.stdout.readline().strip()
        print(result)
        
        if i % 5 == 0:
            left = random.randint(0, 1)
            right = random.randint(0, 1)
            up = random.randint(0, 1)
            space = random.randint(0, 1)
        
        i+=1
def test1():
    p = Popen(['./Asteroids'],  stdout=PIPE, stdin=PIPE)
    for ii in range(1000):
        value = str(ii) + '\n'
        value = bytes(value, 'UTF-8')  # Needed in Python 3.
        p.stdin.write(value)
        p.stdin.flush()
        result = p.stdout.readline().strip()
        print(result)
def test2():
    p = Popen(['./runner'],  stdout=PIPE, stdin=PIPE)
    value = input("Gimme some text:")
    while (value != 'q'):
        value += '\n'
        value = bytes(value, 'UTF-8')  # Needed in Python 3.
        p.stdin.write(value)
        p.stdin.flush()
        message = p.stdout.readline()
        print ("return message ->" + str(message) + " written by python \n")
        value = input("Gimme some text:")
def test3():
    p = Popen(['./Asteroids'],  stdout=PIPE, stdin=PIPE)
    value = input("Gimme some text:")
    while (value != 'q'):
        value += '\n'
        value = bytes(value, 'UTF-8')  # Needed in Python 3.
        p.stdin.write(value)
        p.stdin.flush()
        message = p.stdout.readline()
        print ("return message ->" + str(message) + " written by python \n")
        value = input("Gimme some text:")
    
play_game()
