from subprocess import Popen, PIPE
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
    
test1()
