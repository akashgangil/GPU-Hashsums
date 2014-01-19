import string
import random
import sys

def randomMessage(n):
    f = open('input.in', 'w+')
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(int(n))]
    message = "".join(lst)
    f.write(message)
    f.close()
    
if __name__ == "__main__":
    randomMessage(sys.argv[1])
