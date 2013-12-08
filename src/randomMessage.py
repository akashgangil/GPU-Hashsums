import string
import random
import sys

def randomMessage(n):
    f = open('input.in', 'w+')
    for i in xrange(int(n)):
      lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(1000000)]
      message = "".join(lst)
      f.write(message)
    f.close()
    
if __name__ == "__main__":
    randomMessage(sys.argv[1])
