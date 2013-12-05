import string
import random
import sys

def randomMessage(n):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(int(n))]
    message = "".join(lst)
    f = open('input.in', 'w')
    f.write(message)
    f.close()
    
if __name__ == "__main__":
    print "argument is" + sys.argv[1]
    randomMessage(sys.argv[1])
