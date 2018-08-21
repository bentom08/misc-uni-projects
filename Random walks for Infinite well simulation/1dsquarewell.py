from math import pi, log
from random import randrange
import numpy
import pylab

J = 8.0

maxwalk = 200

def walk(J, maxwalk):
    j = 0
    n = 0
    while abs(float(j)) < J and n < maxwalk:
        if randrange(0,2) == 0:
            j -= 1
        else :
            j += 1
        n += 1
    return n

a = []
for i in range(10000):
    a.append(walk(J, maxwalk))
  
b = [0 for i in range(maxwalk)]

for i in range(len(a)):
    for j in range(maxwalk):
        if a[i] == j+1:
            b[j] += 1

for i in range(len(b)/2):
    b.remove(b[i])
    
for i in range(len(b)):
    if b[i] == 0:
        pass
    else:
        b[i] = log(b[i])
x = numpy.arange(0, maxwalk, 2)
pylab.plot(x, b)
z = numpy.polyfit(x, b, 1)
p = numpy.poly1d(z)
pylab.plot(x,p(x))
pylab.show()

print -z[0]*J**2, pi**2/8