# B Taylor
# 23/03/16
# 2-Dimensional Circular well random walk

# import modules needed
from math import pi, log
from random import randrange
import numpy
import pylab

J = 20.0 # distance from origin that the walk will stop. Set higher to reduce systematic error.

maxwalk = 1500 # number of steps before the program will automatically terminate

"""
    The walk function simulates one random walk and determines how long it 
takes a single "drunk man" to either "get arrested" or take the maximum number
of steps (maxwalk). 

    It takes the arguments J, which determines how far from the
origin the walker needs to get before getting arrested, and maxwalk which
determines how many steps the walker has to take before the loop terminates.

    The function returns the number of steps the walker took before getting
arrested. 

"""
def walk(J, maxwalk):
    j1 = 0 # the x coordinate of the walker
    j2 = 0 # the y coordinate of the walker
    n = 0 # the numbver of steps the walker has taken
    # loop will run until either the walker is more than distance J from the
    # origin or has taken more than maxwalk steps
    while (float(j1)**2 + float(j2)**2 < J**2) and n < maxwalk:
        # the function randrange is used to choose a direction at random
        direction = randrange(0,4)
        # the relevant x or y coordinate is adjusted accordingly
        if direction == 0:
            j1 -= 1
        elif direction == 1:
            j1 += 1
        elif direction == 2:
            j2 -= 1
        elif direction == 3:
            j2 += 1
        n += 1 # 1 is added to the step count each iteration of the loop
    return n # the step count is returned when the loop is broken

a = [] # this array is used to store the lengths of all the walks taken
for i in range(100000): # this is the number of walks taken
    a.append(walk(J, maxwalk)) # the result of the walk function is appended
  
  
# this array is used to organise the data into "bins" so it can be displayed
# on a graph. The array starts as an array of length maxwalk and filled with
# 0's.
b = [0 for i in range(maxwalk)] 

for i in range(len(a)): 
    b[a[i]-1] += 1 # adds the value obtained to the relevant bin

# goes through each value and finds the log, to create the log graph
for i in range(len(b)):
    if b[i] == 0:
        pass
    else:
        b[i] = log(b[i])
        
# a plot of the log graph
pylab.subplot(211)
pylab.plot(range(len(b)), b)

# trunctates the data to the relevant part (easiest to get a straight line)
b = b[:6*maxwalk/10]
b = b[int(J*10):]

# creates a new plot with the newly truncated data
x = numpy.arange(J*10, 6*maxwalk/10)
pylab.subplot(212)  
pylab.plot(x, b)

# creates and plots a linear line of best fit 
z = numpy.polyfit(x, b, 1)
p = numpy.poly1d(z)
pylab.plot(x,p(x))

pylab.show() # shows the plots created

print -z[0]*J**2, pi**2/8