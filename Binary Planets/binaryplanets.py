
from math import cos, sin, sqrt
import pylab as plt

R = 1.0 # distance of each star from origin
Om = 1.0 # angular velocity of stars

def planet(x, y, t):
    
    X1 = R*cos(Om*t)
    Y1 = R*sin(Om*t)
    X2 = -R*cos(Om*t)
    Y2 = -R*sin(Om*t)
    
    r1 = sqrt((x - X1)**2 + (y - Y1)**2)
    r2 = sqrt((x - X2)**2 + (y - Y2)**2)
    r = [r1, r2]
    
    return r
    
def runkut(n, x, y, h):
    "Advances the solution of diff eqn defined by derivs from x to x+h"
    y0=y[:]
    
    k1=derivs(n, x, y)
    for i in range(1,n+1):
        y[i]=y0[i]+0.5*h*k1[i]
        
    k2=derivs(n, x+0.5*h, y)
    for i in range(1,n+1):
        y[i]=y0[i]+h*(0.2071067811*k1[i]+0.2928932188*k2[i])
        
    k3=derivs(n, x+0.5*h, y)
    for i in range(1,n+1): 
        y[i]=y0[i]-h*(0.7071067811*k2[i]-1.7071067811*k3[i])
        
    k4=derivs(n, x+h, y)
    for i in range(1,n+1):
        a=k1[i]+0.5857864376*k2[i]+3.4142135623*k3[i]+k4[i]
        y[i]=y0[i]+0.16666666667*h*a
        
    x+=h
    return (x,y)

def derivs(n, x, y):
    dy=[0 for i in range(0,n+1)]
    dy[1] = y[2]
    dy[2] = -R**3*Om**2*((y[1] - R*cos(Om*x))/(planet(y[1], y[3], x)[0])**3 + (y[1] + R*cos(Om*x))/(planet(y[1], y[3], x)[1])**3)
    dy[3] = y[4]
    dy[4] = -R**3*Om**2*((y[3] - R*sin(Om*x))/(planet(y[1], y[3], x)[0])**3 + (y[3] + R*sin(Om*x))/(planet(y[1], y[3], x)[1])**3)
    
    return dy

iX = -1.2 # inital X coordinate
iY = 0.0   # inital Y coordinate
x=0.0; y=[0, iX, 0.0, iY, -2.5]
N = 30000 # step number in 1 second (1/step size)
xVals = [iX]
yVals = [iY]
starX = [R]
starY = [0]
time = 10.0


for j in range(0,N):
    (x,y) = runkut(4, x, y, time/N)
    xVals.append(y[1])
    yVals.append(y[3])
    starX.append(R*cos(Om*x))
    starY.append(R*sin(Om*x))

starP = int(6.27/time*N)

orbitX = xVals[-starP:]
orbitY = yVals[-starP:]
plt.plot(xVals, yVals)
plt.plot(starX, starY, 'k')
plt.plot(orbitX, orbitY, 'r')
plt.axis('equal')
plt.grid()
plt.xlabel('R')
plt.ylabel('R')

plt.show()