from math import cos, sin, sqrt
import pylab

def runkut(v, t, f, k, n, h):
    k1 = diffEq(v, t, f, k, n)
    k2 = diffEq(v + h/2*k1, t + h/2, f, k, n)
    k3 = diffEq(v + h/2*k2, t + h/2, f, k, n)
    k4 = diffEq(v + h*k3, t + h, f, k, n)
    v = v + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return v, t+h
    
def diffEq(v, t, f, k, n):
    # Mass = 1kg
    # Length = 1m
    return f*cos(t*(1-n)) - k*v/sqrt(9.8) - sin(theta)
    
N = 100 # steps per second (i.e. 1/step size)
f = 1 # maximum force applied to pendulum in F = mg (mass in kg * g = 9.8)
k = 0. # damping coefficient
n = 0. # proportional difference between the force variation and the natural frequency for small oscillations (0 to ~0.5)

t = 0 # inital time
v = 0 # inital angular velocity of pendulum
global theta
theta = 0 # inital angle of pendulum

thetaVals = [theta]
vVals = [v]
thetaVals1 = [] # Values for last few osillations
vVals1 = []  # Values for last few osillations

for i in range(100000):
    v, t = runkut(v, t, f, k, n, 1.0/N)
    theta = theta + v*1.0/N
    thetaVals.append(theta)
    vVals.append(v)

for i in range(5000):
    v, t = runkut(v, t, f, k, n, 1.0/N)
    theta = theta + v*1.0/N
    thetaVals1.append(theta)
    vVals1.append(v)

fig, ax = pylab.subplots()
ax.plot(thetaVals, vVals, 'b')
ax.plot(thetaVals1, vVals1, 'r')
ax.grid(True)
pylab.xlabel("Angular displacement (radians)")
pylab.ylabel("Angular velocity (rad/s)")

pylab.show()