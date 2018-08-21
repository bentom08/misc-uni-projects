# B Taylor
# 24/03/17
# Mass-Radius ration of white dwarf stars

from math import pi, sqrt
import matplotlib.pyplot as plt


m_e = 9.1093897e-31 # mass of electron
m_p = 1.6726231e-27 # mas of proton
c = 2.99792458e8 # speed of light in a vacuum
h_bar = 1.05457266e-34 # reduced planck constant
g = 6.67259e-11 # grav acceleration

Y_e = 0.5 # average electrons per nucleon
rho_0 = m_p*m_e**3*c**3/(3*pi**2*h_bar**3*Y_e)

def z(rho):
    return rho/rho_0

def dp_drho(rho):
    gamma = (z(rho)**2)**(1.0/3)/(3*sqrt(1+ (z(rho)**2)**(1.0/3)))
    return Y_e*m_e*c**2*gamma/m_p


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

#----------------------------------------------------------------------------

def derivs(n, x, y):
    dy=[0 for i in range(0,n+1)]
    dy[1]= 4*pi*x**2*y[2]
    if y[1] == 0:
        dy[2] == 0
    else:
        dy[2]= -1*(dp_drho(y[2]))**(-1)*g*y[1]*y[2]/x**2
    return dy

#----------------------------------------------------------------------------

solarm = 1.98e+30
solarr = 6.95e+8
rhoc = 1e+7
rho = []
m = []
r = []
y = [0, 0, 0]
sirius = 0
eri = 0
stein = 0

while y[1]/solarm < 1.5:
    x= 0; y=[0, 0, rhoc]
    while y[2] > 0:
        (x,y) = runkut(2, x, y, 1000)
        #r.append(x)
        #m.append(y[1])
        #rho.append(y[2])
    rhoc *= 1.1
    r.append(x/solarr)
    m.append(y[1]/solarm)
    if y[1]/solarm > 1.053 and sirius == 0:
        print "Sirius B:", x/solarr
        sirius = 1
        print Y_e
    if y[1]/solarm > 0.48 and eri == 0:
        print "40 Eri B:",x/solarr
        eri = 1
    if y[1]/solarm > 0.5 and stein == 0:
        print "Stein 2051:",x/solarr
        stein = 1

while y[2] > 0:
    (x,y) = runkut(2, x, y, 1000)
    r.append(x)
    m.append(y[1])
    rho.append(y[2])
"""
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(r, m, 'b', label='m')
axarr[0].grid()
axarr[1].plot(r, rho, 'g', label='rho')
axarr[1].grid()
"""

plt.plot(m, r)
plt.legend(loc='best')
plt.ylabel("R (Solar Radii)")
plt.xlabel('M (Solar Masses)')
plt.grid()

plt.show()