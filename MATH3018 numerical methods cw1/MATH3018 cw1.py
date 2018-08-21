# B Taylor 22/11/2016
# MATH 3018  Numerical Methods Coursework 1

from scipy.optimize import fsolve
import numpy as np
import pylab

def MyRK3_step(f, t, qn, dt, options=(1, 1, 1)):
    """
    Task 1    
    
    A single step of the third order Runge-Kutta method. 
    
    Inputs are: 
    
    f: function, ODE to solve
    
    t: float, time
    
    qn: np array or single number, initial q value
    
    dt: float, time step taken
    
    options: 3-tuple, 1st is gamma, 2nd is epsilon, 3rd is omega
    
    returns the next RK3 step q(n+1) as a np array of 2
    """

    assert(hasattr(f, '__call__')), \
    "f must be a callable function"
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and\
    np.all(np.isreal(t))), \
    "t must be real and finite"
    assert((not np.any(np.isnan(qn))) and np.all(np.isfinite(qn)) and\
    np.all(np.isreal(qn))), \
    "qn must be real and finite"
    assert(type(qn) == np.ndarray), \
    "qn must be a numpy array, integer, float or long integer"
    assert((not np.any(np.isnan(dt))) and np.all(np.isfinite(dt)) and\
    np.all(np.isreal(dt))), \
    "dt must be real and finite"
    assert(dt > 0), \
    "dt must be greater than zero"
    assert((type(options) == tuple) and (len(options) == 3)), \
    "options must be tuple of length 3"
    
    k1 = f(t, qn, options)
    k2 = f(t + dt/2, qn + dt/2*k1, options)
    k3 = f(t + dt, qn + dt*(2*k2 - k1), options)
    
    return qn + dt/6*(k1 + 4*k2 + k3)
    
def MyGRRK3_step(f, t, qn, dt, options=(1, 1, 1)):
    """
    Task 2    
    
    A single step of the third order Gauss-Radau Runge-Kutta method.

    Inputs are: 
    
    f: function, ODE to solve
    
    t: float, time
    
    qn: np array or single number, initial q value
    
    dt: float, time step taken
    
    options: 3-tuple, 1st is gamma, 2nd is epsilon, 3rd is omega
    
    returns the next GRRK3 step q(n+1) as a np array of 2
    """
    
    assert(hasattr(f, '__call__')), \
    "f must be a callable function"
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and\
    np.all(np.isreal(t))), \
    "t must be real and finite"
    assert((not np.any(np.isnan(qn))) and np.all(np.isfinite(qn)) and\
    np.all(np.isreal(qn))), \
    "qn must be real and finite"
    assert(type(qn) == np.ndarray), \
    "qn must be a numpy array"
    assert((not np.any(np.isnan(dt))) and np.all(np.isfinite(dt)) and\
    np.all(np.isreal(dt))), \
    "dt must be real and finite"
    assert(dt > 0), \
    "dt must be greater than zero"
    assert((type(options) == tuple) and (len(options) == 3)), \
    "options must be tuple of length 3"
    
    k1 = f(t + dt/3, qn, options)
    k2 = f(t + dt, qn, options)

    g = lambda k1: k1 - f(t + dt/3, qn + dt/12*(5*k1 - k2), options)                  
    h = lambda k2: k2 - f(t + dt, qn + dt/4*(3*k1 + k2), options)

    k1 = fsolve(g, k1)
    k2 = fsolve(h, k2)
  
    return qn + dt/4*(3*k1 + k2)


def f(t, q, options=(1, 1, 1)):
    """
    Task 1    
    
    Function containing the differential equation to be solved.
    
    Inputs are:
    
    t: float, time
     
    q: np array of length 2, data at time t (x, y values)
    
    options: 3-tuple, 1st is gamma, 2nd is epsilon, 3rd is omega
             all are set to 1 by default
    
    returns dq/dt as a np array of 2
    """
    
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and\
    np.all(np.isreal(t))), \
    "t must be real and finite"
    assert((not np.any(np.isnan(q))) and np.all(np.isfinite(q)) and\
    np.all(np.isreal(q))), \
    "q must be real and finite"
    assert(len(q) == 2), \
    "q must be length 2"
    assert(type(q) == np.ndarray), \
    "q must be a numpy array"
    assert((type(options) == tuple) and (len(options) == 3)), \
    "options must be tuple of length 3"
    
    gamma, ep, om = options

    matrix1 = np.matrix([[gamma, ep], 
                        [ep, -1]])
  
    matrix2 = np.matrix([[(-1 + q[0]**2 - np.cos(t))/(2*q[0])],
                       [(-2 + q[1]**2 - np.cos(om*t))/(2*q[1])]])
                       
    matrix3 = np.matrix([[np.sin(t)/(2*q[0])],
                       [om*np.sin(om*t)/(2*q[1])]])

    ans = matrix1 * matrix2 - matrix3
    
    return np.array([ans[0,0], ans[1,0]])
    
 
def q(x, y):
    """
    Values of x and y at time, t.
    """ 
    return np.array([x, y])

     
def non_stiff_plots():
    """
    Function to plot RK3 and GRRK3 methods for a non-stiff example
    
    Task 3
    """
    t = 0
    dt = 0.05
    qn = q(np.sqrt(2), np.sqrt(3))
    options = -2, 0.05, 5
    
    RKtvals = [0]
    RKxvals = [qn[0]]
    RKyvals = [qn[1]]
    
    while t <= 1:
        qn = MyRK3_step(f, t, qn, dt, options)
        RKxvals.append(qn[0])
        RKyvals.append(qn[1])
        t += dt
        RKtvals.append(t)
    
    t = 0
    qn = q(np.sqrt(2), np.sqrt(3))
    
    GRtvals = [0]
    GRxvals = [qn[0]]
    GRyvals = [qn[1]]    
        
    while t <= 1:
        qn = MyGRRK3_step(f, t, qn, dt, options)
        GRxvals.append(qn[0])
        GRyvals.append(qn[1])
        t += dt
        GRtvals.append(t)
        
    x = np.linspace(0, 1, 50)
    y = np.sqrt(1 + np.cos(x))
    
    w = np.linspace(0, 1, 50)
    z = np.sqrt(2 + np.cos(5*w))
    
    ax1 = pylab.subplot(121)
    ax2 = pylab.subplot(122)
    ax1.plot(RKtvals, RKxvals, "-b", label="RK3")
    ax1.plot(GRtvals, GRxvals, "-g", label="GRRK3")
    ax1.plot(x, y, "-r", label="exact")
    ax2.plot(RKtvals, RKyvals, "-b")
    ax2.plot(GRtvals, GRyvals, "-g")
    ax2.plot(w, z, "-r")
    ax1.set_xlim(0,1)
    ax2.set_xlim(0,1)
    ax1.set_xlabel("t")
    ax2.set_xlabel("t")
    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    ax1.set_title("x = sqrt(1 + cos(t))")
    ax2.set_title("y = sqrt(2 + cos(5t))")
    pylab.show()

def non_stiff_convergence():
    """
    Function to plot the convergence for each method for a non-stiff example.
    
    Task 4
    """
    t = 0
    dt = 0.1
    qn = q(np.sqrt(2), np.sqrt(3))
    options = -2, 0.05, 5
    
    RKyvals = [[qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]]]
    RKtvals = [[0], [0], [0], [0], [0], [0], [0], [0]]
    RKdtvals = []
    
    for i in range(8):
        while t <= 1:
            qn = MyRK3_step(f, t, qn, dt, options)
            RKyvals[i].append(qn[1])
            t += dt
            RKtvals[i].append(t)
        RKdtvals.append(dt)
        dt /= 2
        t = 0
        qn = q(np.sqrt(2), np.sqrt(3))
    
    dt = 0.1
    GRyvals = [[qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]]]
    GRtvals = [[0], [0], [0], [0], [0], [0], [0], [0]]
    GRdtvals = []
    
    for i in range(8):
        while t <= 1:
            qn = MyGRRK3_step(f, t, qn, dt, options)
            GRyvals[i].append(qn[1])
            t += dt
            GRtvals[i].append(t)
        GRdtvals.append(dt)
        dt /= 2
        t = 0
        qn = q(np.sqrt(2), np.sqrt(3))
        
    RKerror = []
    GRerror = []
    
    for i in range(8):
        total = 0
        for j in range(len(RKyvals[i])):
            total += abs(RKyvals[i][j] - np.sqrt(2 + np.cos(5*RKtvals[i][j])))
        RKerror.append(total*RKdtvals[i])
      
    for i in range(8):
        total = 0
        for j in range(len(GRyvals[i])):
            total += abs(GRyvals[i][j] - np.sqrt(2 + np.cos(5*GRtvals[i][j])))
        GRerror.append(total*GRdtvals[i])
    
    prk = np.polyfit(RKdtvals, RKerror, 3)
    prk1 = np.poly1d(prk)
    x = np.linspace(0, RKdtvals[0], 50)
    y = prk1(x)
    pgr = np.polyfit(GRdtvals, GRerror, 3)
    pgr1 = np.poly1d(pgr)
    x1 = np.linspace(0, GRdtvals[0], 50)
    y1 = pgr1(x1)
    
    
    pylab.plot(RKdtvals, RKerror, "-b", label="RK3 error")
    pylab.plot(GRdtvals, GRerror, "-g", label="GRRK3 error")
    pylab.plot(x, y, "--b")
    pylab.plot(x1, y1, "--g")
    pylab.xlabel("dt")
    pylab.ylabel("1-norm error for y(t)")
    pylab.xlim(0, 0.1)
    pylab.ylim(0, 0.00075)
    pylab.show()
    
def stiff_RK3():
    """
    Function to plot the RK3 method for a stiff example. As seen in 
    the plot, it is unstable, and gets too extreme (> 10**150) after less
    than 0.1s
    
    Task 5
    """
    t = 0
    dt = 0.001
    qn = q(np.sqrt(2), np.sqrt(3))
    options = -200000, 0.5, 20
    
    RKtvals = [0]
    RKxvals = [qn[0]]
    RKyvals = [qn[1]]
    
    while t <= 1:
        try:
            qn = MyRK3_step(f, t, qn, dt, options)
            RKxvals.append(qn[0])
            RKyvals.append(qn[1])
            t += dt
            RKtvals.append(t)
        except AssertionError: # When q becomes infinite, break the loop
            break
        
    x = np.linspace(0, 1, 50)
    y = np.sqrt(1 + np.cos(x))
    
    w = np.linspace(0, 1, 50)
    z = np.sqrt(2 + np.cos(20*w))
        
    ax1 = pylab.subplot(121)
    ax2 = pylab.subplot(122)
    ax1.plot(RKtvals, RKxvals, "-b", label="RK3")
    ax1.plot(x, y, "-r")
    ax2.plot(RKtvals, RKyvals, "-b")
    ax2.plot(w, z, "-r")
    ax1.set_ylim(1.2, 1.45)
    ax2.set_ylim(1, 1.8)
    ax1.set_xlabel("t")
    ax2.set_xlabel("t")
    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    pylab.show()
    
def stiff_GRRK3():
    """
    Function to plot the GRRK3 method for a stiff example.
    
    Task 6
    """
    t = 0
    dt = 0.005
    qn = q(np.sqrt(2), np.sqrt(3))
    options = -200000, 0.5, 20
    
    GRtvals = [0]
    GRxvals = [qn[0]]
    GRyvals = [qn[1]]
    
    while t <= 1:
        qn = MyGRRK3_step(f, t, qn, dt, options)
        GRxvals.append(qn[0])
        GRyvals.append(qn[1])
        t += dt
        GRtvals.append(t)
        
    x = np.linspace(0, 1, 50)
    y = np.sqrt(1 + np.cos(x))
    
    w = np.linspace(0, 1, 50)
    z = np.sqrt(2 + np.cos(20*w))
        
    ax1 = pylab.subplot(121)
    ax2 = pylab.subplot(122)
    ax1.plot(GRtvals, GRxvals, "-b", label="RK3")
    ax1.plot(x, y, "-r")
    ax2.plot(GRtvals, GRyvals, "-b")
    ax2.plot(w, z, "-r")
    ax1.set_xlim(0,1)
    ax2.set_xlim(0,1)
    ax1.set_ylim(1.2, 1.45)
    ax2.set_ylim(1, 1.8)
    ax1.set_xlabel("t")
    ax2.set_xlabel("t")
    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    pylab.show()

def stiff_convergence():
    """
    Function to plot the convergence for the GRRK3 method for a stiff example.
    
    Task 7
    """
    t = 0
    dt = 0.05
    qn = q(np.sqrt(2), np.sqrt(3))
    options = -200000, 0.5, 20
    
    GRyvals = [[qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]], [qn[1]]]
    GRtvals = [[0], [0], [0], [0], [0], [0], [0], [0]]
    GRdtvals = []
    
    for i in range(8):
        while t <= 1:
            qn = MyGRRK3_step(f, t, qn, dt, options)
            GRyvals[i].append(qn[1])
            t += dt
            GRtvals[i].append(t)
        GRdtvals.append(dt)
        dt /= 2
        t = 0
        qn = q(np.sqrt(2), np.sqrt(3))
        
    GRerror = []
      
    for i in range(8):
        total = 0
        for j in range(len(GRyvals[i])):
            total += abs(GRyvals[i][j] - np.sqrt(2 + np.cos(20*GRtvals[i][j])))
        GRerror.append(total*GRdtvals[i])
    
    pgr = np.polyfit(GRdtvals, GRerror, 3)
    pgr1 = np.poly1d(pgr)
    x1 = np.linspace(0, GRdtvals[0], 50)
    y1 = pgr1(x1)
    
    pylab.plot(GRdtvals, GRerror, "-g", label="GRRK3 error")
    pylab.plot(x1, y1, "--g")
    pylab.xlabel("dt")
    pylab.ylabel("1-norm error for y(t)")
    pylab.xlim(0, 0.05)
    pylab.ylim(0, 0.003)
    pylab.show()

non_stiff_plots()
non_stiff_convergence()
stiff_RK3()
stiff_GRRK3()
stiff_convergence()