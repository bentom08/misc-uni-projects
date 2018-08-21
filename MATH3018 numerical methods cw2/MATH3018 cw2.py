# B Taylor 11/12/2016
# MATH3018 Numerical Methods Coursework 2

"""
The Algorithm I have chosen to use for this problem is the Shooting method
ie. converting the original boundary value problem into an initial value problem,
and altering the guess for the initial value until it conforms to the boundary
value problem.

This algorithm uses the scipy function "brentq" as its root finding tool, which is
a version of the Secant method.

I believe this is a good method to use for solving this problem because it is very
fast and accurate, and the problem is easy to convert into an ivp. The bvp also
only has one solution meaning there is no chance or it converging to the "wrong"
one.

Also many of the disadvantages of the shooting method don't apply here. It fails
to satisfy the boundary conditions exactly, but this simulation is about prices
which have a finite precision, meaning the boundary conditions do not have to be
extremely precise.

The main disadvantage using this method is the two initial guesses required. Poor
guesses will cause the algorithm not to converge. I have attempted to circumvent
this somewhat by varying the initial guesses if the original guesses did not
converge until it finds one that does.



I think the mathematical problem has been posed correctly to solve the original
problem, assuming that the model provided in equation P is sufficiently accurate
for the problem.
"""

from sympy import diff, symbols, lambdify
from scipy.integrate import odeint
from scipy.optimize import brentq
import numpy as np
import pylab as plt

global conv
conv = []

def L(t, y, y1, alpha, beta):
    """
    
    The Lagrangian Equation.
    
    t: sympy symbol, time
    y: sympy symbol, y(t)
    y1: sympy symbol, the differential wrt t of y(t)
    alpha: float, the numerical value of alpha
    beta: float, the numerical value of beta
    
    Returns an equation using sympy symbols t, y and y1
    
    """
    
    assert((not np.any(np.isnan(alpha))) and np.all(np.isfinite(alpha)) and\
    np.all(np.isreal(alpha))), \
    "Alpha must be real and finite"
    assert((not np.any(np.isnan(beta))) and np.all(np.isfinite(beta)) and\
    np.all(np.isreal(beta))), \
    "Beta must be real and finite"
    assert(type(alpha) == float), \
    "Alpha must be a float"
    assert(type(beta) == float), \
    "Beta must be a float"
    
    return alpha*y1**2 + beta*(t**2-1)*y1**3 - y

def EL_func(L, t0, alpha, beta):
    """
    
    The Calculation of the Euler-Lagrange Equation.
    
    L: func, the Lagrangian Equation
    t0: float or int, time
    alpha: float, the numerical value of alpha
    beta: float, the numerical value of beta
    
    Returns the Euler Lagrange equation as a function of y1
    
    """
    
    assert(hasattr(L, '__call__')), \
    "L must be a callable function"
    assert((not np.any(np.isnan(alpha))) and np.all(np.isfinite(alpha)) and\
    np.all(np.isreal(alpha))), \
    "Alpha must be real and finite"
    assert((not np.any(np.isnan(beta))) and np.all(np.isfinite(beta)) and\
    np.all(np.isreal(beta))), \
    "Beta must be real and finite"
    assert(type(alpha) == float), \
    "Alpha must be a float"
    assert(type(beta) == float), \
    "Beta must be a float"
    assert((not np.any(np.isnan(t0))) and np.all(np.isfinite(t0)) and\
    np.all(np.isreal(t0))), \
    "t0 must be real and finite"
    assert(t0 >= 0), \
    "t0 must be greater than or equal to zero"
    assert(type(t0) == float or type(t0) == int), \
    "t0 must be a float or int"
    
    y, y1, t = symbols("y y1 t")
    
    dy = diff(L(t, y, y1, alpha, beta), y)
    dtdy1 = diff(L(t, y, y1, alpha, beta), t, y1)
    dydy1 = diff(L(t, y, y1, alpha, beta), y, 1)
    dy1sq = diff(L(t, y, y1, alpha, beta), y1, y1)

    eq = dy - dtdy1 - y1*dydy1

    eq /= dy1sq
    
    f = lambdify(t, eq)
    eq = f(t0)
    
    return eq

def ivp(z, x, eq):
    """
    
    The initial Value Problem.
    
    z: numpy array, the differential of y at 0
    x: null, needed as a second value for odeint but not used
    eq: eqauation of sympy symbols, the Euler Lagrange equation (return value of EL_func)
    
    Returns a numpy array containing a system of ODEs
    
    """
    
    assert(type(z) == np.ndarray), \
    "z must be a numpy array"
    
    y1 = symbols("y1")
    dzdt = np.zeros_like(z)
    dzdt[0] = z[1]
    dzdt[1] = eq.subs(y1, z[1])
    
    return dzdt
    
def shooting(z, eq, a, b):
    """
    
    The shooting method algorithm
    
    z: float, number adjusted to find the solution
    eq: sympy symbols, the Euler Lagrange equation (return value of EL_func)
    a: float or int, first boundary value at t = 0
    b: float or int, second boundary value at t = 1
    
    Returns the differnce between the value calculated for the ODEs and b (used to find the root)
    
    """
    
    assert(type(z) == float), \
    "z must be a float"
    assert(type(a) == float or type(a) == int), \
    "a must be a float or int"
    assert(type(b) == float or type(b) == int), \
    "b must be a float or int"
    assert((not np.any(np.isnan(a))) and np.all(np.isfinite(a)) and\
    np.all(np.isreal(a))), \
    "a must be real and finite"
    assert((not np.any(np.isnan(b))) and np.all(np.isfinite(b)) and\
    np.all(np.isreal(b))), \
    "b must be real and finite"
    assert((not np.any(np.isnan(z))) and np.all(np.isfinite(z)) and\
    np.all(np.isreal(z))), \
    "z must be real and finite"
    
    global conv
    x = [0.0, 1.0]
    z0 = [a, z]
    Z = odeint(ivp, z0, x, args = (eq,))
    phi = Z[-1, 0] - b
    conv.append(abs(phi))
    return phi
    
def main(L, a, b, alpha, beta):
    """
    
    Main Function, which plots a graph of y(t) against t for the given values.
    
    L: func, the Lagrangian Equation
    a: float or int, the first boundary value at t = 0
    b: float or int, the second boundary condition at t = 1
    alpha: float, the numerical value of alpha
    beta: float, the numerical value of beta
    
    Returns the array of phi values iterated through.
    
    """
    
    assert(hasattr(L, '__call__')), \
    "L must be a callable function"
    assert((not np.any(np.isnan(alpha))) and np.all(np.isfinite(alpha)) and\
    np.all(np.isreal(alpha))), \
    "Alpha must be real and finite"
    assert((not np.any(np.isnan(beta))) and np.all(np.isfinite(beta)) and\
    np.all(np.isreal(beta))), \
    "Beta must be real and finite"
    assert(type(alpha) == float), \
    "Alpha must be a float"
    assert(type(beta) == float), \
    "Beta must be a float"
    assert(type(a) == float or type(a) == int), \
    "a must be a float or int"
    assert(type(b) == float or type(b) == int), \
    "b must be a float or int"
    assert((not np.any(np.isnan(a))) and np.all(np.isfinite(a)) and\
    np.all(np.isreal(a))), \
    "a must be real and finite"
    assert((not np.any(np.isnan(b))) and np.all(np.isfinite(b)) and\
    np.all(np.isreal(b))), \
    "b must be real and finite"
    
    global conv
    conv = []
    n = 1.0
    while True:
        try: #Try-except loop to vary the intial guesses, if it doesn't converge
            z0 = brentq(shooting, -n, n, args = (EL_func(L, 0, alpha, beta), a, b,))
        except ValueError:
            n += 0.5
            conv = []
            continue
        break

    x = np.linspace(0.0, 1.0, 100)
    Z0_soln = [0.0, z0]
    Z_soln = odeint(ivp, Z0_soln, x, args = (EL_func(L, 0, alpha, beta),))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, Z_soln[:, 0]+1, 'g-', label = "Solution for Alpha = {0:.2f} and Beta = {1:.2f}".format(alpha, beta))
    ax.legend(loc = 1)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("y(t) Price in dimensionless units")
    ax.set_ylim(0.9, 1)
    plt.show()
    
    return conv
    
main(L, 1, 0.9, 5.0, 5.0)
conv = main(L, 1, 0.9, 7.0/4.0, 5.0)
n = range(1, len(conv)+1)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(n, conv, 'g-', label = "Convergence with Alpha = {0:.2f} and Beta = {1:.2f}".format(7.0/4.0, 5.0))
ax.legend(loc = 1)
ax.set_xlabel("Number of iterations (n)")
ax.set_ylabel("Phi")
plt.show()