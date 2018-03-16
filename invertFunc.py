# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:33:34 2018

@author: vatle
"""

import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def G(x):
    return sc.norm.cdf(x) + sc.norm.pdf(x) / x

x = np.linspace(0.01, 7, 100)
y = G(x)

plt.figure()
plt.plot(x, y)
plt.show()

container = []

def findInflexionPoints(x, f, tol, cut):
    if cut != 0:
        y = f(x)
        yleft = y[:cut]
        yright = y[cut:]
        convexity = np.array([ 1*(np.abs((yright[i+1] - 2 * yright[i] + yright[i]) / ((x[1] - x[0])**2)) > tol) for i in range(1, len(yright) - 1)])
        #container.append(np.array([ np.abs((y[i+1] - 2 * y[i] + y[i]) / ((x[1] - x[0])**2)) for i in range(1, len(x) - 1)]))
        variation = convexity[1:] - convexity[:-1]
        variation = variation.tolist()
        idxRight = [i+1+len(yleft) for i, e in enumerate(variation) if e != 0]
        
        convexity2 = np.array([ 1*((yleft[i+1] - 2 * yleft[i] + yleft[i]) / ((x[1] - x[0])**2) > 0) for i in range(1, len(yleft) - 1)])
        container.append(np.array([ (yleft[i+1] - 2 * yleft[i] + yleft[i]) / ((x[1] - x[0])**2) for i in range(1, len(yleft) - 1)]))
        variation2 = convexity2[1:] - convexity2[:-1]
        variation2 = variation2.tolist()
        idxLeft = [i+1 for i, e in enumerate(variation2) if e != 0]
        return idxLeft + idxRight
    else:
        y = f(x)
        convexity = np.array([ 1*(np.abs((y[i+1] - 2 * y[i] + y[i]) / ((x[1] - x[0])**2)) > tol) for i in range(1, len(y) - 1)])
        #container.append(np.array([ np.abs((y[i+1] - 2 * y[i] + y[i]) / ((x[1] - x[0])**2)) for i in range(1, len(x) - 1)]))
        variation = convexity[1:] - convexity[:-1]
        variation = variation.tolist()
        idx = [i + 1 for i, e in enumerate(variation) if e != 0]
        return idx


def invertFunction(G, m, M, tol):
    axis = np.linspace(m, M, 100)
    # for decreasing function
    x = [G(m), G(M)]
    y = [m, M]
    f = [interp1d(x, y)]
    le = 0
    tle = len(x)
    cut = 0
    while le != tle:
        le = len(x)
        idx = findInflexionPoints(axis, lambda x: f[-1](G(x)), tol, cut)
        for i in idx:
            x.append(G(axis[i]))
            y.append(axis[i])
        order = np.argsort(x)
        x = np.array(x)[order].tolist()
        y = np.array(y)[order].tolist()
        if len(x) <= 3:
            cut = 0
        else:
            print(y[1])
            cut = axis.tolist().index(y[1])
        newfunc = interp1d(x, y, kind="linear")  
        f.append(newfunc)
        tle = len(x)
    return x, y, f

m = 0.01
M = 7
tol = 1
x, y, f = invertFunction(G, m, M, tol)
axis = np.linspace(m, M, 100)


plt.figure()
plt.plot(axis, axis)
#plt.scatter(y, f[-1](x))
plt.plot(axis, f[0](G(axis)))
plt.plot(axis, f[1](G(axis)))
plt.plot(axis, f[2](G(axis)))
plt.plot(axis, f[3](G(axis)))
plt.show()


plt.figure()
plt.plot(G(axis), f[-1](G(axis)))
plt.scatter(x, y)
plt.show()
    
