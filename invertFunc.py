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
    xlist = [y]
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
        xlist.append(y)
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
    return x, y, f, xlist

def derivatives(x, f):
    y = f(x)
    first = [(y[i] - y[i-1]) / (x[1] - x[0]) for i in range(1, len(x))]   
    second = [(y[i+1] - 2 * y[i] + y[i-1]) / (x[1] - x[0])**2 for i in range(1, len(x) - 1)]  
    third = [ (y[i+1] - 3 * y[i] + 3 * y[i-1] - y[i-2]) / (x[1] - x[0])**3 for i in range(2, len(x) - 1)]
    return first, second, third

def detectInflexionv1(x, f, index):
    # Not used
    y = f(x)
    first = [(y[i] - y[i-1]) / (x[1] - x[0]) for i in range(1, len(x))]
    if len(index) == 2:
        aboveMean = np.array([ 1*(d >= np.mean(first)) for d in first])
        variation = aboveMean[1:] - aboveMean[:-1]
        idx = [i+1 for i, e in enumerate(variation) if e != 0]
    else:
        idx = []
        for i in range(0, len(index) - 1):
            aboveMean = np.array([ 1*(first[j] >= np.mean(first[index[i]:index[i+1]])) for j in range(index[i], index[i+1])])
            variation = aboveMean[1:] - aboveMean[:-1]
            idx += [j + 1 + index[i] for j, e in enumerate(variation) if e != 0]
    return idx

def detectInflexionv2(x, f, index):
    y = f(x)
    first = [(y[i] - y[i-1]) / (x[1] - x[0]) for i in range(1, len(x))]
    if len(index) == 2:
        aboveMean = np.array([ 1*(d >= np.mean(first)) for d in first])
        variation = aboveMean[1:] - aboveMean[:-1]
        idx = [i+1 for i, e in enumerate(variation) if e != 0]
    else:
        aboveMean = np.array([ 1*(first[j] >= np.mean(first[index[-2]:index[-1]])) for j in range(index[-2], index[-1])])
        variation = aboveMean[1:] - aboveMean[:-1]
        idx = [j + 1 + index[-2] for j, e in enumerate(variation) if e != 0]
    return idx

def detectInflexionv3(x, f, index):
    y = f(x)
    first = [(y[i] - y[i-1]) / (x[1] - x[0]) for i in range(1, len(x))]
    if len(index) == 2:
        aboveMean = np.array([ 1*(d >= np.mean(first)) for d in first])
        variation = aboveMean[1:] - aboveMean[:-1]
        idx = [i+1 for i, e in enumerate(variation) if e != 0]
    else:
        idx = []
        for i in range(0, len(index) - 1):
            error = np.sqrt(np.mean((y[index[i]:index[i+1]] - x[index[i]:index[i+1]])**2))
            if error >= 0.02:
                aboveMean = np.array([ 1*(first[j] >= np.mean(first[index[i]:index[i+1]])) for j in range(index[i], index[i+1])])
                variation = aboveMean[1:] - aboveMean[:-1]
                idx += [j + 1 + index[i] for j, e in enumerate(variation) if e != 0]
    return idx

def detectInflexionv4(x, f, index):
    y = f(x)
    first = [(y[i] - y[i-1]) / (x[1] - x[0]) for i in range(1, len(x))]
    if len(index) == 2:
        aboveMean = np.array([ 1*(d >= np.mean(first)) for d in first])
        variation = aboveMean[1:] - aboveMean[:-1]
        idx = [i+1 for i, e in enumerate(variation) if e != 0]
    else:
        aboveMean = np.array([ 1*(first[j] >= np.mean(first[index[0]:index[1]])) for j in range(index[0], index[1])])
        variation = aboveMean[1:] - aboveMean[:-1]
        idx = [j + 1 + index[0] for j, e in enumerate(variation) if e != 0]
    return idx

def invertFunction2(G, m, M):
    axis = np.linspace(m, M, 100)

    x = [G(m), G(M)]
    y = [m, M]
    index = [0, len(axis) - 1]
    f = [interp1d(x, y)]

    # First step: built the curve from the beginning to the end
    for i in range(15):
        idx = detectInflexionv2(axis, lambda x: f[-1](G(x)), index)
        index += idx
        index = np.sort(index).tolist()
        for i in idx:
            x.append(G(axis[i]))
            y.append(axis[i])
            
        order = np.argsort(x)
        xlist.append(y)
        x = np.array(x)[order].tolist()
        y = np.array(y)[order].tolist()

        newfunc = interp1d(x, y, kind="linear")  
        f.append(newfunc)
        
    # Second step: Add mid points to improve the accuracy
    for i in range(3):
        idx = detectInflexionv3(axis, lambda x: f[-1](G(x)), index)
        index += idx
        index = np.sort(index).tolist()
        for i in idx:
            x.append(G(axis[i]))
            y.append(axis[i])
            
        order = np.argsort(x)
        xlist.append(y)
        x = np.array(x)[order].tolist()
        y = np.array(y)[order].tolist()

        newfunc = interp1d(x, y, kind="linear")  
        f.append(newfunc)    
    return index, x, y, f

def invertFunction3(G, m, M):
    axis = np.linspace(m, M, 100)

    x = [G(m), G(M)]
    y = [m, M]
    index = [0, len(axis) - 1]
    f = [interp1d(x, y)]

    # First step: built the curve from the beginning to the end
    for i in range(15):
        idx = detectInflexionv4(axis, lambda x: f[-1](G(x)), index)
        index += idx
        index = np.sort(index).tolist()
        for i in idx:
            x.append(G(axis[i]))
            y.append(axis[i])
            
        order = np.argsort(x)
        xlist.append(y)
        x = np.array(x)[order].tolist()
        y = np.array(y)[order].tolist()

        newfunc = interp1d(x, y, kind="linear")  
        f.append(newfunc)
        
    # Second step: Add mid points to improve the accuracy
    for i in range(3):
        idx = detectInflexionv3(axis, lambda x: f[-1](G(x)), index)
        index += idx
        index = np.sort(index).tolist()
        for i in idx:
            x.append(G(axis[i]))
            y.append(axis[i])
            
        order = np.argsort(x)
        xlist.append(y)
        x = np.array(x)[order].tolist()
        y = np.array(y)[order].tolist()

        newfunc = interp1d(x, y, kind="linear")  
        f.append(newfunc)    
    return index, x, y, f


m = 0.01
M = 7
tol = 1
index, x, y, f = invertFunction2(G, m, M)
axis = np.linspace(m, M, 100)

plt.figure()
plt.plot(axis, axis)
plt.plot(axis, f[-1](G(axis)))
plt.grid()
plt.show()

plt.figure()
plt.plot(G(axis), f[-1](G(axis)))
plt.grid()
plt.show()


m2 = -7
M2 = -0.01
index2, x2, y2, f2 = invertFunction3(G, m2, M2)
axis2 = np.linspace(m2, M2, 100)

plt.figure()
plt.plot(axis2, axis2)
plt.plot(axis2, f2[-1](G(axis2)))
plt.grid()
plt.show()

plt.figure()
plt.plot(G(axis2), f2[-1](G(axis2)))
plt.grid()
plt.show()


# Export the interpolation points in excel
xlist = [repr(i) for i in x]
ylist = [repr(i) for i in y]

data = pd.DataFrame({'x': xlist, 'y':ylist})
data.to_csv("M:\Vathana\inthemoney.csv")

xlist2 = [repr(i) for i in x2]
ylist2 = [repr(i) for i in y2]

data2 = pd.DataFrame({'x': xlist2, 'y':ylist2})
data2.to_csv("M:\Vathana\outhemoney.csv")
