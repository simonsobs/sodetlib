import numpy as np
from numpy import pi

import scipy.optimize as op
from scipy.optimize import fsolve

import pylab as pl
pl.ion()

# Note: lambda is a reserved word, so I use lamb instead

Phi0 = 2.068e-15 # magnetic flux quantum

def phieofphi(phi,lamb):
    phie = phi + lamb*np.sin(phi)
    return phie

def phisum(phi,*args):
    phie,lamb = args
    return (phieofphi(phi,lamb) - phie)

def phiofphie(phie, lamb):
    phiguess = phie
    argvals = (phie,lamb)
    phi = fsolve(phisum, phiguess, args=argvals)
    return phi

def f0ofphi(phi, f2, P, lamb):
    # Odd formulation hopefully makes variation in lamb not affect f2 or P
    f0 = f2 + (P/2) * ((1-lamb**2)/lamb) * ((lamb*np.cos(phi))/(1 + lamb*np.cos(phi)) + (lamb**2)/(1-lamb**2))
    return f0

def f0ofphie(phie, f2, P, lamb):
    phi = phiofphie(phie,lamb)
    f0 = f0ofphi(phi,f2,P,lamb)
    return f0

def f0ofI(I, I0, m, f2, P, lamb):
    phie = (I - I0)*m
    f0 = f0ofphie(phie,f2,P,lamb)
    return f0

def guessfitparams(I,f0):
    Pguess = np.max(f0) - np.min(f0)
    f2guess = (np.max(f0) + np.min(f0)) / 2.0
    I0guess = I[np.argmax(f0)]
    mguess = 13.5 #pi / np.abs( I[np.argmax(f0)] - I[np.argmin(f0)] ) # assumes 0.5 to 1.5 periods
    lambguess = 0.5
    return I0guess,mguess,f2guess,Pguess,lambguess

def lambdafit(I,f0,showplot=0):

    guessparams = guessfitparams(I,f0)
    popt, pcov = op.curve_fit(f0ofI, I, f0, guessparams)
    
    I0fit,mfit,f2fit,Pfit,lambfit = popt
    f0fit = f0ofI(I,I0fit,mfit,f2fit,Pfit,lambfit)
    
    if showplot == 1:
        pl.figure(789)
        pl.subplot(2,1,1)
        pl.cla()
        pl.plot(I*1e6,f0,'ob',I*1e6,f0fit,'--r')
        pl.xlabel('Current (uA)')
        pl.ylabel('Frequency (GHz)')
        pl.draw()
        
    
    return I0fit,mfit,f2fit,Pfit,lambfit



def quicktest(I, I0, m, f2, P, lamb):
    f0 = f0ofI(I,I0,m,f2,P,lamb)
    s = 0.05*np.random.normal(0,1,len(f0))    # RMS noise of 0.05
    f0ps = f0 + s
    
#    guessparams = guessfitparams(I,f0ps)
#    I0fit,mfit,f2fit,Pfit,lambfit = lambdafit(I,f0ps,guessparams)
#    f0fit = f0ofI(I,I0fit,mfit,f2fit,Pfit,lambfit)

    I0fit,mfit,f2fit,Pfit,lambfit = lambdafit(I,f0ps,showplot=1)
    
#    pl.figure()
#    pl.plot(I,f0,'-g',I,f0ps,'.-b',I,f0fit,'--r')

    print(lambfit)

    return f0ps
