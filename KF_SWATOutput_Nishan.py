# Kalman filter example demo in Python

# Reference "An Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer Science, TR 95-041,
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

# Coded by Nishan Kumar Biswas
# Grdauate Research Assistant, Department of CEE
# University of Washington, Seattle,USA
# nbiswas@uw.edu, http://students.washington.edu/nbiswas/
# Github clone version 1.0.0.1

import numpy as np
import matplotlib.pyplot as plt

# for making figure
plt.rcParams['figure.figsize'] = (10, 8)

# Reading SWAT Model Output file
observations = []
simulations = []
swatmodelresultfilepath = open('best_sim.txt', 'r')
modelresultfile = swatmodelresultfilepath.readlines()
for i in xrange(len(modelresultfile)-3):
    element = modelresultfile[i+3].split()
    if(len(element)==2):
        #According to SWAT Model result file
        observations.append(float(element[0]))
        simulations.append(float(element[1]))
    
# intial parameters
n_iter = len(observations)
size = (n_iter,) # size of array

Q = 1e-5 # process variance

# allocate space for arrays
xbar=np.zeros(size)      # a posteri estimate of x
P=np.zeros(size)         # a posteri error estimate
xbarminus=np.zeros(size) # a priori estimate of x
Pminus=np.zeros(size)    # a priori error estimate
K=np.zeros(size)         # gain or blending factor

R = 0.03**2 # estimate of measurement variance, change to see effect

# intial guesses
xbar[0] = 0.0
P[0] = 1.0

for k in range(1,n_iter):
    # time update
    xbarminus[k] = xbar[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xbar[k] = xbarminus[k]+K[k]*(simulations[k]-xbarminus[k])
    P[k] = (1-K[k])*Pminus[k]

plt.figure()
plt.plot(observations,'r',label='Pure Observations')
plt.plot(simulations,'k+',label='Model Simulations')
plt.plot(xbar,'b-',label='Posteri Estimates')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Voltage')

plt.figure()
valid_iter = range(1,n_iter) # Pminus not valid at step 0
plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('$(Voltage)^2$')
plt.setp(plt.gca(),'ylim',[0,.01])
plt.show()
