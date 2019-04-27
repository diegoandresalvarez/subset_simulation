# -*- coding: utf-8 -*-
# SUBSET SIMULATION OF A SIMPLE PERFORMANCE FUNCTION
#
# ------------------------------------------------------------------------------
# CREATED BY: Diego Andrés Alvarez Marín, Juan José Sepúlveda
# MAIL:       daalvarez@unal.edu.co, jjsepulvedag@unal.edu.co
# DATE:       April 2019
# UNIVERSITY: Universidad Nacional de Colombia
# ------------------------------------------------------------------------------

from scipy.stats import multivariate_normal, norm
from subset_simulation import subsim, subsim_curve
import matplotlib.pyplot as plt
import numpy as np


# Limit state function
'''
def g_lim(x): # x será un array con dos elementos (lista con 2 elementos)
    G = 4 - x[0]/4 + np.sin(5*x[0]) - x[1] # G = 4 - y/4 + sin(5*y) - z
    return -G
d  = 2
'''
# Estimate P(X>5) given that X ~ N(mu = 0, sigma = 1) using subset simulation
# 1 - norm.cdf(5) >>>  = 2.8665e-07
def g_lim(x):
    return -(x[0] - 5)
d = 1

#%%
p0 = 0.1
pi          = multivariate_normal(mean = np.zeros(d), cov = np.identity(d))
pi_marginal = d*[norm(loc = 0, scale = 1)]

theta, g, b, pf = subsim(pi, pi_marginal, g_lim, p0, g_failure='<=0')
print("The probability of failure is", pf)

#%% curve of the probability of failure for each demand level
demand, pf_demand = subsim_curve(g, b, p0, g_failure='<=0')

plt.figure()
m = len(b) # number of intermediate failure domains
plt.plot(b, p0**np.arange(1, m+1), 'o', label='intermediate failure probabilities')
plt.semilogy(demand, pf_demand, 'b-', label='SubSim')
plt.xlabel('b (demand level)')
plt.ylabel('Probability of failure')
plt.title('Curve of the probability of failure for each demand level')
plt.grid(True, which='both')

x = np.linspace(-3,5,100)
y = 1-norm.cdf(x)
plt.semilogy(5-x,y, 'r-', label='MCS')
plt.legend()
plt.show()
#%% bye, bye!
