import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

T = 1.0  # Time horizon
num_points = 10000  # Number of time steps
dT = T / num_points  # Time step

def cauchy_path(m):
    initial_value = 0.0  # Starting value for the process
    # Simulate random Cauchy increments
    increments = np.reshape(cauchy.rvs(loc=0, scale=dT, size=num_points*m), (m, num_points))

    # Construct the path of the Cauchy process
    cauchy_path = np.cumsum(increments, axis=1)
    cauchy_path = np.insert(cauchy_path, 0, initial_value, axis = 1)  # Include the initial value
    return cauchy_path

def supC(m):
    cp = cauchy_path(m)
    supC = np.max(cp, axis=1)
    return supC


import supremumstable as ss
import time
time_0 = time.time()
ssup = ss.SupStable(1, 0)
ssup.set_params(omega=(2/3,1/3,.95*1,3,-50,12))
S = ssup.rv(10**4)
print(time.time() - time_0)
C = supC(10**4)

from scipy.stats import ks_2samp
print(ks_2samp(S, C))