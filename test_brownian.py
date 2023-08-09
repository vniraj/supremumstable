from scipy.stats import norm, ks_1samp
import numpy as np

def supB_cdf(x):
    #P(supB < x)
    c = np.ndarray(shape=(len(x),))
    for i in range(len(x)):
        if x[i] > 0:
            c[i] = 1 - 2*(1 - norm.cdf(x[i]))
        else:
            c[i] = 0
    return c

from supremumstable import SupStable
(alpha, beta) = (2, 0)
stablesup = SupStable(alpha, beta)
stablesup.set_params(omega=(2/3,1/3,.95*alpha,3,-50,12))

S = np.sort(stablesup.rv(10**4))
print(ks_1samp(S/np.sqrt(2), supB_cdf))

    