from supremumstable import SupStable
from positivestable import PositiveStable
from scipy.stats import ks_2samp
import numpy as np
import time

for i in range(1,10):
    (alpha, beta) = (0.8, 0.3)
    n = 10000
    S = np.array([])
    P = np.array([])

    stablesup = SupStable(alpha, beta)
    stablesup.set_params(omega=(2/3,1/3,.95*alpha,3,-50,12))

    time_0 = time.time()
    S = stablesup.rv(n)
    print(time.time() - time_0)


    stablepos = PositiveStable(alpha, beta)
    P = stablepos.rv(n)
    print('KS ', ks_2samp(S, P))

