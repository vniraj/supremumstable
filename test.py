from supremumstable import SupStable
from positivestable import PositiveStable
from scipy.stats import ks_2samp
import numpy as np

(alpha, beta) = (1.5, -1)
n = 1000
S = np.array([])
P = np.array([])

stablesup = SupStable(alpha, beta)
stablesup.set_params(omega=(2/3,1/3,.95*alpha,3,-50,12))
stablepos = PositiveStable(alpha, beta)

for i in range(1):
    S_end = stablesup.rv(n, show_progress=True)
    P_end = stablepos.rv(n)
    print(i, ' KS ', ks_2samp(S_end, P_end))
    S = np.append(S, S_end)
    P = np.append(P, P_end)

print('KS Total',ks_2samp(S, P))


