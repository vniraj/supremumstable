# SupStable

SupStable is a Python Module for exact simulation of Suprema of Stable Processes by implementing the main algorithm from [[1]](#1). It also has additional functions related to Stable, Unilateral Stable, and Positive Stable distributions. 

## Usage
Copy the files supremumstable.py, positivestable.py, unilateralstable.py, and stable.py in the project folder.
To generate sample of size 100 of supremum of a stable process with parameters (alpha, beta)
```python
from supremumstable import SupStable

(alpha, beta) = (2, 0)
SupS = SupStable(2,0)
sample = SupS.rv(100)
```

## Tests

The following tests can be done to verify the validity of the algorithm, based on the Kolmogorov-Smirnov 2-sample (or 1-sample) Test.
### Brownian Motion
For (alpha, beta) = (2, 0), the stable process aligns with the Brownian motion[[3]](#3). Thus, SupStable is tested as:
```python
from scipy.stats import norm, ks_1samp
import numpy as np

def supB_cdf(x):
    #P(supB < x)
    c = 1 - 2*(1 - norm.cdf(x))
    c[np.where(x < 0)] = 0
    return c

from supremumstable import SupStable
(alpha, beta) = (2, 0)
gen = np.random.Generator(np.random.PCG64(000)) #seed=000
ssup = SupStable(alpha, beta, generator=gen) #to set seed, a generator with the seed must be passed

S = ssup.rv(10**4)
print(ks_1samp(S/np.sqrt(2), supB_cdf)) #pvalue = 0.957 
```
### Cauchy Process

For (alpha, beta) = (1, 0), the stable process aligns with a Cauchy process.[[3]](#3)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

T = 1.0  # Time horizon
num_points = 10000  # Number of time steps
dT = T / num_points  # Time step

def cauchy_process(m):
    initial_value = 0.0 
    increments = np.reshape(cauchy.rvs(loc=0, scale=dT, size=num_points*m), (m, num_points))
    cauchy_path = np.cumsum(increments, axis=1)
    cauchy_path = np.insert(cauchy_path, 0, initial_value, axis = 1)
    return cauchy_path

def supC(m):
    cp = cauchy_process(m)
    supC = np.max(cp, axis=1)
    return supC

from supremumstable import SupStable
ssup = SupStable(1, 0, generator= np.random.Generator(np.random.PCG64(1234)))
S = ssup.rv(10**4)
C = supC(10**4)

from scipy.stats import ks_2samp
print(ks_2samp(S, C)) #pvalue = 0.863
```

### Spectrally Negative Infinite Variation Stable Process
For beta = -1, alpha > 1, the distribution of SupStable aligns with PosStable, according to Theorem 3.1 in [[2]](#2)

```python
from supremumstable import SupStable
from positivestable import PosStable
from scipy.stats import ks_2samp
import numpy as np

(alpha, beta) = (1.5, -1)
n = 10000
ssup = SupStable(alpha, beta, generator = np.random.Generator(np.random.PCG64(1234)))

S = ssup.rv(n)

spos = PosStable(alpha, beta)
P = spos.rv(n)
print(ks_2samp(S, P)) #pvalue = 0.834
```

## Included Classes
In the below classes, the parameters to the constructor: `alpha`, `beta` are such that `(alpha,beta) ∈ (0,2]×[-1,1]-[0,1]×{-1}` (based on Zolotarev's (C) form of parametrization [[3]](#3).) <br>
Passing `None` to `generator` initialises it with `numpy.random.default_rng()`. 

### supremumstable.SupStable
```python
SupStable(alpha, beta, generator = None)
```
The following functions are supported:
1. `params()` <br>
returns `(alpha, beta, theta, rho)` parameters of the stable variable

2. `all_params()` <br>
returns `(alpha, beta, theta, rho, d, delta, gamma, kappa, Delta_0, m_star, eta)` parameters of algorithm 9 in [[1]](#1) used to generate samples of SupStable

3. `minimum()` <br>
returns minimum value in the support of the supremum of stable process

4. `maximum()` <br>
returns maximum value in the support of the supremum of stable process

5. `insupport(x)` <br>
returns `True` if `x` is in the support otherwise `False`

6. `mean()` <br>
returns mean/expectation of supremum of stable process

6. `calc_eta()` <br>
calculates value of eta for algorithm 9 in [[1]](#1)

7. `set_params(omega, eta = None)` <br>
set parameters `omega = (d, delta, gamma, kappa, Delta_0, m_star)` required for algorithm 9 in [[1]](#1). `eta` is calculated using `calc_eta()` if `None` is passed.

8. `calc_params()` <br>
calculates and sets parameters using values given in [[1]](#1)

9. `print_params()` <br>
prints algorithm parameters

10. `rv(size=1, show_progress=False)` <br>
returns samples of SupStable of size `size` using `algorithm_9()`
### stable.Stable
```python
Stable(alpha, beta, generator = None)
```
The following functions are supported:
1. `params()` <br>
returns `(alpha, beta, theta, rho)` parameters of the stable variable

2. `minimum()` <br>
returns minimum value in the support of the stable variable

3. `maximum()` <br>
returns maximum value in the support of the stable variable

4. `insupport(x)` <br>
returns `True` if `x` is in the support otherwise `False`

5. `mean()` <br>
returns mean/expectation of the stable variable

6. `mellin(x)` <br>
returns mellin transform of the stable variable evaluated at `x`

7. `rv(size=1)` <br>
returns a sample of size `size` of the stable variable using the CMS method [[4]](#4)

8. `pdf(x)` <br>
returns probability density function of the stable variable evaluated at `x`

9. `cdf(x)` <br>
returns cumulative density function of the stable variable evaluated at `x`

10. `mgf(x)` <br>
returns moment generating function of the stable variable evaluated at `x`

11. `var()` <br>
returns variance of the stable variable



### unilateralstable.UniStable
`beta` is `1` or `-1` and `alpha ∈ (0,2]` in:
```python
UniStable(alpha, beta = 1, generator = None)
```
The following functions can be thought of as their counterparts in `stable.Stable` applied to the Unilateral Stable distribution: <br>
`minimum`, `params`, `maximum`, `insupport`, `pdf`, `cdf`, `rv`, `mgf`, and `mellin`

### positivestable.PosStable
`(alpha,beta) ∈ (0,2]×[-1,1]-[0,1]×{-1}` in:
```python
PosStable(alpha, beta, generator = None)
```
The following functions can be thought of as their counterparts in `stable.Stable` applied to the Unilateral Stable distribution: <br>
`minimum`, `params`, `maximum`, `insupport`, `pdf`, `cdf`, `rv`, `mgf`, and `mellin`



## References
<a id="1">[1]</a> 
Cázares, J.I.G., Mijatović, A. and Bravo, G.U. (2019) Exact simulation of the extrema of stable processes, arXiv.org. Available at: https://arxiv.org/abs/1806.01870

<a id = "2">[2]</a>
Michna, Z. (2013) Explicit formula for the supremum distribution of a spectrally negative stable process, Project Euclid. Available at: https://doi.org/10.1214/ECP.v18-2236

<a id = "3">[3]</a>
Devroye, L. and James, L. (2014) On simulation and properties of the Stable Law - Statistical Methods &amp; Applications, SpringerLink. Available at: https://link.springer.com/article/10.1007/s10260-014-0260-0

<a id = "4">[4]</a>
J. M. Chambers, C. L. Mallows & B. W. Stuck (1976) A Method for Simulating Stable Random Variables, Journal of the American Statistical Association, 71:354, 340-344, DOI: 10.1080/01621459.1976.10480344


## License

[MIT](https://choosealicense.com/licenses/mit/)
