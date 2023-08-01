import numpy as np
from scipy.special import gamma

import unilateralstable as us

class PositiveStable:
    def __init__(self, alpha, beta, generator = None):
        self.alpha = alpha
        self.beta = beta
        assert (0 < self.alpha <= 2 and -1 <= self.beta <= 1) and not (0 < alpha <= 1 and beta == -1), "Invalid parameters"
        if alpha == 2:
            (self.alpha, self.beta, self.theta, self.rho) = (2, 0, 0, .5)
        else:
            self.theta = beta*(1 if alpha <= 1 else (alpha - 2)/alpha)
            self.rho = (1 + beta*(1 if alpha <= 1 else (alpha - 2)/alpha))/2
        self.gen = np.random.default_rng() if generator == None else generator
        self.u_stable = us.UnilateralStable(self.alpha, generator = self.gen)

    def rv(self, size=1):
        n = size
        ar = self.alpha * self.rho
        gen = self.gen
        if self.beta == 1 and self.alpha == 1:
            return 1
        elif self.beta == 1 and self.alpha < 1:
            u1 = gen.uniform(0, np.pi, size = n)
            e1 = gen.exponential(size=n)
            s1 = np.sin(u1)
            return (np.sin(self.alpha * u1) / s1) * np.power(s1 * e1 / np.sin((1 -self.alpha) * u1) , 1 - 1 /self.alpha)
        elif self.beta == -1 and self.alpha > 1:
            u1 = gen.uniform(0, np.pi, size=n)
            e1 = gen.exponential(size=n)
            s1 = np.sin(u1)
            return np.power((np.sin(self.rho * u1) / s1) * np.power((s1 * e1 / np.sin((1 - self.rho) * u1)) , (1 - 1 /self.rho)) , (-self.rho))
        else:
            u1 = gen.uniform(0, np.pi, size=n)
            u2 = gen.uniform(0, np.pi, size=n)
            e1 = gen.exponential(size=n)
            e2 = gen.exponential(size=n)
            s1 = np.sin(u1)
            s2 = np.sin(u2)
            return np.power( (np.sin(ar * u1) / s1) * np.power(s1 * e1 / np.sin((1 - ar) * u1), (1 - 1 /ar)) / ((np.sin(self.rho * u2) / s2) * np.power(s2 * e2 / np.sin((1 - self.rho) * u2), (1 - 1 /self.rho)) ), self.rho)

    def auxV2(self, x):
        a = self.alpha
        b = self.theta
        y = (np.pi/2)*x
        t = (np.pi/2)*a*b
        return np.power((np.power(np.sin(a*y + t),a)/np.cos(y)) , (1 /(1 -a))) * np.cos((a-1) * y + t)
    
    def cdf(self, x):
        if self.alpha == 1 and self.beta == 1:
            return np.where(x >= 1, 1, 0)
        m = 100
        l = 15
        delta = 1
        pir = np.pi * self.rho
        if self.alpha > 1:
            raa = 1 / (self.alpha - 1)
            a1 = 0 if x == 0 else np.abs(np.power(x, raa))
            a2 = -np.power(a1, self.alpha)
            s1 = self.rho
            s2 = 1 - self.rho
            if x < delta:
                v = np.arange(1, np.floor(l*self.alpha)+1)
                w = np.power((-1) , ((v - 1) % 2)) * gamma(v / self.alpha + .1) * np.sin(pir * v) / (v * gamma(v + 1))
                return 1 - np.sum(w * np.power(np.abs(x), -self.alpha * v))/(self.alpha*np.pi)
            else:
                nodes, weights = np.polynomial.legendre.leggauss(m)
                seq1 = self.auxV2(nodes * s1 + s2)
                return 1 - s1*(np.sum(np.exp(a2 * seq1)*weights)) / (2*self.rho)
        else:
            if x > delta:
                v = np.arange(1, np.floor(l*self.alpha)+1)
                w = np.power((-1) , ((v - 1) % 2)) * gamma(v * self.alpha + .1) * np.sin(pir * self.alpha * v) / (v * gamma(v + 1))
                return 1 - np.sum(w * np.power(np.abs(x), -self.alpha * v))/(self.alpha*np.pi)
            else:
                nodes, weights = np.polynomial.legendre.leggauss(m)
                nodes = nodes / 2 + .5
                weights = weights / 2
                nodes1 = nodes / (1 - nodes)
                nodes2 = 1 / np.power(1 - nodes, 2)
                pir = np.pi * self.rho
                mat = np.abs(x)*nodes1
                mat = self.u_stable.cdf(mat)
                cauchy_pdf = 1 / (np.pi * np.sin(pir) * (1 + np.power((nodes1 + np.cos(pir))/np.sin(pir), 2)))
                fC = nodes2 * cauchy_pdf
                cdf = np.sum(fC * mat * weights)/self.rho
                if np.any(cdf < 0) or np.any(cdf > 1):
                    print('invalid cdf: ', cdf[np.where(cdf < 0 or cdf > 1)])
                return cdf
            
    def mellin(self, x):
        if (np.real(x) > self.alpha and (self.alpha <= 1 or (self.alpha < 2 and self.beta != -1))) or np.real(x) <= -1:
            return np.inf
        if (self.alpha > 1 and self.beta == -1) and self.alpha == 2:
            return gamma(1 + x)/gamma(1 + x/self.alpha)
        return (np.sin(np.pi * self.rho * x) * gamma(1 + x)) /(self.alpha * self.rho * np.sin(np.pi * x / self.alpha) * gamma(1 + x / self.alpha))
    
    def mgf(self, x):
        l = 15
        if x == 0:
            return 1
        if self.alpha == 2:
            return 2 * np.exp(np.power(x, 2) / 4)*...