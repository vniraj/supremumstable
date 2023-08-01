import numpy as np
from scipy import integrate
from scipy.special import gamma

class UnilateralStable:
    def __init__(self, alpha, beta = 1, generator = None):
        self.alpha = alpha
        self.beta = beta
        assert 0 < self.alpha <= 2
        assert np.abs(self.beta) == 1
        self.gen = np.random.default_rng() if generator == None else generator

    def zolotarev(self, u):
        C = self.alpha/(1-self.alpha)
        A = pow(np.sin(self.alpha*u),C)*np.sin((1-self.alpha)*u)/pow(np.sin(u),1+C)
        return A

    def cdf(self, eks):
        integral = np.zeros(np.size(eks))
        for i in range(np.size(eks)): 
            if self.beta == -1:
                temp_US = UnilateralStable(self.alpha, 1)
                integral[i] = temp_US.cdf(-eks[i])
                continue
            if self.alpha == 1:
                integral[i] = 1 if eks[i] >= 1 else 0
                continue
            C = self.alpha/(1-self.alpha)
            integral[i] = integrate.quadrature(lambda u: np.exp(-self.zolotarev(u)/np.power(np.abs(eks[i]), C))/np.pi, 0, np.pi, maxiter=1000)[0]
        if np.any(integral < 0) or np.any(integral > 1):
            print('UnilateralStable: invalid cdf: ', integral[np.where(integral < 0 or integral > 1)])
        return integral
    
    def rv(self, size=1):
        if self.alpha == 1:
            return np.ones(size)*self.beta
        c = 1/self.alpha
        ra = 1 - self.alpha
        c1 = c - 1
        gen = self.gen
        u = gen.uniform(0, np.pi, size=size)
        return self.beta * (np.sin(u * self.alpha)/np.power(np.sin(u), c)) * np.power(np.sin(u * ra)/gen.exponential(size=size), c1)
    
    def mgf(self, x):
        if self.beta == -1:
            return UnilateralStable(self.alpha).mgf(-x)
        x = np.array(x)
        if self.alpha == 1:
            return np.exp(x)
        res = np.zeros(np.size(x))
        for i in range(np.size(x)):
            if x[i] == 0:
                res[i] = 1
            elif x[i]<0:
                res[i] = np.exp(-np.power(np.abs(x[i]), self.alpha))
            else:
                res[i] = np.inf
        return res

    def mellin(self, x):
        if self.beta == -1:
            return UnilateralStable(self.alpha).mellin(-x)
        if x >= self.alpha:
            return np.inf
        return gamma(1 - x/self.alpha)/gamma(1 - x)