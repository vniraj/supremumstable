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

    def minimum(self):
        return self.beta if self.alpha == 1 else (0 if self.beta == 1 else -np.inf)

    def params(self):
        return (self.alpha, self.beta)
    
    def maximum(self):
        return self.beta if self.alpha == 1 else (np.inf if self.beta == 1 else 0)
    
    def insupport(self, x):
        return x == self.beta if self.alpha == 1 else x*self.beta >= 0

    def zolotarev(self, u):
        C = self.alpha/(1-self.alpha)
        A = np.power(np.sin(self.alpha*u),C)*np.sin((1-self.alpha)*u)/pow(np.sin(u),1+C)
        return A

    def pdf(self, x, degree = 1000):
        if self.beta == -1:
            return UnilateralStable(self.alpha, 1).pdf(-x)
        if x < 0:
            return 0
        elif x == 0:
            return gamma(1 + 1/self.alpha) * np.sin(np.pi*self.rho)/(self.rho * np.pi)
        c = 1/self.alpha - 1
        C = self.alpha/(1-self.alpha)
        C1 = -C - 1
        integral = integrate.fixed_quad(lambda u: -self.zolotarev(u)*np.exp(-self.zolotarev(u)/np.power(x, C)), 0, np.pi, n = degree)[0]
        integral = integral/np.pi * np.power(x, C1)/C1
        return integral

    def cdf(self, eks, degree = 1000):
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
            integral[i] = (1/np.pi)*integrate.fixed_quad(lambda u: np.exp(-self.zolotarev(u)/np.power(np.abs(eks[i]), C)), 0, np.pi, n = degree)[0]
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
