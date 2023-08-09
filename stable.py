import numpy as np
from scipy.special import gamma

class Stable:
    def __init__(self, alpha, beta, generator = None):
        self.gen = np.random.default_rng() if generator == None else generator
        self.alpha = alpha
        self.beta = beta
        assert 0 < self.alpha <= 2
        assert -1 <= self.beta <= 1
        if alpha == 2:
            (self.alpha, self.beta, self.theta, self.rho) = (2, 0, 0, .5)
        else:
            self.theta = beta*(1 if alpha <= 1 else (alpha - 2)/alpha)
            self.rho = (1 + beta*(1 if alpha <= 1 else (alpha - 2)/alpha))/2

    def mean(self):
        if self.alpha <= 1:
            return np.inf
        cr = 1 - self.rho
        return (np.sin(np.pi*self.rho) - np.sin(np.pi*cr))/(self.alpha*np.sin(np.pi/self.alpha)*gamma(1 + 1/self.alpha))
    
    def mellin(self, x):
        if (np.real(x) >= self.alpha and (self.alpha <= 1 or (self.alpha < 2 and self.beta != -1))) or np.real(x) <= -1:
            return np.inf
        if (self.alpha > 1 and self.beta == -1) or self.alpha == 2:
            return self.rho * gamma (1+x)/gamma(1 + x/self.alpha)
        return self.rho * (np.sin(np.pi*self.rhp*x)*gamma(1+x))/(self.alpha*self.rho*np.sin(np.pi*x/self.alpha)*gamma(1 + x/self.alpha))
    
    def rv(self, size):
        zeta = -self.beta * np.tan(np.pi*self.alpha/2)
        xi = np.arctan(-zeta)/self.alpha if self.alpha != 1 else np.pi/2
        U = self.gen.uniform(-np.pi/2, np.pi/2, size=size)
        W = self.gen.exponential(size=size)
        if self.alpha == 1:
            return (1/xi)*((np.pi/2 + self.beta * U)*np.tan(U) - self.beta*np.log((np.pi/2*W*np.cos(U))/(np.pi/2 + self.beta*U)))
        else:
            return np.power(1+np.power(zeta, 2), 1/(2*self.alpha))*np.sin(self.alpha*(U + xi))/(np.power(np.cos(U), 1/self.alpha))*np.power(np.cos(U - self.alpha*(U + xi))/W, (1-self.alpha)/self.alpha)