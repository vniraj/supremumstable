import numpy as np
from scipy.stats import norm
from scipy.special import gamma
import scipy.integrate as integrate
import unilateralstable as us

class PosStable:
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
        self.u_stable = us.UniStable(self.alpha, generator = self.gen)

    def params(self):
        return (self.alpha, self.beta, self.theta, self.rho)
    
    def minimum(self):
        return 1 if self.beta == 1 and self.alpha == 1 else 0
    
    def maximum(self):
        return 1 if self.beta == 1 and self.alpha == 1 else np.inf
    
    def insupport(self, x):
        return x == 1 if self.beta == 1 and self.alpha == 1 else x >= 0

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
    
    def cdf(self, x, degree = 1000):
        if self.alpha == 1 and self.beta == 1:
            return np.where(x >= 1, 1, 0)
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
                return 1 - (integrate.fixed_quad(lambda u: np.exp(a2 * self.auxV2(u*s1 + s2)), -1, 1, n = degree)[0]) / 2
        else:
            if x > delta:
                v = np.arange(1, np.floor(l*self.alpha)+1)
                w = np.power((-1) , ((v - 1) % 2)) * gamma(v * self.alpha + .1) * np.sin(pir * self.alpha * v) / (v * gamma(v + 1))
                return 1 - np.sum(w * np.power(np.abs(x), -self.alpha * v))/(self.alpha*np.pi)
            else:
                f1 = lambda u: u/(1-u)
                f2 = lambda u: 1/np.power(1-u, 2)
                mat = lambda u: self.u_stable.cdf(np.abs(x)*f1(u))
                cauchy_pdf = lambda u: 1 / (np.pi * np.sin(pir) * (1 + np.power((f1(u) + np.cos(pir))/np.sin(pir), 2)))
                fC = lambda u: f2(u) * cauchy_pdf(u)
                cdf = integrate.fixed_quad(lambda u: fC(u) * mat(u), 0, 1, n = degree)[0]/self.rho
                if np.any(cdf < 0) or np.any(cdf > 1):
                    print('invalid cdf: ', cdf[np.where(cdf < 0 or cdf > 1)])
                return cdf
            
    def pdf(self, x, degree = 1000):
        if self.alpha <= 1 and self.beta == 1:
            return 1 if x == 1 else 0
        if x < 0:
            return 0
        elif x == 0:
            return gamma(1 + 1/self.alpha) * np.sin(np.pi*self.rho)/(self.rho * np.pi)
        delta = 1
        l = 15
        pir = np.pi * self.rho
        pir1 = np.pi * (1 - self.rho)
        if self.alpha > 1:
            a0 = self.alpha(np.abs(self.alpha - 1)*2)
            raa = 1/(self.alpha - 1)
            a1 = 0 if x == 0 else np.power(np.abs(x), raa)
            a2 = -np.power(a1, self.alpha)
            s1 = self.rho
            s2 = 1 - self.rho
            a01 = a0*s1
            a02 = a0*s2
            if x < delta:
                v = np.arange(1, np.floor(l*self.alpha)+1)
                w = np.power((-1) , ((v - 1) % 2)) * gamma(v / self.alpha + 1) * np.sin(pir * v) / (np.pi * gamma(v + 1))
                return np.sum(w * np.power(np.abs(x), v-1))
            else:
                f1 = lambda u: self.auxV2(u*s1 + s2)
                f2 = lambda u: f1(u)*np.exp(a2 * f1(u))
                return a01*a1*(integrate.fixed_quad(f2, -1, 1, n = degree)[0])
        else:
            if x > delta:
                v = np.arange(1, np.floor(l*self.alpha)+1)
                w = np.power((-1) , ((v - 1) % 2)) * gamma(v * self.alpha + 1) * np.sin(pir * self.alpha * v) / gamma(v + 1)

                return np.sum(w * np.power(np.abs(x), -self.alpha*v-1))/np.pi
            else:
                #range = [0, 1]
                f1 = lambda u: u/(1-u)
                f2 = lambda u: u/np.power(1 - u, 3)
                pir = np.pi * self.rho
                mat = lambda u: self.u_stable.pdf(np.abs(x)*f1(u))
                cauchy_pdf = lambda u: 1 / (np.pi * np.sin(pir) * (1 + np.power((u + np.cos(pir))/np.sin(pir), 2)))
                fC = lambda u: f2(u) * cauchy_pdf(f1(u)) * mat(u)
                return integrate.fixed_quad(fC, 0, 1, n = degree)[0]/self.rho


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
            return 2 * np.exp(np.power(x, 2) / 4)*norm.cdf(x/2, 0, np.sqrt(2))
        if self.beta == -1 and x >= -1:
            v = np.arange(1, np.floor(l*self.alpha)+1)
            w = 1/gamma(v + 1)
            return np.sum(w * np.power(x, v))
        
        #range = [0, 1]
        f2 = lambda u: 1/np.power(1 - u, 2)
        f1 = lambda u: u/(1-u)
        pir = np.pi * self.rho
        cauchy_pdf = lambda u: 1 / (np.pi * np.sin(pir) * (1 + np.power((u + np.cos(pir))/np.sin(pir), 2)))
        fC = lambda u: f2(u) * cauchy_pdf(f1(u))
        if self.beta == -1:
            mat = lambda u: np.exp(-np.power(np.abs(x)*f1(u), self.alpha))
            return integrate.fixed_quad(lambda u: fC(u) * mat(u), 0, 1, n = 1000)[0]/self.rho
        else:
            if x > 0:
                return np.inf
            else:
                mat = lambda u: np.exp(-np.power(np.abs(x)*f1(u), self.alpha))
                return integrate.fixed_quad(lambda u: fC(u) * mat(u), 0, 1, n = 1000)[0]/self.rho

    def mean(self):
        if self.alpha <= 1:
            return np.inf
        return np.sin(np.pi*self.rho)/(self.alpha*self.rho*np.sin(np.pi/self.alpha)*gamma(1 + 1/self.alpha))
    
    def var(self):
        if self.alpha < 2 and self.beta != -1:
            return np.inf
        elif self.alpha == 1:
            if np.abs(self.beta) != 1:
                return np.inf
            else:
                return 0
        else:
            return 2 / gamma(1 + 2/self.alpha) - 1/np.power(gamma(1 + 1/self.alpha), 2)
        