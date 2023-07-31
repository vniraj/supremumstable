import numpy as np
from scipy import integrate

class UnilateralStable:
    def __init__(self, alpha, beta = 1):
        self.alpha = alpha
        self.beta = beta
        assert 0 < self.alpha <= 2
        assert np.abs(self.beta) == 1

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
            print('invalid cdf: ', integral[np.where(integral < 0 or integral > 1)])
        return integral
